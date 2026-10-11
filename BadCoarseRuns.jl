# BadCoarseRuns.jl
#
# The poor-coarse control: Lorenz-63 at ΔT = 1 with fine RK4 (h = 1e-4) and explicit
# Euler coarse chunks (H = 0.01 or 0.02), at N = 16 and 128 with eight replicates,
# under the Lyapunov and secant rules and coarse-only. BadCoarseTables.jl turns the
# records into tab:bad_coarse and tab:bad_coarse_full.
#
# Each replicate perturbs the prescribed start twice and spins both up for 40 time
# units; both coarse steps use the same starts. The first start gives the serial
# fine reference and the runs compared with it, the second an independent fine
# baseline. Trajectories are sampled at the chunk-local times 0.02, 0.04, …, 1 and
# compared with the reference by the mean marginal W₁. Runs that fail are kept, with
# W₁ = NaN.
#
# Needs Random, Serialization, Statistics, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto BadCoarseRuns.jl
#          An interrupted run resumes from Records/bad_coarse_partial.jls.
# Writes:  Records/bad_coarse.jls (96 records)

using Random, Serialization, Statistics
using NSDERungeKutta, NSDETimeParallel

h, Hs, interval = 1e-4, (0.01, 0.02), 0.02  # fine step, coarse steps, sampling interval
Ns, replicates, rules = (16, 128), 8, ("coarse", "psi_lyap", "psi_lip")
λ, ϵ = 0.9056, 1e-9  # nominal Lyapunov exponent and tolerance
prescribed = [2.0, 3.0, -14.0]
ivp(u0, T = 1.0) = Lorenz(copy(u0), (0.0, T))
fine = chunk_solver(RK4(h = h), 1.0)
unit_chunk(u, solver) = solve(ivp(u), solver)

# The states every `stride` steps of each chunk, without its start: one row per sample.
samples(chunks, stride) = permutedims(stack(s for c in chunks for s in c.u[1 + stride:stride:end]))

# The samples of a serial run of N chunks from u0.
function serial(u0, N, solver, stride)
    chunks = []
    for _ in 1:N
        push!(chunks, unit_chunk(isempty(chunks) ? u0 : chunks[end].u[end], solver))
    end
    return samples(chunks, stride)
end

function spin_up(u)
    for _ in 1:40
        u = unit_chunk(u, fine).u[end]
    end
    return u
end

# Perturbed starts in the order the archived records drew them: for each N and
# replicate, the reference start, then the baseline start. PrefixRuns.jl rebuilds
# the first N = 128 reference start from the same draws.
rng = MersenneTwister(20260911)
starts = Dict((N, rep) => [prescribed .+ 0.5 .* (rand(rng, 3) .- 0.5) for _ in 1:2] for N in Ns for rep in 1:replicates)

# Mean over coordinates of the exact empirical W₁ between samples of equal size.
mean_w1(X, Y) = mean(mean(abs.(sort(x) .- sort(y))) for (x, y) in zip(eachcol(X), eachcol(Y)))

function parareal(u0, N, rule, coarse)
    weight = rule == "psi_lyap" ? FixedWeight(exp(λ)) : SecantWeight()
    criterion = WeightedDefect(ϵ = ϵ, weight = weight)
    return solve(ivp(u0, Float64(N)), Parareal(RK4(h = h), coarse; N, criterion))
end

# One replicate at both coarse steps: a record per step and rule.
function replicate(N, rep)
    stride = round(Int, interval / h)
    u0, u1 = spin_up.(starts[(N, rep)])
    X = serial(u0, N, fine, stride)  # reference
    basew1 = mean_w1(X, serial(u1, N, fine, stride))  # independent fine baseline
    rows = []
    for H in Hs, rule in rules
        if rule == "coarse"
            Y = serial(u0, N, chunk_solver(Euler(h = H), 1.0), round(Int, interval / H))
            outcome = (fine_batches = 0, status = "coarse")
        else
            sol = parareal(u0, N, rule, Euler(h = H))
            d = sol.diagnostics
            accepted = d.status in (:converged, :exact)
            U = boundaryvalues(sol.lastiterate)
            Y = accepted ? samples([unit_chunk(U[n], fine) for n in 1:N], stride) : nothing
            outcome = (fine_batches = d.fine_batches, status = string(d.status))
        end
        w1 = Y !== nothing && all(isfinite, Y) ? mean_w1(X, Y) : NaN
        push!(rows, (; N, rep, criterion = rule, H, w1, basew1, outcome...))
    end
    return rows
end

records_dir = joinpath(@__DIR__, "Records")
partial_file = joinpath(records_dir, "bad_coarse_partial.jls")
mkpath(records_dir)
isfile(joinpath(records_dir, "bad_coarse.jls")) && error("Records/bad_coarse.jls already exists; delete it to recompute.")

jobs = [(N, rep) for N in Ns for rep in 1:replicates]
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
todo = sort([job for job in jobs if !haskey(done, job)]; by = first, rev = true)  # longest first

# Hand items to the threads one at a time, in order, from a shared queue.
function threaded_foreach(f, items)
    queue = Channel{eltype(items)}(Inf)
    foreach(item -> put!(queue, item), items)
    close(queue)
    @sync for _ in 1:Threads.nthreads()
        Threads.@spawn for item in queue
            f(item)
        end
    end
end

guard = ReentrantLock()
threaded_foreach(todo) do job
    outcome = replicate(job...)
    lock(guard) do
        done[job] = outcome
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$(length(jobs)): N=$(job[1]) replicate $(job[2])")
    end
end

records = [r for job in jobs for r in done[job]]
serialize(joinpath(records_dir, "bad_coarse.jls"), sort(records; by = r -> (r.H, r.N, r.rep)))
rm(partial_file)

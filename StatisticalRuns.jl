# StatisticalRuns.jl
#
# Replicated statistical comparisons for Lorenz-63 and Lorenz-96 at ΔT = 1, with
# RK4 fine (h = 1e-4) and coarse (H = 1e-2) chunks, N = 16, 32, 64, 128 and eight
# replicates. StatisticalTables.jl turns the records into tab:w1_baseline,
# tab:statistics_full, tab:observables_full and tab:sensitivity_summary, and
# StatisticalFigures.jl into fig:statistics.
#
# Each replicate perturbs the prescribed start twice and spins both up for 40 time
# units. The first gives the serial fine reference, the coarse-only run and the
# Parareal runs under the Lyapunov and secant rules; the second gives an
# independent fine run as a sampling baseline. Every trajectory is sampled at the
# chunk-local times 0.02, 0.04, …, 1 (0.05, …, 1 for Lorenz-96) and compared with
# the reference by the mean marginal W₁, a sliced W₁ over 16 fixed directions, and
# the errors in four means and in the lag-0.5 autocorrelation of x₁. At N = 16 the
# first four replicates also rerun the secant rule over tolerances and guards.
#
# Needs Random, Serialization, SHA, Statistics, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto StatisticalRuns.jl
#          An interrupted run resumes from Records/statistics_partial.jls.
# Writes:  Records/statistics.jls (192 records), Records/sensitivity.jls (72) and
#          Records/portraits.jls (the samples drawn in fig:statistics)

using Random, Serialization, SHA, Statistics
using NSDERungeKutta, NSDETimeParallel

h, H = 1e-4, 1e-2  # fine and coarse steps; every chunk has unit length
Ns, replicates = (16, 32, 64, 128), 8
rules = ("coarse", "psi_lyap", "psi_lip")
# Prescribed start, nominal Lyapunov exponent λ, sampling interval and seed of the start perturbations.
problems = Dict(
    "lorenz" => (start = [2.0, 3.0, -14.0], λ = 0.9056, interval = 0.02, seed = 20260909),
    "lorenz96" => (start = [8.0 + (i == 20 ? 0.01 : 0.0) for i in 1:40], λ = 1.68, interval = 0.05,
                   seed = 20260910),
)
ivp(name, u0, T = 1.0) = name == "lorenz" ? Lorenz(copy(u0), (0.0, T)) : Lorenz96(copy(u0), (0.0, T))
fine, coarse = chunk_solver(RK4(h = h), 1.0), chunk_solver(RK4(h = H), 1.0)
unit_chunk(name, u, solver) = solve(ivp(name, u), solver)

# The states every `stride` steps of each chunk, without its start: one row per sample.
samples(chunks, stride) = permutedims(stack(s for c in chunks for s in c.u[1 + stride:stride:end]))

# Serial run of N chunks from u0: its interfaces U₀, …, U_N and its samples.
function serial(name, u0, N, solver, stride)
    chunks = []
    for _ in 1:N
        push!(chunks, unit_chunk(name, isempty(chunks) ? u0 : chunks[end].u[end], solver))
    end
    return [[u0]; [c.u[end] for c in chunks]], samples(chunks, stride)
end

function spin_up(name, u)
    for _ in 1:40
        u = unit_chunk(name, u, fine).u[end]
    end
    return u
end

# Perturbed starts in the order the archived records drew them: for each N and
# replicate, the reference start, then the baseline start.
function perturbed_starts(name)
    p = problems[name]
    rng = MersenneTwister(p.seed)
    return Dict((N, rep) => [p.start .+ 0.5 .* (rand(rng, length(p.start)) .- 0.5) for _ in 1:2]
                for N in Ns for rep in 1:replicates)
end

# Mean over coordinates of the exact empirical W₁ between samples of equal size.
mean_w1(X, Y) = mean(mean(abs.(sort(x) .- sort(y))) for (x, y) in zip(eachcol(X), eachcol(Y)))

# The neighbour product is x₁x₂ for Lorenz-63 and the mean of the cyclic xᵢxᵢ₋₁ for Lorenz-96.
function means(X)
    d = size(X, 2)
    return (x1 = mean(X[:, 1]), energy = mean(sum(abs2, X; dims = 2)) / d, coupled = mean(X[:, 1] .* X[:, 3]),
            neighbour = d == 3 ? mean(X[:, 1] .* X[:, 2]) : mean(X .* circshift(X, (0, 1))))
end

function autocorrelation(x, lag)
    z = x .- mean(x)
    return mean(z[1:(end - lag)] .* z[(lag + 1):end]) / mean(abs2, z)
end

# How far the samples Y are from the reference samples X; P holds the slicing directions.
function distances(X, Y, P, lag)
    mx, my = means(X), means(Y)
    return (w1 = mean_w1(X, Y), sliced = mean_w1(X * P, Y * P), mean_error = abs(my.x1 - mx.x1),
            energy_error = abs(my.energy - mx.energy), coupled_error = abs(my.coupled - mx.coupled),
            neighbour_error = abs(my.neighbour - mx.neighbour),
            ac_error = abs(autocorrelation(Y[:, 1], lag) - autocorrelation(X[:, 1], lag)))
end

function parareal(name, u0, N, rule; ϵ = 1e-9, τ = 1e-10)
    weight = rule == "psi_lyap" ? FixedWeight(exp(problems[name].λ)) : SecantWeight(τ = τ)
    return solve(ivp(name, u0, Float64(N)),
                 Parareal(RK4(h = h), RK4(h = H); N, criterion = WeightedDefect(ϵ = ϵ, weight = weight)))
end

# The accepted interfaces and the fine samples from each accepted chunk start.
function accepted(name, sol, N, stride)
    U = boundaryvalues(sol.lastiterate)
    return U, samples([unit_chunk(name, u, fine) for u in U[1:N]], stride)
end

fingerprint(U) = bytes2hex(sha256(reinterpret(UInt8, reduce(vcat, U))))

# One replicate: a record per rule, the sensitivity reruns and, for the portrait, the samples.
function replicate(name, N, rep, starts)
    p = problems[name]
    fine_stride, coarse_stride = round(Int, p.interval / h), round(Int, p.interval / H)
    lag = round(Int, 0.5 / p.interval)
    d = length(p.start)
    P = randn(MersenneTwister(107), d, 16)
    P ./= sqrt.(sum(abs2, P; dims = 1))  # unit slicing directions

    u0, u1 = spin_up(name, starts[1]), spin_up(name, starts[2])
    _, X = serial(name, u0, N, fine, fine_stride)  # reference
    _, B = serial(name, u1, N, fine, fine_stride)  # independent fine baseline
    b = distances(X, B, P, lag)
    baseline = (basew1 = b.w1, basesliced = b.sliced, base_mean = b.mean_error, base_energy = b.energy_error,
                base_coupled = b.coupled_error, base_neighbour = b.neighbour_error, base_ac = b.ac_error)

    rows, portrait = [], nothing
    for rule in rules
        if rule == "coarse"
            U, Y = serial(name, u0, N, coarse, coarse_stride)
            counts = (status = "coarse", sweeps = 0, fine_batches = 0)
        else
            sol = parareal(name, u0, N, rule)
            U, Y = accepted(name, sol, N, fine_stride)
            counts = (status = string(sol.diagnostics.status), sweeps = sol.diagnostics.sweeps,
                      fine_batches = sol.diagnostics.fine_batches)
        end
        push!(rows, (; problem = name, N, rep, criterion = rule, distances(X, Y, P, lag)..., baseline..., counts...,
                     interface_hash = fingerprint(U)))
        if name == "lorenz" && N == last(Ns) && rep == 1 && rule == "psi_lip"
            portrait = (fine = X, par = Y, base = B)
        end
    end

    sensitivity = []
    if N == first(Ns) && rep <= 4
        for ϵ in (1e-7, 1e-9, 1e-11), τ in (1e-12, 1e-10, 1e-8)
            sol = parareal(name, u0, N, "psi_lip"; ϵ, τ)
            _, Y = accepted(name, sol, N, fine_stride)
            push!(sensitivity, (; problem = name, N, rep, epsilon = ϵ, tau = τ, sweeps = sol.diagnostics.sweeps,
                                w1 = mean_w1(X, Y), status = string(sol.diagnostics.status)))
        end
    end
    return (; rows, sensitivity, portrait)
end

records_dir = joinpath(@__DIR__, "Records")
partial_file = joinpath(records_dir, "statistics_partial.jls")
mkpath(records_dir)
isfile(joinpath(records_dir, "statistics.jls")) && error("Records/statistics.jls already exists; delete it to recompute.")

starts = Dict(name => perturbed_starts(name) for name in keys(problems))
jobs = [(name, N, rep) for name in ("lorenz", "lorenz96") for N in Ns for rep in 1:replicates]
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
todo = sort([job for job in jobs if !haskey(done, job)];
            by = job -> job[2] * length(problems[job[1]].start), rev = true)  # longest first

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
    name, N, rep = job
    outcome = replicate(name, N, rep, starts[name][(N, rep)])
    lock(guard) do
        done[job] = outcome
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$(length(jobs)): $name N=$N replicate $rep")
    end
end

serialize(joinpath(records_dir, "statistics.jls"), [row for job in jobs for row in done[job].rows])
serialize(joinpath(records_dir, "sensitivity.jls"), [row for job in jobs for row in done[job].sensitivity])
serialize(joinpath(records_dir, "portraits.jls"),
          Dict("lorenz" => only(done[job].portrait for job in jobs if done[job].portrait !== nothing)))
rm(partial_file)

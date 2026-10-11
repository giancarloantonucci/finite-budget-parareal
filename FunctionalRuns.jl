# FunctionalRuns.jl
#
# Long-time functionals of Lorenz-63 and Lorenz-96: the long references and the
# exploratory comparisons quoted in the supplement (sec:functional_accuracy).
# FunctionalTables.jl turns the records into tab:functional_reference and prints
# the quoted exploratory numbers.
#
# Observables, all sampled every 0.01 time units (t = 0.01, …, T): for Lorenz-63
# the mean of z, the energy ‖u‖²/3 and the lag-0.5 autocorrelation of x; for
# Lorenz-96 the spatial mean, the energy ‖u‖²/40 and the neighbour product x_i x_{i-1}.
# Records keep sufficient statistics, so estimates pool exactly across blocks.
#
# References: four chains per problem and step (h = 1e-4 and 5e-5), each a perturbed
# start spun up for 40 time units, then 64 (Lorenz-63) or 16 (Lorenz-96) blocks of
# 128 time units. Exploratory runs: eight spun-up starts per problem, at T = N = 32
# and 128, with serial RK4 at h = 0.01 (the coarse solver alone) and Parareal (fine
# RK4, h = 1e-4) under the Lyapunov and secant rules with RK4 or explicit Euler
# coarse chunks (H = 0.01).
#
# Needs LinearAlgebra, Random, Serialization, NSDERungeKutta and NSDETimeParallel.
# A Lorenz-96 Parareal run at N = 128 holds about 0.4 GB.
# Run:     julia --threads=auto FunctionalRuns.jl
#          An interrupted run resumes from Records/functional_partial.jls.
# Writes:  Records/reference_lorenz.jls, Records/reference_lorenz96.jls,
#          Records/functionals_lorenz.jls, Records/functionals_lorenz96.jls

using LinearAlgebra, Random, Serialization
using NSDERungeKutta, NSDETimeParallel

interval, lag = 0.01, 50  # sampling interval; the autocorrelation lag 0.5 in samples
problems = Dict(
    "lorenz" => (start = [2.0, 3.0, -14.0], λ = 0.9056, blocks = 64, reference_seed = 710631, start_seed = 910631),
    "lorenz96" => (start = [8.0 + (i == 20 ? 0.01 : 0.0) for i in 1:40], λ = 1.68, blocks = 16,
                   reference_seed = 710961, start_seed = 910961),
)
problem_names = ("lorenz", "lorenz96")
ivp(name, u0, T = 1.0) = name == "lorenz" ? Lorenz(copy(u0), (0.0, T)) : Lorenz96(copy(u0), (0.0, T))
unit_chunk(name, u, h) = solve(ivp(name, u), chunk_solver(RK4(h = h), 1.0))

# The states at the sampling times of n unit chunks from u, and the final state.
function sampled(name, u, h, n)
    stride, rows = round(Int, interval / h), Vector{Float64}[]
    for _ in 1:n
        sol = unit_chunk(name, u, h)
        append!(rows, sol.u[1 + stride:stride:end])
        u = sol.u[end]
    end
    return permutedims(stack(rows)), u
end

function spin_up(name, u, h)
    for _ in 1:40
        u = unit_chunk(name, u, h).u[end]
    end
    return u
end

# Sums from which the three observables pool exactly over any set of records.
function sufficient(name, X)
    m, d = size(X)
    if name == "lorenz"
        a, b = X[1:(end - lag), 1], X[(1 + lag):end, 1]
        return (n = m, sum1 = sum(X[:, 3]), energy = sum(abs2, X) / d, pairs = m - lag, sa = sum(a), sb = sum(b),
                saa = sum(abs2, a), sbb = sum(abs2, b), sab = dot(a, b), neighbour = 0.0)
    end
    return (n = m, sum1 = sum(X) / d, energy = sum(abs2, X) / d, pairs = 0, sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0,
            sab = 0.0, neighbour = sum(X .* X[:, [d; 1:(d - 1)]]) / d)
end

# The three observables from pooled sufficient statistics.
function observables(name, ss)
    n = sum(s.n for s in ss)
    mean1, energy = sum(s.sum1 for s in ss) / n, sum(s.energy for s in ss) / n
    name == "lorenz96" && return [mean1, energy, sum(s.neighbour for s in ss) / n]
    k, sa, sb = sum(s.pairs for s in ss), sum(s.sa for s in ss), sum(s.sb for s in ss)
    va, vb = sum(s.saa for s in ss) - sa^2 / k, sum(s.sbb for s in ss) - sb^2 / k
    return [mean1, energy, (sum(s.sab for s in ss) - sa * sb / k) / sqrt(va * vb)]
end

# Perturbed starts, in the order the archived records drew them.
perturbed(name, rng) = problems[name].start .+ 0.5 .* (rand(rng, length(problems[name].start)) .- 0.5)
chain_starts = Dict(name => (rng = MersenneTwister(problems[name].reference_seed); [perturbed(name, rng) for _ in 1:4])
                    for name in problem_names)
run_starts = Dict(name => (rng = MersenneTwister(problems[name].start_seed); [perturbed(name, rng) for _ in 1:8])
                  for name in problem_names)

function reference_chain(name, chain, h)
    u = spin_up(name, chain_starts[name][chain], h)
    rows = []
    for block in 1:problems[name].blocks
        X, u = sampled(name, u, h, 128)
        push!(rows, (; problem = name, h, chain, block, sufficient(name, X)...))
    end
    return rows
end

# Fine samples of the accepted trajectory on the grid, each from the chunk containing
# it (the left one at a seam), by the solver's linear interpolation.
function accepted_samples(chunks, N)
    states = map(collect(interval:interval:N)) do t
        c = chunks[min(N, ceil(Int, t))]
        c(clamp(t, c.t[1], c.t[end]))
    end
    return permutedims(stack(states))
end

function exploratory(name, N, rep)
    p = problems[name]
    u0 = spin_up(name, run_starts[name][rep], 1e-4)
    X, _ = sampled(name, u0, 0.01, N)
    rows = [(; problem = name, N, rep, method = "serial_rk4", coarse = "none", h = 0.01, K = 0, status = "finite",
             phi = observables(name, [sufficient(name, X)]))]
    for (coarse, solver) in (("rk4", RK4(h = 0.01)), ("euler", Euler(h = 0.01))), rule in ("psi_lyap", "psi_lip")
        weight = rule == "psi_lyap" ? FixedWeight(exp(p.λ)) : SecantWeight()
        sol = solve(ivp(name, u0, Float64(N)),
                    Parareal(RK4(h = 1e-4), solver; N, criterion = WeightedDefect(ϵ = 1e-9, weight = weight)))
        d = sol.diagnostics
        phi = d.status == :nonfinite ? fill(NaN, 3) :
              observables(name, [sufficient(name, accepted_samples(sol.lastiterate, N))])
        push!(rows, (; problem = name, N, rep, method = rule, coarse, h = 0.01, K = d.fine_batches,
                     status = string(d.status), phi))
    end
    return [(; r.problem, r.N, r.rep, r.method, r.coarse, r.h, r.K, r.status,
             phi1 = r.phi[1], phi2 = r.phi[2], phi3 = r.phi[3]) for r in rows]
end

records_dir = joinpath(@__DIR__, "Records")
partial_file = joinpath(records_dir, "functional_partial.jls")
mkpath(records_dir)
any(isfile(joinpath(records_dir, "$(f)_$name.jls")) for f in ("reference", "functionals") for name in problem_names) &&
    error("Functional records already exist in Records/; delete them to recompute.")

jobs = [[("reference", name, chain, h) for name in problem_names for chain in 1:4 for h in (1e-4, 5e-5)];
        [("exploratory", name, N, rep) for name in problem_names for N in (32, 128) for rep in 1:8]]
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
todo = filter(job -> !haskey(done, job), jobs)  # the long reference chains come first

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
    kind, name, a, b = job
    outcome = kind == "reference" ? reference_chain(name, a, b) : exploratory(name, a, b)
    lock(guard) do
        done[job] = outcome
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$(length(jobs)): $job")
    end
end

rows_of(kind, name) = [row for job in filter(job -> job[1] == kind && job[2] == name, jobs) for row in done[job]]
for name in problem_names
    serialize(joinpath(records_dir, "reference_$name.jls"), rows_of("reference", name))
    serialize(joinpath(records_dir, "functionals_$name.jls"), rows_of("exploratory", name))
end
rm(partial_file)

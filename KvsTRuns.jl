# KvsTRuns.jl
#
# Parareal scaling runs for four problems and up to four stopping rules: strong
# scaling (T = 128), weak scaling (ΔT = 1) and a growing horizon (N = 32).
# KvsTTables.jl turns the records into the scaling tables (tab:scaling_*) and
# KvsTFigures.jl into the Lorenz-63 batch-count figures (fig:KvsTLorenz,
# fig:SerialWork).
#
# Each record holds the run's sweeps, fine batches, status and model speedup
# (eq:cost), the interface error ‖e‖₂ against the serial fine solution and, for
# Lorenz-63 and Lorenz-96, the dense mean marginal W₁ distances of the accepted
# and coarse-only trajectories to it.
#
# Needs LinearAlgebra, Serialization, Statistics, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto KvsTRuns.jl
#          An interrupted run resumes from Records/scaling_partial.jls.
# Writes:  Records/scaling.jls (672 records from 600 distinct runs)

using LinearAlgebra, Serialization, Statistics
using NSDERungeKutta, NSDETimeParallel

# Fine step h, nominal Lyapunov exponent λ, tolerance ϵ, coarsening ratios ξ
# (coarse step ξh), stopping rules, growing-horizon end times and W₁ sampling interval.
problems = Dict(
    "logistic" => (h = 1e-3, λ = 0.0, ϵ = 1e-12, ξs = (10, 100),
                   rules = ("standard", "psi_w1", "psi_lip"), Ts = 16:16:128, sampling = nothing),
    "lorenz" => (h = 1e-4, λ = 0.9056, ϵ = 1e-9, ξs = (10, 100),
                 rules = ("standard", "psi_w1", "psi_lyap", "psi_lip"), Ts = 8:8:128, sampling = 0.02),
    "lorenz96" => (h = 1e-4, λ = 1.68, ϵ = 1e-9, ξs = (100,),
                   rules = ("standard", "psi_w1", "psi_lyap", "psi_lip"), Ts = 8:8:128, sampling = 0.05),
    "periodic" => (h = 1e-4, λ = 0.0, ϵ = 1e-9, ξs = (10, 100),
                   rules = ("standard", "psi_w1", "psi_lip"), Ts = 8:8:128, sampling = nothing),
)
problem_names = ("logistic", "lorenz", "lorenz96", "periodic")
Ns = (2, 4, 8, 16, 32, 64, 128)

function ivp(name, T)
    name == "logistic" && return Logistic([0.1], (0.0, T))
    name == "lorenz" && return Lorenz([2.0, 3.0, -14.0], (0.0, T))
    name == "lorenz96" && return Lorenz96([8.0 + (i == 20 ? 0.01 : 0.0) for i in 1:40], (0.0, T))
    name == "periodic" && return Lorenz([0.0, 3.6839592143970576, 73.30237399900534], (0.0, T); ρ = 100.5)
end

solver(name, h) = name == "logistic" ? BackwardEuler(h = h, εᵣ = 1e-13, εₐ = 1e-15, Mₙ = 30) : RK4(h = h)

# The Lyapunov weight max{1, e^{λΔT}} per chunk is the base e^{max(0, λ)} per unit time.
function stopping(rule, p)
    rule == "standard" && return RelativeUpdate(atol = 1e-3 * p.ϵ, rtol = p.ϵ)
    rule == "psi_w1" && return WeightedDefect(ϵ = p.ϵ, weight = FixedWeight(1.0))
    rule == "psi_lyap" && return WeightedDefect(ϵ = p.ϵ, weight = FixedWeight(exp(max(0, p.λ))))
    rule == "psi_lip" && return WeightedDefect(ϵ = p.ϵ, weight = SecantWeight())
end

# Serial propagation from u0, one solution per chunk, on Parareal's chunk grid and
# schedule, so that its interfaces are bitwise those of exact completion.
function serial_chunks(problem, solver, N)
    t0, tN = problem.tspan
    T = [(N - n + 1) / N * t0 + (n - 1) / N * tN for n in 1:(N + 1)]
    schedule = chunk_solver(solver, (tN - t0) / N)
    chunks, u = [], problem.u0
    for n in 1:N
        push!(chunks, solve(remake(problem; u0 = copy(u), tspan = (T[n], T[n + 1])), schedule))
        u = chunks[end].u[end]
    end
    return chunks
end

function sampling_grid(T, interval)
    n = round(Int, T / interval)
    return range(T / n, T; length = n)
end

# States at the sampling times, each read from the chunk containing it (the left
# one at a seam) by the solver's linear interpolation; one row per time.
function samples(chunks, N, grid, ΔT)
    states = map(grid) do t
        chunk = chunks[min(N, ceil(Int, t / ΔT))]
        chunk(clamp(t, chunk.t[1], chunk.t[end]))
    end
    return permutedims(stack(states))
end

# Mean over coordinates of the exact empirical W₁ between samples of equal size.
mean_w1(X, Y) = mean(mean(abs.(sort(x) .- sort(y))) for (x, y) in zip(eachcol(X), eachcol(Y)))

# What every rule's run on one grid is compared with: the serial fine interfaces
# and, for the chaotic problems, the fine samples and the coarse-only distance.
function reference_data(name, ξ, N, T)
    p = problems[name]
    problem = ivp(name, T)
    fine = serial_chunks(problem, solver(name, p.h), N)
    U★ = [[problem.u0]; [chunk.u[end] for chunk in fine]]
    p.sampling === nothing && return (; U★)
    grid = sampling_grid(T, p.sampling)
    X = samples(fine, N, grid, T / N)
    coarse = samples(serial_chunks(problem, solver(name, ξ * p.h), N), N, grid, T / N)
    return (; U★, grid, X, w1_coarse = mean_w1(X, coarse))
end

function run_case(name, ξ, rule, N, T, ref)
    p = problems[name]
    parareal = Parareal(solver(name, p.h), solver(name, ξ * p.h); N, criterion = stopping(rule, p))
    sol = solve(ivp(name, T), parareal)  # serial backend: the runs themselves share the threads
    d = sol.diagnostics
    failed = d.status == :nonfinite
    statistical = p.sampling !== nothing
    return (sweeps = d.sweeps, fine_batches = d.fine_batches, status = string(d.status),
            error = failed ? NaN : norm(norm.(boundaryvalues(sol.lastiterate) .- ref.U★)),
            w1_dense = failed || !statistical ? NaN :
                       mean_w1(ref.X, samples(sol.lastiterate, N, ref.grid, T / N)),
            coarse_w1_dense = statistical ? ref.w1_coarse : NaN,
            speedup = failed ? NaN : theoretical_speedup(sol, 1 / ξ))  # coarse/fine chunk cost 1/ξ
end

# One record per table and figure entry. The scenarios share some (N, T), so the
# distinct runs are keyed without the scenario.
scenarios(name) = [[("strong", N, 128.0) for N in Ns];
                   [("weak", N, Float64(N)) for N in Ns];
                   [("growing", 32, Float64(T)) for T in problems[name].Ts]]
settings = [(; problem = name, xi = ξ, scenario, N, T, dt = T / N, criterion = rule)
            for name in problem_names for ξ in problems[name].ξs
            for (scenario, N, T) in scenarios(name) for rule in problems[name].rules]
run_key(s) = (s.problem, s.xi, s.criterion, s.N, s.T)

records_file = joinpath(@__DIR__, "Records", "scaling.jls")
partial_file = joinpath(@__DIR__, "Records", "scaling_partial.jls")
mkpath(dirname(records_file))
isfile(records_file) && error("Records/scaling.jls already exists; delete it to recompute.")

runs = unique(run_key.(settings))
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
todo = [k for k in runs if !haskey(done, k)]
sort!(todo; by = k -> k[5] * length(ivp(k[1], 1.0).u0) / problems[k[1]].h, rev = true)  # longest first

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
references = Dict()
threaded_foreach(unique([(name, ξ, N, T) for (name, ξ, _, N, T) in todo])) do key
    ref = reference_data(key...)
    lock(guard) do
        references[key] = ref
    end
end
threaded_foreach(todo) do k
    name, ξ, rule, N, T = k
    outcome = run_case(name, ξ, rule, N, T, references[(name, ξ, N, T)])
    lock(guard) do
        done[k] = outcome
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$(length(runs)): $name ξ=$ξ N=$N T=$T $rule → $(outcome.status)")
    end
end

serialize(records_file, [merge(s, done[run_key(s)]) for s in settings])
rm(partial_file)

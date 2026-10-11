# LongChunkRuns.jl
#
# The long-chunk grids of sec:submitted_grid: Lorenz-63 from (2, 3, -14), fine RK4
# h = 1e-4, coarse RK4 H = ξh with ξ = 10, 100, under standard stopping and the ψ rules
# with w = e^{λΔT} (λ = 0.9056) and the secant weight, ε = 1e-9. The 150 (N, T) pairs are
# N = 100 with T = 10, 20, …, 500; T = 500 with N = 1, 3, …, 99 and 100; and T = 5N for
# the same N. Where a step does not divide ΔT = T/N, a chunk takes full steps and one
# shortened last step. LongChunkTables.jl and LongChunkFigures.jl turn the records into
# tab:submitted_grid_summary, tab:submitted_grid_endpoint and fig:submitted_grid.
#
# Each record holds the sweeps s, fine batches K_F, the stop (named as in the archived
# records: criterion, exact_prefix, budget_exhausted, nonfinite), the interface error
# ‖U - U*‖₂ against the serial fine run and the modelled speed-up N/(K_F + ζ K_G), with
# ζ the coarse-to-fine step ratio per chunk (a shortened step counts as a step).
#
# Needs Serialization, LinearAlgebra, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto LongChunkRuns.jl
#          An interrupted run resumes from Records/submitted_grid_partial.jls.
# Writes:  Records/submitted_grid.jls (900 records)

using LinearAlgebra, Serialization
using NSDERungeKutta, NSDETimeParallel

λ, ϵ, h, start = 0.9056, 1e-9, 1e-4, [2.0, 3.0, -14.0]
pairs = unique([[(100, 500.0)]; [(100, Float64(T)) for T in 10:10:500]; [(N, 500.0) for N in 1:2:99];
                [(N, 5.0 * N) for N in 1:2:99]])
criteria = ("standard" => RelativeUpdate(atol = 1e-3 * ϵ, rtol = ϵ),
            "psi_lyap" => WeightedDefect(ϵ = ϵ, weight = FixedWeight(exp(λ))),
            "psi_lip" => WeightedDefect(ϵ = ϵ, weight = SecantWeight()))
status_name = Dict(:converged => "criterion", :exact => "exact_prefix", :budget_exhausted => "budget_exhausted",
                   :nonfinite => "nonfinite")

# Steps per chunk of NSDE's own schedule.
steps(step, ΔT) = (plan = chunk_solver(RK4(h = step), ΔT).plan; plan.full + (plan.tail > 0))

function grid_point((ξ, N, T))
    ΔT = T / N
    problem = Lorenz(copy(start), (0.0, N * ΔT))
    ζ = steps(ξ * h, ΔT) / steps(h, ΔT)
    U★ = reference(problem, Parareal(RK4(h = h), RK4(h = ξ * h); N))
    return map(criteria) do (criterion, rule)
        sol = solve(problem, Parareal(RK4(h = h), RK4(h = ξ * h); N, criterion = rule))
        d = sol.diagnostics
        finite = d.status != :nonfinite
        U = boundaryvalues(sol.lastiterate)
        (; xi = ξ, N, T, dt = ΔT, criterion, sweeps = d.sweeps, fine_batches = d.fine_batches,
         status = status_name[d.status], error = finite ? norm([norm(U[n] - U★[n]) for n in eachindex(U★)]) : NaN,
         speedup = finite ? theoretical_speedup(d, ζ) : NaN)
    end
end

records_dir = joinpath(@__DIR__, "Records")
partial_file = joinpath(records_dir, "submitted_grid_partial.jls")
mkpath(records_dir)
isfile(joinpath(records_dir, "submitted_grid.jls")) && error("Records/submitted_grid.jls already exists; delete it to recompute.")
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
jobs = [(ξ, N, T) for ξ in (10, 100) for (N, T) in pairs]
todo = sort(filter(job -> !haskey(done, job), jobs); by = job -> -job[3])  # longest horizon first

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
    rows = grid_point(job)
    lock(guard) do
        done[job] = rows
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$(length(jobs)): ξ = $(job[1]), N = $(job[2]), T = $(job[3])")
    end
end

serialize(joinpath(records_dir, "submitted_grid.jls"), [row for job in jobs for row in done[job]])
rm(partial_file)

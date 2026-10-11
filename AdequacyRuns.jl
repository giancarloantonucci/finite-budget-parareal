# AdequacyRuns.jl
#
# The Lorenz-63 stopping-adequacy study (sec:stopping_adequacy) at T = N = 32, ΔT = 1,
# with fine RK4 (h = 1e-4), in the order the protocol ran it:
#   calibration  64 serial fine runs, which fix the accuracy limits;
#   pilot        8 starts: a serial run with each of the seven candidate coarse solvers,
#                and a forced Parareal run to exact completion with each, checked
#                bitwise against the serial fine interfaces (the feasibility screen);
#   first        32 starts with the pilot's choice, RK4 H = 0.1: coarse-only and the
#                three rules;
#   second       32 fresh starts with the screened choice, RK4 H = 0.05: coarse-only,
#                serial fine, serial RK4 at h = 0.005, the three rules, and a forced run
#                whose every iterate is rebuilt by fine propagation from its starts.
# The two choices are fixed below; AdequacyTables.jl checks that the protocol's rules
# pick them from these records, and turns the records into tab:stopping_adequacy and
# tab:functional_main. AdequacyFigures.jl draws fig:stopping_adequacy.
#
# Every start perturbs (2, 3, -14) and is spun up for 40 time units with fine RK4. Runs
# are sampled every 0.01 time units (t = 0.01, …, 32) for the mean of z, the energy
# ‖u‖²/3 and the lag-0.5 autocorrelation of x. The rules are ψ with the Lyapunov weight
# (λ = 0.9056) and with the secant weight, both at ε = 1e-9, and the standard rule with
# tolerances 1e-12 (absolute) and 1e-9 (relative).
#
# Needs LinearAlgebra, Random, Serialization, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto AdequacyRuns.jl
#          Records that already exist are kept; only the stages whose records are
#          missing are run. An interrupted run resumes from Records/adequacy_partial.jls.
# Writes:  Records/adequacy_calibration.jls (64 records), adequacy_pilot.jls (56),
#          adequacy_feasibility.jls (56), adequacy_first.jls (128), adequacy.jls (192)
#          and adequacy_forced.jls (1024, one per start and number of corrections)

using LinearAlgebra, Random, Serialization
using NSDERungeKutta, NSDETimeParallel

N, lag, λ, interval = 32, 50, 0.9056, 0.01
fine = RK4(h = 1e-4)
coarse_solver(method, H) = method == "rk4" ? RK4(h = H) : Euler(h = H)
candidates = [("euler", 0.005), ("euler", 0.01), ("euler", 0.02), ("rk4", 0.025), ("rk4", 0.04), ("rk4", 0.05), ("rk4", 0.1)]
first_choice, second_choice = ("rk4", 0.1), ("rk4", 0.05)
rules = ["psi_lyap" => WeightedDefect(ϵ = 1e-9, weight = FixedWeight(exp(λ))),
         "psi_lip" => WeightedDefect(ϵ = 1e-9, weight = SecantWeight()),
         "standard" => RelativeUpdate(atol = 1e-12, rtol = 1e-9)]

# Starts for each stage, in the order the archived records drew them.
perturbed(rng) = [2.0, 3.0, -14.0] .+ 0.5 .* (rand(rng, 3) .- 0.5)
starts = Dict(name => (rng = MersenneTwister(seed); [perturbed(rng) for _ in 1:number])
              for (name, seed, number) in (("calibration", 2026101001, 64), ("pilot", 2026101002, 8),
                                           ("first", 2026101003, 32), ("second", 2026101004, 32)))

unit_chunk(u, n, solver) = solve(Lorenz(copy(u), (n - 1.0, Float64(n))), chunk_solver(solver, 1.0))
whole(u) = Lorenz(copy(u), (0.0, Float64(N)))

function spin_up(u)
    for n in 1:40
        u = unit_chunk(u, n, fine).u[end]
    end
    return u
end

# The serial run of N unit chunks from u.
function serial(u, solver)
    chunks = []
    for n in 1:N
        push!(chunks, unit_chunk(isempty(chunks) ? u : chunks[end].u[end], n, solver))
    end
    return chunks
end

# The three observables from samples at t = 0.01, …, N, each taken from the chunk that
# contains it (the left one at a seam) by the solver's linear interpolation.
function observables(chunks)
    X = permutedims(stack(map(collect(interval:interval:N)) do t
        c = chunks[min(N, ceil(Int, t))]
        c(clamp(t, c.t[1], c.t[end]))
    end))
    all(isfinite, X) || return fill(NaN, 3)
    m, d = size(X)
    x = @view X[:, 1]
    a, b = (@view x[1:(end - lag)]), (@view x[(1 + lag):end])
    k, sa, sb = m - lag, sum(a), sum(b)
    va, vb = sum(abs2, a) - sa^2 / k, sum(abs2, b) - sb^2 / k
    return [sum(@view X[:, 3]) / m, sum(abs2, X) / d / m, (dot(a, b) - sa * sb / k) / sqrt(va * vb)]
end

function row(rep, method, h, phi; s = -1, K = 0, status = all(isfinite, phi) ? "finite" : "nonfinite",
             fine_calls = 0, coarse_calls = 0)
    return (; rep, method, h, s, K, status, fine_calls, coarse_calls, phi1 = phi[1], phi2 = phi[2], phi3 = phi[3])
end

function accepted(rep, u, (method, H))
    map(rules) do (name, criterion)
        sol = solve(whole(u), Parareal(fine, coarse_solver(method, H); N, criterion))
        d = sol.diagnostics
        phi = d.status == :nonfinite ? fill(NaN, 3) : observables(sol.lastiterate)
        row(rep, name, H, phi; s = d.sweeps, K = d.fine_batches, status = string(d.status),
            fine_calls = d.fine_chunk_calls, coarse_calls = d.coarse_chunk_calls)
    end
end

function calibration(rep)
    u = spin_up(starts["calibration"][rep])
    return Dict("adequacy_calibration" => [row(rep, "fine", 1e-4, observables(serial(u, fine)))])
end

function pilot(rep)
    u = spin_up(starts["pilot"][rep])
    serial_rows = [(; rep, method, h = H, phi = observables(serial(u, coarse_solver(method, H))))
                   for (method, H) in candidates]
    feasibility = map(candidates) do (method, H)
        parareal = Parareal(fine, coarse_solver(method, H); N, criterion = NoStop())
        sol = solve(whole(u), parareal)
        exact = sol.diagnostics.status == :exact
        (; rep, method, h = H, status = string(sol.diagnostics.status), K = sol.diagnostics.fine_batches, exact,
         bitwise = exact && boundaryvalues(sol.lastiterate) == reference(whole(u), parareal))
    end
    return Dict("adequacy_pilot" => [(; r.rep, r.method, r.h, phi1 = r.phi[1], phi2 = r.phi[2], phi3 = r.phi[3])
                                     for r in serial_rows],
                "adequacy_feasibility" => feasibility)
end

function first_confirmation(rep)
    u = spin_up(starts["first"][rep])
    method, H = first_choice
    return Dict("adequacy_first" => [row(rep, "coarse", H, observables(serial(u, coarse_solver(method, H))));
                                     accepted(rep, u, first_choice)])
end

function second_confirmation(rep)
    u = spin_up(starts["second"][rep])
    method, H = second_choice
    rows = [row(rep, "coarse", H, observables(serial(u, coarse_solver(method, H)))),
            row(rep, "fine", 1e-4, observables(serial(u, fine))),
            row(rep, "serial_half", 0.005, observables(serial(u, RK4(h = 0.005))))]
    # Every iterate of a run forced to exact completion, rebuilt from its starts.
    sol = solve(whole(u), Parareal(fine, coarse_solver(method, H); N, criterion = NoStop()); history = :interfaces)
    iterates = sol.diagnostics.interfaces
    forced = map(0:(N - 1)) do s
        phi = s < length(iterates) ? observables([unit_chunk(iterates[s + 1][n], n, fine) for n in 1:N]) : fill(NaN, 3)
        row(rep, "forced", H, phi; s, K = s + 1)
    end
    return Dict("adequacy" => [rows; accepted(rep, u, second_choice)], "adequacy_forced" => forced)
end

records_dir = joinpath(@__DIR__, "Records")
partial_file = joinpath(records_dir, "adequacy_partial.jls")
# The records each stage writes.
outputs = Dict("calibration" => ("adequacy_calibration",), "pilot" => ("adequacy_pilot", "adequacy_feasibility"),
               "first" => ("adequacy_first",), "second" => ("adequacy", "adequacy_forced"))
missing_file(f) = !isfile(joinpath(records_dir, f * ".jls"))
mkpath(records_dir)
run_stages = filter(name -> any(missing_file, outputs[name]), ["second", "pilot", "first", "calibration"])  # the longest first
isempty(run_stages) && error("All adequacy records exist in Records/; delete those to recompute.")
println("Running: ", join(run_stages, ", "))

jobs = [(name, rep) for name in run_stages for rep in 1:length(starts[name])]
stages = Dict("calibration" => calibration, "pilot" => pilot, "first" => first_confirmation, "second" => second_confirmation)
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
todo = filter(job -> !haskey(done, job), jobs)

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
    outcome = stages[job[1]](job[2])
    lock(guard) do
        done[job] = outcome
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$(length(jobs)): $job")
    end
end

for name in run_stages, f in outputs[name]
    missing_file(f) || continue
    serialize(joinpath(records_dir, f * ".jls"), [r for rep in 1:length(starts[name]) for r in done[(name, rep)][f]])
end
rm(partial_file)

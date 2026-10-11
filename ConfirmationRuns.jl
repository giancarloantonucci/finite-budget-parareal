# ConfirmationRuns.jl
#
# The independent Lorenz-96 confirmation of the exploratory energy gain
# (sec:functional_confirmation): 32 fresh starts at T = N = 128, ΔT = 1.
# ConfirmationTables.jl turns the records into tab:functional_confirmation; the
# Lorenz-96 rows of tab:functional_main come from the same records.
#
# Each start perturbs the prescribed one and is spun up for 40 time units with fine
# RK4 (h = 1e-4). From it run coarse-only (serial RK4, h = 0.01), serial RK4 at
# h = 0.005, and Parareal (fine RK4 h = 1e-4, coarse RK4 H = 0.01) under the
# Lyapunov (λ = 1.68) and secant rules with ε = 1e-9. Each run is sampled every 0.01
# time units (t = 0.01, …, 128) for the spatial mean, the energy ‖u‖²/40 and the
# neighbour product x_i x_{i-1}, each averaged over the samples.
#
# Needs Random, Serialization, NSDERungeKutta and NSDETimeParallel. A Parareal run
# holds about 0.4 GB.
# Run:     julia --threads=auto ConfirmationRuns.jl
#          An interrupted run resumes from Records/confirmation_partial.jls.
# Writes:  Records/confirmation.jls (128 records)

using Random, Serialization
using NSDERungeKutta, NSDETimeParallel

N, replicates, λ, interval = 128, 32, 1.68, 0.01
start = [8.0 + (i == 20 ? 0.01 : 0.0) for i in 1:40]
unit_chunk(u, h) = solve(Lorenz96(copy(u), (0.0, 1.0)), chunk_solver(RK4(h = h), 1.0))

function spin_up(u)
    for _ in 1:40
        u = unit_chunk(u, 1e-4).u[end]
    end
    return u
end

# The three observables, averaged over the m × 40 samples X (one row per time).
function observables(X)
    m, d = size(X)
    return [sum(X) / d / m, sum(abs2, X) / d / m, sum(X .* X[:, [d; 1:(d - 1)]]) / d / m]
end

function serial_samples(u, h)
    stride, rows = round(Int, interval / h), Vector{Float64}[]
    for _ in 1:N
        sol = unit_chunk(u, h)
        append!(rows, sol.u[1 + stride:stride:end])
        u = sol.u[end]
    end
    return permutedims(stack(rows))
end

# Samples of the accepted trajectory, each from the chunk containing it (the left one
# at a seam), by the solver's linear interpolation.
function accepted_samples(chunks)
    states = map(collect(interval:interval:N)) do t
        c = chunks[min(N, ceil(Int, t))]
        c(clamp(t, c.t[1], c.t[end]))
    end
    return permutedims(stack(states))
end

# Perturbed starts, in the order the archived records drew them.
rng = MersenneTwister(2026100932)
starts = [start .+ 0.5 .* (rand(rng, 40) .- 0.5) for _ in 1:replicates]

function replicate(rep)
    u0 = spin_up(starts[rep])
    rows = [(; rep, method, h, K = 0, status = "finite", fine_calls = 0, coarse_calls = 0,
             phi = observables(serial_samples(u0, h))) for (method, h) in (("coarse", 0.01), ("serial_half", 0.005))]
    for (method, weight) in (("psi_lyap", FixedWeight(exp(λ))), ("psi_lip", SecantWeight()))
        sol = solve(Lorenz96(copy(u0), (0.0, Float64(N))),
                    Parareal(RK4(h = 1e-4), RK4(h = 0.01); N, criterion = WeightedDefect(ϵ = 1e-9, weight = weight)))
        d = sol.diagnostics
        phi = d.status == :nonfinite ? fill(NaN, 3) : observables(accepted_samples(sol.lastiterate))
        push!(rows, (; rep, method, h = 0.01, K = d.fine_batches, status = string(d.status),
                     fine_calls = d.fine_chunk_calls, coarse_calls = d.coarse_chunk_calls, phi))
    end
    return [(; r.rep, r.method, r.h, r.K, r.status, r.fine_calls, r.coarse_calls,
             phi1 = r.phi[1], phi2 = r.phi[2], phi3 = r.phi[3]) for r in rows]
end

records_dir = joinpath(@__DIR__, "Records")
partial_file = joinpath(records_dir, "confirmation_partial.jls")
mkpath(records_dir)
isfile(joinpath(records_dir, "confirmation.jls")) && error("Records/confirmation.jls already exists; delete it to recompute.")
done = isfile(partial_file) ? deserialize(partial_file) : Dict()  # only load records you made yourself
todo = filter(rep -> !haskey(done, rep), 1:replicates)

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
threaded_foreach(todo) do rep
    rows = replicate(rep)
    lock(guard) do
        done[rep] = rows
        serialize(partial_file * ".tmp", done)
        mv(partial_file * ".tmp", partial_file; force = true)
        println("$(length(done))/$replicates: start $rep")
    end
end

serialize(joinpath(records_dir, "confirmation.jls"), [row for rep in 1:replicates for row in done[rep]])
rm(partial_file)

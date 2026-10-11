# PrefixRuns.jl
#
# Per-interface error of Lorenz-63 runs at N = 128, ΔT = 1 against the serial fine
# solution, with fine RK4 (h = 1e-4) and coarse RK4 or explicit Euler (H = 1e-2)
# chunks, under four acceptances: coarse-only, the Lyapunov and secant rules, and
# the standard check. PrefixTables.jl turns the records into tab:prefix_summary and
# PrefixFigures.jl into fig:prefix.
#
# Three cases: RK4 and Euler coarse solvers from the prescribed start (2, 3, -14),
# and Euler from the first N = 128 replicate start of the poor-coarse control in
# BadCoarseRuns.jl.
#
# Needs LinearAlgebra, Random, Serialization, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto PrefixRuns.jl
# Writes:  Records/prefix_accuracy.jls (12 records)

using LinearAlgebra, Random, Serialization
using NSDERungeKutta, NSDETimeParallel

N, h, H = 128, 1e-4, 1e-2
λ, ϵ = 0.9056, 1e-9  # nominal Lyapunov exponent and tolerance
prescribed = [2.0, 3.0, -14.0]
ivp(u0, T = 1.0) = Lorenz(copy(u0), (0.0, T))
fine = chunk_solver(RK4(h = h), 1.0)

# The serial interfaces U₀, …, U_N from u0, one chunk at a time.
function serial(u0, solver)
    U = [u0]
    for _ in 1:N
        push!(U, solve(ivp(U[end]), solver).u[end])
    end
    return U
end

# BadCoarseRuns.jl draws its eight N = 16 pairs of starts first, then this one,
# and spins each up for 40 time units.
function replicate_start()
    rng = MersenneTwister(20260911)
    u = last([prescribed .+ 0.5 .* (rand(rng, 3) .- 0.5) for _ in 1:17])
    for _ in 1:40
        u = solve(ivp(u), fine).u[end]
    end
    return u
end

function stopping(rule)
    rule == "standard" && return RelativeUpdate(atol = 1e-3 * ϵ, rtol = ϵ)
    return WeightedDefect(ϵ = ϵ, weight = rule == "psi_lyap" ? FixedWeight(exp(λ)) : SecantWeight())
end

function runs(name, u0, coarse)
    U★ = serial(u0, fine)
    rows = [(; case = name, rule = "coarse", N, errors = norm.(serial(u0, chunk_solver(coarse, 1.0)) .- U★),
             sweeps = 0, fine_batches = 0, status = "coarse")]
    for rule in ("psi_lyap", "psi_lip", "standard")
        sol = solve(ivp(u0, Float64(N)), Parareal(RK4(h = h), coarse; N, criterion = stopping(rule)))
        d = sol.diagnostics
        errors = d.status == :nonfinite ? fill(NaN, N + 1) : norm.(boundaryvalues(sol.lastiterate) .- U★)
        push!(rows, (; case = name, rule, N, errors, sweeps = d.sweeps, fine_batches = d.fine_batches,
                     status = string(d.status)))
        println("$name, $rule: $(d.fine_batches) fine batches, $(d.status)")
    end
    return rows
end

records_file = joinpath(@__DIR__, "Records", "prefix_accuracy.jls")
mkpath(dirname(records_file))
isfile(records_file) && error("Records/prefix_accuracy.jls already exists; delete it to recompute.")
cases = [("RK4 scaling", prescribed, RK4(h = H)), ("Euler prescribed", prescribed, Euler(h = H)),
         ("Euler control", replicate_start(), Euler(h = H))]
serialize(records_file, reduce(vcat, fetch.([Threads.@spawn runs(case...) for case in cases])))

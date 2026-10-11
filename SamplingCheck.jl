# SamplingCheck.jl
#
# The sampling refinement quoted in sec:scaling of the supplement: Lorenz-63
# and Lorenz-96 at N = T = 16 (ΔT = 1), ξ = 100, as in the weak-scaling runs of
# KvsTRuns.jl. Each prints the dense mean marginal W₁ distance to the serial fine
# trajectory of coarse-only and of the two weighted rules, on the tabulated sampling
# grid (interval 0.02 for Lorenz-63, 0.05 for Lorenz-96) and on one eight times finer.
# On the tabulated grid the numbers must equal those of the scaling tables. Nothing is
# written; the check takes seconds.
#
# Needs Statistics, NSDERungeKutta and NSDETimeParallel.
# Run:     julia SamplingCheck.jl

using Statistics
using NSDERungeKutta, NSDETimeParallel

N, T, h, ξ, ϵ = 16, 16.0, 1e-4, 100, 1e-9
problems = [("Lorenz-63", Lorenz([2.0, 3.0, -14.0], (0.0, T)), 0.9056, 0.02),
            ("Lorenz-96", Lorenz96([8.0 + (i == 20 ? 0.01 : 0.0) for i in 1:40], (0.0, T)), 1.68, 0.05)]

# As in KvsTRuns.jl: serial chunks on Parareal's grid, a uniform sampling grid without
# t = 0, each time read from the chunk containing it (the left one at a seam).
function serial_chunks(problem, solver)
    schedule, chunks, u = chunk_solver(solver, T / N), [], problem.u0
    for n in 1:N
        push!(chunks, solve(remake(problem; u0 = copy(u), tspan = ((n - 1) / N * T, n / N * T)), schedule))
        u = chunks[end].u[end]
    end
    return chunks
end
sampling_grid(interval) = (n = round(Int, T / interval); range(T / n, T; length = n))
function samples(chunks, grid)
    states = map(grid) do t
        chunk = chunks[min(N, ceil(Int, t / (T / N)))]
        chunk(clamp(t, chunk.t[1], chunk.t[end]))
    end
    return permutedims(stack(states))
end
mean_w1(X, Y) = mean(mean(abs.(sort(x) .- sort(y))) for (x, y) in zip(eachcol(X), eachcol(Y)))

for (label, problem, λ, interval) in problems
    fine, coarse = RK4(h = h), RK4(h = ξ * h)
    trajectories = Pair{String, Any}["coarse-only" => serial_chunks(problem, coarse)]  # chunk vectors and Parareal iterates
    for (rule, weight) in ("Lyapunov" => FixedWeight(exp(λ)), "secant" => SecantWeight())
        sol = solve(problem, Parareal(fine, coarse; N, criterion = WeightedDefect(ϵ = ϵ, weight = weight)))
        sol.diagnostics.status == :nonfinite && error("$label, $rule: nonfinite run")
        push!(trajectories, rule => sol.lastiterate)
    end
    reference = serial_chunks(problem, fine)
    println(label, ":")
    for Δ in (interval, interval / 8)
        grid = sampling_grid(Δ)
        X = samples(reference, grid)
        println("  interval ", Δ, ": ",
                join(("$name $(round(mean_w1(X, samples(c, grid)); sigdigits = 3))" for (name, c) in trajectories), ", "))
    end
end

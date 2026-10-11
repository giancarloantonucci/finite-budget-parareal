# ErrorContours.jl
#
# Figure for fig:ErrorContoursLorenz: ‖U_n − F(U_{n−1})‖ for Lorenz-63 with
# U = (x, 3, −14), x ∈ [−50, 50]. One panel per ΔT; F is RK4 with h = 1e-4.
#
# Needs LinearAlgebra, NSDERungeKutta, Plots and LaTeXStrings.
# Run:     julia ErrorContours.jl
# Writes:  Figures/ErrorContoursLorenz.pdf

ENV["GKSwstype"] = "100"  # render without a display
using LinearAlgebra, NSDERungeKutta, Plots, LaTeXStrings
gr()

lorenz = Lorenz()  # (σ, ρ, β) = (10, 28, 8/3)
solver = RK4(h = 1e-4)

F(U, ΔT) = solve(remake(lorenz; u0 = U, tspan = (0.0, ΔT)), solver).u[end]

# Interface states; 401 points keep the band edges smooth.
xs = range(-50, 50; length = 401)
states = [[x, 3.0, -14.0] for x in xs]

# Z[i, j] = ‖U_n − F(U_{n−1})‖ with U_n = states[i], U_{n−1} = states[j].
function defects(ΔT)
    FUs = [F(U, ΔT) for U in states]
    return [norm(Uₙ - FUₙ₋₁) for Uₙ in states, FUₙ₋₁ in FUs]
end

chunks = [
    (1e-3, L"\Delta T = 10^{-3}"),
    (1e-2, L"\Delta T = 10^{-2}"),
    (1e-1, L"\Delta T = 10^{-1}"),
    (1.0, L"\Delta T = 1"),
    (10.0, L"\Delta T = 10"),
]
landscapes = [defects(ΔT) for (ΔT, _) in chunks]

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    titlefontsize = 9,
    framestyle = :box,
    tick_direction = :out,
    grid = false,
    margin = 1Plots.mm,
)

# Linear scale: for ΔT ≥ 1 every defect lies in about [17, 90], so a log scale
# would flatten those panels into the top bands.
hi = maximum(maximum, landscapes)
bands = cgrad(:blues, 20; categorical = true)

panels = [
    heatmap(
        xs, xs, Z;
        color = bands,
        clims = (0, hi),
        colorbar = false,
        title = title,
        xlabel = L"x \mathrm{\ of\ } U_{n-1}",
        ylabel = L"x \mathrm{\ of\ } U_n",
        aspect_ratio = :equal,
        xlims = (-50, 50),
        ylims = (-50, 50),
        xticks = [-50, 0, 50],
        yticks = [-50, 0, 50],
    )
    for (Z, (_, title)) in zip(landscapes, chunks)
]

# Shared key in the sixth cell.
key = plot(
    xlims = (0, 1),
    ylims = (0, 1),
    aspect_ratio = :equal,
    framestyle = :none,
    ticks = false,
    legend = false,
)
level(v) = v / hi
ramp = range(0, hi; length = 1000)
heatmap!(
    key, [0.33, 0.38], level.(ramp), repeat(ramp, 1, 2);
    color = bands,
    clims = (0, hi),
    colorbar = false,
)
plot!(key, [0.305, 0.405, 0.405, 0.305, 0.305], [0, 0, 1, 1, 0]; color = :black, linewidth = 0.5)
for v in 0:20:floor(Int, hi)
    plot!(key, [0.405, 0.43], [level(v), level(v)]; color = :black, linewidth = 0.5)
    annotate!(key, 0.445, level(v), text(L"%$v", 8, :left))
end
annotate!(key, 0.66, 0.5, text(L"\Vert U_n - \mathcal{F}(U_{n-1}) \Vert", 9, rotation = 90))

figure = plot(panels..., key; layout = (2, 3), size = (560, 370))
mkpath(joinpath(@__DIR__, "Figures"))
savefig(figure, joinpath(@__DIR__, "Figures", "ErrorContoursLorenz.pdf"))

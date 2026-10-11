# ShadowingSchematic.jl
#
# Schematic for fig:shadowing: a pseudo-orbit with interface jumps
# stays near an orbit from a different start, while the orbit from the
# prescribed start u₀ separates. A sketch, not a computed trajectory.
#
# Needs Plots and LaTeXStrings.
# Run:     julia ShadowingSchematic.jl
# Writes:  Figures/ShadowingSchematic.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Plots, LaTeXStrings
gr()

N = 4  # chunks of unit length
d₀ = 0.15  # offset of every chunk start above the nearby orbit
λ = log(9) / N  # offset growth rate
w = 0.3  # half-width of the nearby orbit's neighbourhood

# Toy flow in which offsets from the base orbit grow like e^{λt}; every curve is an orbit of it.
base(t) = sin(t) / 3
orbit(t, t₀, x₀) = base(t) + (x₀ - base(t₀)) * exp(λ * (t - t₀))
start(n) = base(n) + d₀
label(s, size, halign = :center) = text(s, size, halign, "Computer Modern")

t = range(0, N; length = 301)
axis_y = -0.7
legend_y = 1.35

default(linewidth = 1.6)
p = plot(
    size = (612, 186),  # keeps the scale the legend starts were set for
    xlims = (-0.22, N + 0.1),  # just clears u₀ and T₄
    ylims = (-1.06, 1.44),  # just clears the T_n labels and the legend
    framestyle = :none,
    ticks = false,
    legend = false,
    margin = 0Plots.mm,
)

plot!(p, t, base.(t) .+ w; fillrange = base.(t) .- w, fillalpha = 0.15, color = :green, linewidth = 0)
plot!(p, t, base.(t); color = :green, linestyle = :dash)
for n in 0:(N - 1)
    s = range(n, n + 1; length = 51)
    plot!(p, s, orbit.(s, n, start(n)); color = :blue)
    n < N - 1 && plot!(p, [n + 1, n + 1], [orbit(n + 1, n, start(n)), start(n + 1)]; color = :red)
end
plot!(p, t, orbit.(t, 0, start(0)); color = :gray, linestyle = :dot)

plot!(p, [0, N], [axis_y, axis_y]; color = :black, linewidth = 1)
for n in 0:N
    plot!(p, [n, n], [axis_y - 0.04, axis_y + 0.04]; color = :black, linewidth = 1)
    annotate!(p, n, axis_y - 0.23, label(L"T_{%$n}", 9))
end

scatter!(p, [0], [start(0)]; color = :black, markersize = 3.5, markerstrokewidth = 0)
scatter!(p, [0], [base(0)]; markercolor = :white, markerstrokecolor = :black, markersize = 3.5)
annotate!(p, -0.1, start(0), label(L"u_0", 9, :right))

# Hand-drawn legend: GR clips a one-row legend. Starts space the row evenly from T₀ to T₄.
entries = [
    (0.0, "nearby orbit", (color = :green, linestyle = :dash)),
    (0.935, "pseudo-orbit", (color = :blue,)),
    (1.881, "interface jump", (color = :red,)),
    (2.903, "prescribed-start orbit", (color = :gray, linestyle = :dot)),
]
for (x, name, style) in entries
    plot!(p, [x, x + 0.21], [legend_y, legend_y]; style...)
    annotate!(p, x + 0.265, legend_y, label(name, 8, :left))
end

mkpath(joinpath(@__DIR__, "Figures"))
savefig(p, joinpath(@__DIR__, "Figures", "ShadowingSchematic.pdf"))

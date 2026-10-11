# StatisticalFigures.jl
#
# Figure for fig:statistics, from the samples StatisticalRuns.jl kept: for one
# Lorenz-63 replicate at N = 128 (the first), the marginal densities of x, y and z
# for the serial fine reference, the trajectory the secant rule accepted and the
# independent fine baseline.
#
# Needs Serialization, Plots and LaTeXStrings.
# Run:     julia StatisticalFigures.jl
# Writes:  Figures/StatisticalFidelity.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Serialization, Plots, LaTeXStrings
gr()

portrait = deserialize(joinpath(@__DIR__, "Records", "portraits.jls"))["lorenz"]  # only load records you made yourself

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    linewidth = 1.2,
    framestyle = :box,
    tick_direction = :out,
    grid = false,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend = false,
    margin = 1Plots.mm,
)

rgb(r, g, b) = RGB(r / 255, g / 255, b / 255)
blue, orange = rgb(0, 84, 159), rgb(213, 94, 0)
# The line styles keep the trajectories apart in black and white.
trajectories = [
    (samples = portrait.fine, label = "serial fine", color = :black, linestyle = :solid),
    (samples = portrait.par, label = "accepted", color = blue, linestyle = :dash),
    (samples = portrait.base, label = "independent fine", color = orange, linestyle = :dot),
]
coordinates = (L"x", L"y", L"z")

# Histogram outlines on 60 bins shared by the three trajectories, as densities.
function outline(x, edges)
    counts = zeros(length(edges) - 1)
    for v in x
        counts[clamp(searchsortedlast(edges, v), 1, length(counts))] += 1
    end
    return repeat(edges; inner = 2)[2:(end - 1)], repeat(counts ./ (length(x) .* diff(edges)); inner = 2)
end
ranges = [extrema(vcat((t.samples[:, j] for t in trajectories)...)) for j in 1:3]
outlines = map(1:3) do j
    edges = range(ranges[j]...; length = 61)
    return [outline(t.samples[:, j], collect(edges)) for t in trajectories]
end
tens(lo, hi) = (10 * ceil(Int, lo / 10)):10:(10 * floor(Int, hi / 10))  # ticks every 10, so labels never touch
top = 1.1 * maximum(maximum(last(o)) for os in outlines for o in os)  # shared by the three panels
densities = map(1:3) do j
    p = plot(xlabel = coordinates[j], ylims = (0, top), xticks = tens(ranges[j]...),
             ylabel = j == 1 ? "Density" : "", yformatter = j == 1 ? :auto : _ -> "")  # one set of y labels
    for (t, (x, y)) in zip(trajectories, outlines[j])
        plot!(p, x, y; t.color, t.linestyle)
    end
    return p
end

# A row above the panels holding only the legend, so no panel's data are covered.
key = plot(framestyle = :none, ticks = false, grid = false, margin = 0Plots.mm,
           legend = :bottom, legend_column = length(trajectories))
for t in trajectories
    plot!(key, [NaN], [NaN]; t.label, t.color, t.linestyle)
end

# The first panel is wider by the room its y labels take, so the three plot areas match.
figure = plot(key, plot(densities...; layout = grid(1, 3; widths = [0.38, 0.31, 0.31]));
              layout = grid(2, 1; heights = [0.14, 0.86]), size = (560, 220))
mkpath(joinpath(@__DIR__, "Figures"))
savefig(figure, joinpath(@__DIR__, "Figures", "StatisticalFidelity.pdf"))

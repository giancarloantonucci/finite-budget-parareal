# LongChunkFigures.jl
#
# Figure for fig:submitted_grid, from Records/submitted_grid.jls (LongChunkRuns.jl):
# the fine batches K_F of Lorenz-63 under the three stopping rules on the long-chunk
# grids. Rows: growing horizon (N = 100), strong scaling (T = 500) and weak scaling
# (T = 5N); columns: ξ = 10 and 100. The vertical axes are logarithmic, so that one or
# two batches stand apart from the roughly N batches of standard stopping.
#
# Needs Serialization, Plots and LaTeXStrings.
# Run:     julia LongChunkFigures.jl
# Writes:  Figures/LongChunkGridComparison.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Serialization, Plots, LaTeXStrings
gr()

records = deserialize(joinpath(@__DIR__, "Records", "submitted_grid.jls"))  # only load records you made yourself
all(r -> r.status in ("criterion", "exact_prefix"), records) || error("a run failed or exhausted its budget; the figure shows accepted runs only")

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    titlefontsize = 9,
    linewidth = 1.2,
    markersize = 2,
    markerstrokewidth = 0.5,
    framestyle = :box,
    tick_direction = :out,
    grid = :y,
    gridalpha = 0.06,  # faint even once FinalizeFigures.py widens it to 1 pt
    gridlinewidth = 0.5,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    margin = 1Plots.mm,
)

rgb(r, g, b) = RGB(r / 255, g / 255, b / 255)
style(label, colour, marker) = (; label, marker, color = colour, markercolor = colour, markerstrokecolor = colour)

# The markers keep the rules apart in black and white; the styles match KvsTFigures.jl.
rules = [
    "standard" => style("standard", rgb(90, 90, 90), :square),
    "psi_lyap" => style(L"\psi\ (w = e^{\lambda \Delta T})", rgb(0, 140, 90), :diamond),
    "psi_lip" => style(L"\psi\ (\mathrm{secant}\ w)", rgb(0, 84, 159), :circle),
]
ξs = (10, 100)  # columns
# Rows: which runs, the horizontal coordinate, its label, ticks and limits.
families = [
    (r -> r.N == 100, r -> r.T, L"T \quad (N = 100)", 0:100:500, (-10, 510)),
    (r -> r.T == 500, r -> r.N, L"N \quad (T = 500)", [1; 20:20:100], (-1, 102)),
    (r -> r.T == 5 * r.N, r -> r.N, L"N \quad (T = 5N)", [1; 20:20:100], (-1, 102)),
]

panels = [begin
              (in_family, coordinate, label, ticks, limits) = family
              p = plot(title = family === first(families) ? L"\xi = %$ξ" : "", xlabel = label,
                       ylabel = ξ == first(ξs) ? L"K_F" : "", xticks = ticks, xlims = limits,
                       yscale = :log10, ylims = (0.8, 130),
                       yticks = ([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"]), legend = false)
              for (rule, series) in rules
                  runs = sort(filter(r -> r.xi == ξ && r.criterion == rule && in_family(r), records); by = coordinate)
                  plot!(p, coordinate.(runs), [r.fine_batches for r in runs]; series...)
              end
              p
          end for family in families for ξ in ξs]  # row by row

# A row above the panels holding only the legend, so no panel's data are covered.
key = plot(framestyle = :none, ticks = false, grid = false, margin = 0Plots.mm, legend = :bottom,
           legend_column = length(rules))
for (_, series) in rules
    plot!(key, [NaN], [NaN]; series...)
end

figure = plot(key, plot(panels...; layout = (length(families), length(ξs)));
              layout = grid(2, 1; heights = [0.05, 0.95]), size = (560, 620))
mkpath(joinpath(@__DIR__, "Figures"))
savefig(figure, joinpath(@__DIR__, "Figures", "LongChunkGridComparison.pdf"))

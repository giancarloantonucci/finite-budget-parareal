# KvsTFigures.jl
#
# Lorenz-63 fine batches K_F (fig:KvsTLorenz) and fine-work depth K_F ΔT
# (fig:SerialWork) under the four stopping rules, from the records of KvsTRuns.jl.
# Rows: growing horizon, strong and weak scaling; columns: ξ = 10 and 100. The
# vertical axes are logarithmic, as are the N axes, since N doubles; T on the
# growing horizon is linear.
#
# Needs Serialization, Plots and LaTeXStrings.
# Run:     julia KvsTFigures.jl
# Writes:  Figures/KvsT_Lorenz.pdf, Figures/KDTvsT_Lorenz.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Serialization, Plots, LaTeXStrings
gr()

records = deserialize(joinpath(@__DIR__, "Records", "scaling.jls"))  # only load records you made yourself
lorenz = filter(r -> r.problem == "lorenz", records)

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    titlefontsize = 9,
    linewidth = 1.2,
    markersize = 3,
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

# The markers keep the rules apart in black and white.
rules = [
    "standard" => style("standard", rgb(90, 90, 90), :square),
    "psi_w1" => style(L"\psi\ (w = 1)", rgb(213, 94, 0), :utriangle),
    "psi_lyap" => style(L"\psi\ (w = e^{\lambda \Delta T})", rgb(0, 140, 90), :diamond),
    "psi_lip" => style(L"\psi\ (\mathrm{secant}\ w)", rgb(0, 84, 159), :circle),
]

ξs = (10, 100)                             # columns
scenarios = ("growing", "strong", "weak")  # rows
xlabels = Dict(
    "growing" => L"T \quad (N = 32)",
    "strong" => L"N \quad (T = 128)",
    "weak" => L"N = T \quad (\Delta T = 1)",
)
xvalue(r) = r.scenario == "strong" ? r.N : r.T
Ns = sort(unique(r.N for r in lorenz if r.scenario == "strong"))  # 2, 4, …, 128

# Powers-of-two ticks covering `values`, every other one if there would be more
# than nine; the limits reach a quarter power beyond the end ticks.
function log2axis(values)
    lo, hi = floor(Int, log2(minimum(values))), ceil(Int, log2(maximum(values)))
    ks = lo:(hi - lo > 8 ? 2 : 1):hi
    labels = [k >= 0 ? string(2^k) : string(2.0^k) for k in ks]
    return (2.0 .^ ks, labels), (2.0^(lo - 0.25), 2.0^(hi + 0.25))
end

function panel(y, ylabel, ξ, scenario)
    yticks, ylims = log2axis([y(r) for r in lorenz if r.scenario == scenario])  # shared across the row
    doubling = scenario != "growing"
    p = plot(
        title = scenario == first(scenarios) ? L"\xi = %$ξ" : "",
        xlabel = xlabels[scenario],
        ylabel = ξ == first(ξs) ? ylabel : "",
        xscale = doubling ? :log2 : :identity,
        xticks = doubling ? (Ns, string.(Ns)) : 0:32:128,
        xlims = doubling ? (2^0.75, 2^7.25) : (0, 132),
        yscale = :log2,
        yticks = yticks,
        ylims = ylims,
        legend = false,
    )
    for (rule, series) in rules
        runs = sort(filter(r -> r.xi == ξ && r.scenario == scenario && r.criterion == rule, lorenz); by = xvalue)
        plot!(p, xvalue.(runs), y.(runs); series...)
    end
    return p
end

# A row above the panels holding only the legend, so no panel's data are covered.
function legend_row()
    key = plot(framestyle = :none, ticks = false, grid = false, margin = 0Plots.mm,
               legend = :bottom, legend_column = length(rules))
    for (_, series) in rules
        plot!(key, [NaN], [NaN]; series...)
    end
    return key
end

function scaling_figure(file, y, ylabel)
    panels = [panel(y, ylabel, ξ, scenario) for scenario in scenarios for ξ in ξs]  # row by row
    figure = plot(legend_row(), plot(panels...; layout = (length(scenarios), length(ξs)));
                  layout = grid(2, 1; heights = [0.05, 0.95]), size = (560, 620))
    savefig(figure, joinpath(@__DIR__, "Figures", file * ".pdf"))
end

mkpath(joinpath(@__DIR__, "Figures"))
scaling_figure("KvsT_Lorenz", r -> r.fine_batches, L"K_F")
scaling_figure("KDTvsT_Lorenz", r -> r.fine_batches * r.dt, L"K_F \, \Delta T")

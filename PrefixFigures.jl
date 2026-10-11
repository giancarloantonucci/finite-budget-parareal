# PrefixFigures.jl
#
# Figure for fig:prefix, from the records of PrefixRuns.jl: the per-interface error
# ‖e_n‖ = ‖U_n - U_n*‖ of each acceptance at N = 128 from the prescribed start, with RK4
# (top) and explicit Euler (bottom) coarse solvers, against n = 1, …, 128 on a
# logarithmic axis (U_0 is always exact). Zeros are drawn at 1e-15 so the
# logarithmic axis can show them; a failed run has no curve. Darker lines mark the
# thresholds 1e-3 and 1 behind the accurate-prefix lengths of tab:prefix_summary.
#
# Needs Serialization, StaticArrays, Plots and LaTeXStrings.
# Run:     julia PrefixFigures.jl
# Writes:  Figures/PrefixAccuracy.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Serialization, StaticArrays, Plots, LaTeXStrings  # the archived record holds SVectors
gr()

records = deserialize(joinpath(@__DIR__, "Records", "prefix_accuracy.jls"))  # only load records you made yourself

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    titlefontsize = 9,
    linewidth = 1.2,
    framestyle = :box,
    tick_direction = :out,
    grid = false,  # the threshold lines are the only horizontal lines
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend = false,
    margin = 1Plots.mm,
)

rgb(r, g, b) = RGB(r / 255, g / 255, b / 255)
# The line styles keep the acceptances apart in black and white.
rules = [
    "standard" => (label = "standard", color = rgb(90, 90, 90), linestyle = :dot),
    "psi_lyap" => (label = L"\psi\ (w = e^{\lambda \Delta T})", color = rgb(0, 140, 90), linestyle = :solid),
    "psi_lip" => (label = L"\psi\ (\mathrm{secant}\ w)", color = rgb(0, 84, 159), linestyle = :dashdot),
    "coarse" => (label = "coarse-only", color = rgb(213, 94, 0), linestyle = :dash),
]
zero_level = 1e-15  # where zeros are drawn

cases = [("RK4 scaling", "RK4 coarse solver"), ("Euler prescribed", "Explicit Euler coarse solver")]
panels = map(cases) do (name, title)
    p = plot(title = title, xlabel = name == last(cases)[1] ? L"n" : "",
             ylabel = L"\Vert e_n \Vert", xscale = :log2, xlims = (1, 128),
             xticks = (2 .^ (0:7), string.(2 .^ (0:7))),
             yscale = :log10, ylims = (zero_level / 3, 1e3), yticks = 10.0 .^ (-15:3:0))  # shared by both panels
    hline!(p, [1e-3, 1.0]; color = :black, alpha = 0.45, linewidth = 0.6)
    for (rule, style) in rules
        r = only(r for r in records if r.case == name && r.rule == rule)
        r.status == "nonfinite" && continue
        plot!(p, 1:(length(r.errors) - 1), max.(r.errors[2:end], zero_level); style...)
    end
    return p
end

# A row above the panels holding only the legend, so no panel's data are covered.
key = plot(framestyle = :none, ticks = false, margin = 0Plots.mm,
           legend = :bottom, legend_column = length(rules))
for (_, style) in rules
    plot!(key, [NaN], [NaN]; style...)
end

figure = plot(key, plot(panels...; layout = (2, 1)); layout = grid(2, 1; heights = [0.07, 0.93]), size = (560, 440))
mkpath(joinpath(@__DIR__, "Figures"))
savefig(figure, joinpath(@__DIR__, "Figures", "PrefixAccuracy.pdf"))

# PredictionFigures.jl
#
# Figure for fig:measured_vs_bound, from the records of PredictionRuns.jl: for each of
# the six main predictions (rows: Lorenz-63, logistic, periodic; columns: ξ = 10, 100),
# the measured stored-start error of every forced iterate, the sampled prediction
# ê₀ β̂_W^s and the target r, all in that case's fixed weighted norm. The region below
# the target is shaded: there the measured error sits at roundoff level, beyond what the
# prediction claims. Below the target the y-range reaches as many decades as above it
# (at least three) or down to the first measured error under the target, whichever is
# lower; later points and the prediction may run off the bottom. An exact zero leaves
# no point.
#
# Needs Serialization, Plots and LaTeXStrings.
# Run:     julia PredictionFigures.jl
# Writes:  Figures/MeasuredVsBound.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Serialization, Plots, LaTeXStrings
gr()

records = deserialize(joinpath(@__DIR__, "Records", "prediction.jls"))  # only load records you made yourself
problem_label = Dict("lorenz" => "Lorenz-63", "logistic" => "Logistic", "periodic" => "Periodic")

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    titlefontsize = 9,
    linewidth = 1.2,
    framestyle = :box,
    tick_direction = :out,
    grid = false,  # the target is the only horizontal line
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend = false,
    margin = 1Plots.mm,
)
rgb(r, g, b) = RGB(r / 255, g / 255, b / 255)
# Line styles and markers keep the curves apart in black and white.
measured_style = (label = "measured error", color = rgb(0, 84, 159), linestyle = :solid, marker = :circle,
                  markersize = 2.5, markerstrokewidth = 0)
predicted_style = (label = "sampled prediction", color = rgb(213, 94, 0), linestyle = :dash)
target_style = (label = "target", color = :black, linestyle = :dot, linewidth = 0.8)
below_style = (label = false, fillcolor = rgb(232, 232, 232), linecolor = rgb(232, 232, 232), linewidth = 0)

panels = map([(name, ξ) for name in ("lorenz", "logistic", "periodic") for ξ in (10, 100)]) do (name, ξ)
    r = only(r for r in records if r.problem == name && r.xi == ξ)
    s = 0:(length(r.errors) - 1)
    predicted = r.ehat .* r.beta .^ s
    values = filter(>(0), [r.errors; predicted[1]; r.r])
    high = 5 * maximum(values)
    k = findfirst(e -> 0 < e <= r.r, r.errors)
    first_below = k === nothing ? r.r : r.errors[k]
    low = max(minimum(values) / 20, min(r.r / 10^max(3, log10(high / r.r)), first_below / 20))
    # A plain string with a backslash is set as text with inline maths, so the hyphen stays a hyphen.
    p = plot(title = "$(problem_label[name]), \$\\xi = $ξ\$", xlabel = name == "periodic" ? L"s" : "",
             ylabel = ξ == 10 ? L"E_{\mathrm{start}}(U^s)" : "", yscale = :log10, ylims = (low, high),
             xlims = (-0.3, last(s) + 0.3))
    hspan!(p, [low, r.r]; below_style...)
    hline!(p, [r.r]; target_style...)
    plot!(p, s, predicted; predicted_style...)
    plot!(p, s, [e > 0 ? e : NaN for e in r.errors]; measured_style...)  # an exact zero has no place on a log axis
    return p
end

# A row above the panels holding only the legend, so no panel's data are covered.
key = plot(framestyle = :none, ticks = false, margin = 0Plots.mm, legend = :bottom, legend_column = 3)
for style in (measured_style, predicted_style, target_style)
    plot!(key, [NaN], [NaN]; style...)
end

figure = plot(key, plot(panels...; layout = (3, 2)); layout = grid(2, 1; heights = [0.05, 0.95]), size = (560, 620))
mkpath(joinpath(@__DIR__, "Figures"))
savefig(figure, joinpath(@__DIR__, "Figures", "MeasuredVsBound.pdf"))

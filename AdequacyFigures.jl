# AdequacyFigures.jl
#
# Figure for fig:stopping_adequacy, from the records of AdequacyRuns.jl: the Lorenz-63
# energy-estimation error of each forced iterate in the second confirmation, against its
# fine batches K_F = 1, …, 32, with the pointwise one-sided 95% upper bound and the fixed
# accuracy limit. Errors, bounds and the limit are as in AdequacyTables.jl. The legend
# sits in the panel's empty top-right corner.
#
# Needs Random, Serialization, Statistics, Plots and LaTeXStrings. Reads
# Records/reference_lorenz.jls (FunctionalRuns.jl).
# Run:     julia AdequacyFigures.jl
# Writes:  Figures/StoppingAdequacy.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Random, Serialization, Statistics, Plots, LaTeXStrings
gr()

load(name) = deserialize(joinpath(@__DIR__, "Records", name * ".jls"))  # only load records you made yourself
forced, calibration = load("adequacy_forced"), load("adequacy_calibration")
reference = filter(r -> r.h == 5e-5, load("reference_lorenz"))
draws_count, N, reps = 10000, maximum(r.s for r in forced) + 1, maximum(r.rep for r in forced)

# The reference energy, and its bootstrap draws over pairs of consecutive blocks within each chain.
energy(ss) = sum(s.energy for s in ss) / sum(s.n for s in ss)
centre = energy(reference)
sorted = sort(reference; by = r -> (r.chain, r.block))
blocks = [sorted[i:(i + 1)] for i in 1:2:length(sorted)]
rng = MersenneTwister(2026101010)
draws = [energy(reduce(vcat, blocks[rand(rng, 1:length(blocks), length(blocks))])) for _ in 1:draws_count]

limit = 1.25 * mean(abs(r.phi2 - centre) for r in calibration)
rng = MersenneTwister(2026101011)
idx = rand(rng, 1:reps, draws_count, reps)  # the same resampled starts as in AdequacyTables.jl
estimates(s) = [r.phi2 for r in sort([r for r in forced if r.s == s]; by = r -> r.rep)]
errors = [mean(abs.(estimates(s) .- centre)) for s in 0:(N - 1)]
uppers = map(0:(N - 1)) do s
    values = estimates(s)
    quantile([mean(abs.(values[idx[k, :]] .- draws[k])) for k in 1:draws_count], 0.95)
end

default(
    fontfamily = "Computer Modern",
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    linewidth = 1.2,
    framestyle = :box,
    tick_direction = :out,
    grid = false,  # the limit is the only horizontal line
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend = false,
    margin = 1Plots.mm,
)
rgb(r, g, b) = RGB(r / 255, g / 255, b / 255)
# Line styles and markers keep the curves apart in black and white.
mean_style = (label = "mean error", color = rgb(0, 84, 159), linestyle = :solid, marker = :circle, markersize = 2.5,
              markerstrokewidth = 0)
upper_style = (label = "pointwise 95% upper bound", color = rgb(213, 94, 0), linestyle = :dot)
limit_style = (label = "fixed accuracy limit", color = :black, linestyle = :dash, linewidth = 0.8)

figure = plot(xlabel = L"K_F", ylabel = L"E", xlims = (0.5, N + 0.5), xticks = [1; 8:8:N],
              legend = :topright, size = (560, 240))
# Drawn back to front without labels; empty series then give the legend in reading order.
hline!(figure, [limit]; limit_style..., label = false)
plot!(figure, 1:N, uppers; upper_style..., label = false)
plot!(figure, 1:N, errors; mean_style..., label = false)
for style in (mean_style, upper_style, limit_style)
    plot!(figure, [NaN], [NaN]; style...)
end
mkpath(joinpath(@__DIR__, "Figures"))
savefig(figure, joinpath(@__DIR__, "Figures", "StoppingAdequacy.pdf"))

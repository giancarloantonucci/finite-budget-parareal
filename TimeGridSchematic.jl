# TimeGridSchematic.jl
#
# Schematic for Figure 1 (fig:timegrid): fine steps h, coarse steps H = ξh,
# chunks ΔT = LH.
#
# Needs Plots and LaTeXStrings.
# Run:     julia TimeGridSchematic.jl
# Writes:  Figures/TimeGridSchematic.pdf

ENV["GKSwstype"] = "100"  # render without a display
using Plots, LaTeXStrings
gr()

# Ticks and labels follow from these.
N = 4
L = 2
ξ = 2
ΔT = 1.0
H = ΔT / L
h = H / ξ
T = N * ΔT

header = L"N=%$N,\quad L=%$L,\quad \xi=%$ξ"
label(s) = text(s, 10, "Computer Modern")

p = plot(
    size = (560, 120),
    xlims = (-0.12, T + 0.12),
    ylims = (-0.34, 0.65),  # just clears the T_n labels
    axis = false,
    ticks = false,  # else GR leaves room for tick labels
    framestyle = :none,
    legend = false,
    left_margin = 0Plots.mm,
    right_margin = 0Plots.mm,
    top_margin = 0Plots.mm,
    bottom_margin = 0Plots.mm,
)

# Axis, then ticks for h, H and ΔT; coarser ones drawn on top.
plot!(p, [0, T], [0, 0]; color = :black, linewidth = 1)
for x in 0:h:T
    plot!(p, [x, x], [-0.045, 0.045]; color = :blue, linewidth = 1)
end
for x in 0:H:T
    plot!(p, [x, x], [-0.075, 0.075]; color = :red, linewidth = 1.4)
end
for n in 0:N
    plot!(p, [n * ΔT, n * ΔT], [-0.11, 0.11]; color = :black, linewidth = 1.8)
    annotate!(p, n * ΔT, -0.24, label(L"T_{%$n}"))
end

# Header and a sample of each step, in the second and third chunks.
annotate!(p, T / 2, 0.54, label(header))
plot!(p, [ΔT, ΔT + H], [0.22, 0.22]; color = :red, linewidth = 2)
annotate!(p, ΔT + H / 2, 0.34, label(L"H"))
plot!(p, [2ΔT, 2ΔT + h], [0.22, 0.22]; color = :blue, linewidth = 2)
annotate!(p, 2ΔT + h / 2, 0.34, label(L"h"))

mkpath(joinpath(@__DIR__, "Figures"))
savefig(p, joinpath(@__DIR__, "Figures", "TimeGridSchematic.pdf"))

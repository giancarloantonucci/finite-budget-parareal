# PredictionTables.jl
#
# Tables of the sweep-budget predictions and implementation checks, from the records of
# PredictionRuns.jl, in paper order.
#   Article:    tab:prediction.
#   Supplement: tab:prediction_details, tab:prediction_error_link, tab:prediction_short,
#               tab:prediction_stage, tab:prediction_coarse, tab:prediction_coarse_ratios,
#               tab:roundoff and tab:refinement.
#   Printed:    the numbers the text quotes, including the linear checks and the periodic
#               orbit diagnostics, which the text gives without a table.
# A budget s is the smallest s with β^s ê₀ ≤ r ("--" when β ≥ 1 gives no contraction
# prediction); an observed crossing s_hit is the first iterate whose stored-start error
# meets r. The frozen-source budgets replace the sampled source by the frozen model's
# leading term and keep every other input.
#
# Needs Printf and Serialization.
# Run:     julia PredictionTables.jl
# Writes:  Tables/prediction_compact.tex, prediction_details.tex, prediction_error_link.tex,
#          prediction_short.tex, prediction_stage.tex, prediction_coarse.tex,
#          prediction_coarse_ratios.tex, roundoff.tex, refinement.tex

using Printf, Serialization

load(name) = deserialize(joinpath(@__DIR__, "Records", name * ".jls"))  # only load records you made yourself
problem_order = ("lorenz", "logistic", "periodic")
problem_label = Dict("lorenz" => "Lorenz-63", "logistic" => "Logistic", "periodic" => "Periodic")
predictions = [only(r for r in load("prediction") if r.problem == name && r.xi == ξ) for name in problem_order
               for ξ in (10, 100)]

function budget(β, e0, r)
    e0 <= r && return 0
    β == 0 && return 1
    β >= 1 && return nothing
    return max(0, ceil(Int, log(e0 / r) / -log(β)))
end
crossing(errors, r) = (k = findfirst(<=(r), errors); k === nothing ? nothing : k - 1)
transport(x, N) = N == 1 ? zero(x) : sum(x^i for i in 0:(N - 2))
frozen_source(h, ξ, L, μ, ν, b) = L * ξ * h^2 * (μ^2 * abs(b - ξ * b + (ξ - 1) / 2) + (ξ - 1) * ν / 2)

# The frozen-source factor and budget with the record's own transport, weight and ê₀.
function frozen(r)
    h = hasproperty(r, :h) ? r.h : r.problem == "logistic" ? 1e-3 : 1e-4  # older records lack h
    ΔT = hasproperty(r, :dt) ? r.dt : 1.0
    source = frozen_source(h, r.xi, round(Int, ΔT / (r.xi * h)), r.mu, r.nu, r.problem == "logistic" ? 1.0 : 0.5)
    β = source / r.w * transport(r.lambdaG / r.w, r.N)
    return (; source, β, sweeps = budget(β, r.ehat, r.r))
end

# Three significant figures. A column whose entries include one below 0.01 writes its
# entries below 1 as m × 10^e; every other entry is a plain decimal. Text passes through,
# a missing value is "--" and an exact zero is 0.
function scientific(x)
    mantissa, exponent = split(@sprintf("%.2e", x), 'e')
    e = parse(Int, exponent)
    return "\$$mantissa\\times 10^{$e}\$"
end
function decimal(x)
    mantissa, exponent = split(@sprintf("%.2e", x), 'e')
    e = parse(Int, exponent)
    return "\$" * Printf.format(Printf.Format("%.$(max(0, 2 - e))f"), parse(Float64, mantissa) * 10.0^e) * "\$"
end
function column(xs)
    scientific_column = any(x -> x isa AbstractFloat && isfinite(x) && 0 < abs(x) < 0.01, xs)
    return map(xs) do x
        x isa AbstractString && return x
        (x === nothing || (x isa AbstractFloat && !isfinite(x))) && return "--"
        x isa Integer && return "\$$x\$"
        iszero(x) && return raw"$0$"
        return scientific_column && abs(x) < 1 ? scientific(x) : decimal(x)
    end
end
# Settings in their shortest form, as powers of ten throughout when any is below 0.01.
function settings(xs)
    powers = any(<(0.01), xs)
    return map(xs) do x
        powers || return "\$$(replace(string(x), r"\.0$" => ""))\$"
        e = floor(Int, log10(x) + 1e-9)
        m = replace(string(round(x / 10.0^e; sigdigits = 6)), r"\.0$" => "")
        return m == "1" ? "\$10^{$e}\$" : "\$$m\\times 10^{$e}\$"
    end
end
# Rows of values formatted column by column; the columns in `setting_columns` are settings.
function format(rows; setting_columns = ())
    cols = [(j in setting_columns ? settings : column)([r[j] for r in rows]) for j in eachindex(first(rows))]
    return [[c[i] for c in cols] for i in eachindex(rows)]
end

multirow(n, x) = "\\multirow{$n}{*}{$x}"
row(cells) = join(cells, " & ")
heading(ncols, text) = "\\multicolumn{$ncols}{c}{\\textit{$text}}\\\\*\n"
# In the first `nlabels` columns, each run of equal labels (given the labels to its left)
# is written once, as a \multirow.
function merge_labels(cells, nlabels)
    out = deepcopy(cells)
    for j in 1:nlabels, i in eachindex(cells)
        same(k) = cells[k][1:j] == cells[i][1:j]
        if i > 1 && same(i - 1)
            out[i][j] = ""
        else
            n = something(findfirst(!same, i:length(cells)), length(cells) - i + 2) - 1
            out[i][j] = n == 1 ? cells[i][j] : multirow(n, cells[i][j])
        end
    end
    return out
end
# Rows grouped by key, with \\* inside a group so that pages break only between groups.
function grouped(lines, keys)
    ends = [i == length(lines) || keys[i + 1] != keys[i] for i in eachindex(lines)]
    return join((lines[i] * (ends[i] ? "\\\\\n" * (i < length(lines) ? "\\addlinespace[3pt]\n" : "") : "\\\\*\n")
                 for i in eachindex(lines)))
end
# Labelled rows, grouped by their first column.
labelled(cells, nlabels) = grouped(row.(merge_labels(cells, nlabels)), first.(cells))
# The six main predictions, each problem's two rows under its own heading.
by_problem(ncols, lines) = join((heading(ncols, problem_label[name]) *
                                 join((lines[i] for i in eachindex(predictions) if predictions[i].problem == name), "\\\\*\n") *
                                 "\\\\\n" for name in problem_order), "\\midrule\n")

tables = joinpath(@__DIR__, "Tables")
mkpath(tables)
write_tex(name, text) = write(joinpath(tables, name * ".tex"), "% Generated by PredictionTables.jl.\n" * text)
columns(align, natural) = (natural ? "@{}" : raw"@{\extracolsep{\fill}}") * align * "@{}"
spacing(sep) = "\\setlength{\\tabcolsep}{$(sep)pt}\\renewcommand{\\arraystretch}{1.15}"

# A supplement table at its natural width that may run over pages; the heading repeats
# and the page foot carries no rule of its own.
function supplement_table(name, caption, align, head, body; sep = 10)
    ncols = length(align)
    write_tex(name, "\\Needspace{20\\baselineskip}\\begingroup\\small" * spacing(sep) * "\n" *
                    "\\begin{longtable}{$(columns(align, true))}\n" *
                    "\\caption{\\rev{$caption}}\\label{tab:$name}\\\\\n" *
                    "\\toprule\n" * head * "\\midrule\\endfirsthead\n" *
                    "\\multicolumn{$ncols}{l}{\\tablename~\\thetable\\ (continued)}\\\\\n" *
                    "\\toprule\n" * head * "\\midrule\\endhead\n" *
                    "\\multicolumn{$ncols}{r}{Continued on next page}\\\\\\endfoot\n" *
                    "\\bottomrule\\endlastfoot\n" * body * "\\end{longtable}\\endgroup\n")
end

# Article: tab:prediction, whose float and caption are in the paper.
main_cells = format([Any[r.xi, r.w, r.source, frozen(r).source, r.beta, r.ehat, budget(r.beta, r.ehat, r.r),
                      frozen(r).sweeps, crossing(r.errors, r.r)] for r in predictions])
write_tex("prediction_compact",
          spacing(1.5) * "\n\\begin{tabular*}{\\textwidth}{$(columns("c"^9, false))}\\toprule\n" *
          raw"& & \multicolumn{2}{c}{Source} & & & \multicolumn{3}{c}{Sweeps}\\\cmidrule(lr){3-4}\cmidrule(lr){7-9}" * "\n" *
          raw"$\xi$ & $w$ & $\hat S_{\mathrm{map}}$ & $\hat S_{\mathrm{fr}}$ & $\hat\beta_W$ & $\hat e_0$ & $s_{\mathrm{pred}}$ & $s_{\mathrm{fr}}$ & $s_{\mathrm{hit}}$" *
          "\\\\\n\\midrule\n" * by_problem(9, row.(main_cells)) * "\\bottomrule\\end{tabular*}\n")

samples = only(unique(r.derivative_samples for r in predictions))
supplement_table("prediction_details",
                 "Sampled derivative diagnostics for \\Cref{tab:prediction}. The vector-field estimates use every " *
                 "coarse-step node of the coarse pass; the chunk-map estimates use $samples of the $(only(unique(r.N for r in predictions))) " *
                 "coarse-pass chunk starts, and their maxima are inflated by 1.25. " *
                 raw"Maxima use the state-induced matrix norm; hats denote sampled estimates.",
                 "c"^6, raw"& \multicolumn{2}{c}{Vector field} & \multicolumn{3}{c}{Chunk maps}\\\cmidrule(lr){2-3}\cmidrule(lr){4-6}" *
                        "\n" * raw"$\xi$ & $\hat\mu$ & $\hat\nu$ & $\hat\Lambda_{\mathcal F}$ & $\hat\Lambda_{\mathcal G}$ & $\hat S_{\mathrm{map}}$" * "\\\\\n",
                 by_problem(6, row.(format([Any[r.xi, r.mu, r.nu, r.lambdaF, r.lambdaG, r.source] for r in predictions]))))

all(budget(r.beta, r.elook, r.r) == budget(r.beta, r.ehat, r.r) for r in predictions) ||
    error("a one-look budget differs from the transported one; tab:prediction_error_link needs its budget columns back")
supplement_table("prediction_error_link",
                 raw"Transported-defect and one-look initial-error estimates. The relative excess is $g=100[C_N\psi(U^0)/\hat e_0-1]$. Using $C_N\psi(U^0)$ in place of $\hat e_0$ leaves every budget in \Cref{tab:prediction} unchanged.",
                 "c"^4, raw"& \multicolumn{2}{c}{Initial-error estimate} & \\\cmidrule(lr){2-3}" *
                        "\n" * raw"$\xi$ & $\hat e_0$ & $C_N\psi(U^0)$ & $g$ (\%)" * "\\\\\n",
                 by_problem(4, row.(format([Any[r.xi, r.ehat, r.elook, 100 * (r.elook / r.ehat - 1)] for r in predictions]))))

short = sort(load("prediction_short"); by = r -> (r.xi, -r.h))
supplement_table("prediction_short",
                 raw"Lorenz-63 short-window comparison, $N=32$, RK4, start $(2,3,-14)$, and target $r=10^{-9}$. Each pair halves $h,H$ at fixed $L,\xi,N$ ($L=20$ for $\xi=10$, $L=2$ for $\xi=100$). Sixteen derivative samples and inflation 1.25 give $\hat\beta_W$. Replacing only the source by the frozen leading term gives $\hat\beta_{\mathrm{fr}}$ and the uncapped budget $s_{\mathrm{fr}}$, retaining the same weight and initial-error estimate. A dash means no contraction prediction. The observed crossing $s_{\mathrm{hit}}$ uses this same fixed stored-start norm.",
                 "c"^9, raw"& & & & \multicolumn{2}{c}{Contraction factor} & \multicolumn{3}{c}{Sweeps}\\\cmidrule(lr){5-6}\cmidrule(lr){7-9}" *
                        "\n" * raw"$\xi$ & $h$ & $\Delta T$ & $\hat\mu\Delta T$ & $\hat\beta_W$ & $\hat\beta_{\mathrm{fr}}$ & $s_{\mathrm{pred}}$ & $s_{\mathrm{fr}}$ & $s_{\mathrm{hit}}$" * "\\\\\n",
                 labelled(format([Any[r.xi, r.h, r.dt, r.mu * r.dt, r.beta, frozen(r).β, budget(r.beta, r.ehat, r.r),
                                   frozen(r).sweeps, crossing(r.errors, r.r)] for r in short]; setting_columns = (2, 3)), 1);
                 sep = 5)

supplement_table("prediction_stage",
                 raw"Coarse-node RK4-transport diagnostic for the same short windows. The sampled $\hat\mu$ gives stage majorants and weight $w_{\mathrm{RK}}$ by \Cref{eq:rk4_transport_diagnostic}; the frozen source gives $\hat\beta_{\mathrm{RK}}$, and all first-batch defects give $\hat e_{0,\mathrm{RK}}$. Predicted and observed counts use this new fixed weight. Budgets are uncapped; native exact completion supplies the $N-1=31$-correction fallback. These are sampled diagnostics, not uniform enclosures.",
                 "c"^7, raw"& & & & & \multicolumn{2}{c}{Sweeps}\\\cmidrule(lr){6-7}" * "\n" *
                        raw"$\xi$ & $h$ & $w_{\mathrm{RK}}$ & $\hat\beta_{\mathrm{RK}}$ & $\hat e_{0,\mathrm{RK}}$ & $s_{\mathrm{RK}}$ & $s_{\mathrm{hit},\mathrm{RK}}$" * "\\\\\n",
                 labelled(format([Any[r.xi, r.h, r.stage_w, r.stage_beta, r.stage_ehat, budget(r.stage_beta, r.stage_ehat, r.r),
                                   crossing(r.stage_errors, r.r)] for r in short]; setting_columns = (2,)), 1))

# Matched coarse starts. A nonfinite coarse pass leaves nothing to sample, so its row is left out.
coarse = sort(load("prediction_coarse"); by = r -> (r.N, r.H, string(r.method) == "euler"))
coarse_name(r) = string(r.method) == "rk4" ? "RK4" : "Euler"
sampled = filter(r -> r.w !== nothing, coarse)
failed = join(unique("$(coarse_name(r)) at \$H=$(r.H)\$" for r in coarse if r.w === nothing), " and ")
underflow = [r for r in sampled if exp(-r.N * log(r.w)) == 0]
underflow_N, underflow_H = join(unique(r.N for r in underflow), ", "), join(unique(r.H for r in underflow), ", ")
supplement_table("prediction_coarse",
                 raw"Matched coarse-start predictions for Lorenz-63, prescribed start $(2,3,-14)$, $\Delta T=1$, fine RK4 with $h=10^{-4}$, and $r=10^{-9}$. Sixteen coarse-pass derivative samples, inflated by 1.25, define the row's fixed weight $w$ and factor. All initial defects define $\hat e_0$. Predicted and observed crossings use that same weighted error, not the stopping rule. A dash means no contraction prediction. " *
                 "The coarse passes of $failed are nonfinite, so those rows are left out. Weights differ between rows; the final discount factor \$w^{-N}\$ " *
                 "for RK4 at \$N=$underflow_N\$, \$H=$underflow_H\$ underflows in Float64.",
                 "c"^8, raw"& & & & & & \multicolumn{2}{c}{Sweeps}\\\cmidrule(lr){7-8}" * "\n" *
                        raw"$N$ & $H$ & Coarse & $w$ & $\hat\beta_W$ & $\hat e_0$ & $s_{\mathrm{pred}}$ & $s_{\mathrm{hit}}$" * "\\\\\n",
                 labelled(format([Any[r.N, r.H, coarse_name(r), r.w, r.beta, r.ehat, budget(r.beta, r.ehat, r.r),
                                   crossing(r.errors, r.r)] for r in sampled]; setting_columns = (2,)), 2))

ratio(e, s) = s + 2 <= length(e) && e[s + 1] > 0 ? e[s + 2] / e[s + 1] : NaN  # q_s = E_{s+1}/E_s
forced(r) = string(r.status) in ("exact", "exact_prefix") ? "exact" : "nonfinite after \$$(length(r.lyap_errors) - 1)\$"
supplement_table("prediction_coarse_ratios",
                 raw"Common-norm error ratios for the coarse starts in \Cref{tab:prediction_coarse}. Here $E_s=E_{\mathrm{start}}(U^s)$ as defined in \Cref{sec:stored_starts} with fixed $w=e^{0.9056}$, $q_s=E_{s+1}/E_s$, and $s_{\mathrm{hit,L}}$ is the first crossing of $10^{-9}$ in this norm. These crossings must not be compared with predictions in a different norm. Runs are forced beyond acceptance to exact completion at $N-1$ corrections or to failure; the last column gives the outcome and, for a failure, the last finite sweep. Full histories are in the records, including below-target ratios that may be dominated by roundoff.",
                 "c"^8, raw"& & & \multicolumn{3}{c}{Consecutive-error ratios} & & \\\cmidrule(lr){4-6}" * "\n" *
                        raw"$N$ & $H$ & Coarse & $q_0$ & $q_1$ & $q_2$ & $s_{\mathrm{hit,L}}$ & Forced run" * "\\\\\n",
                 labelled(format([Any[r.N, r.H, coarse_name(r), (ratio(r.lyap_errors, s) for s in 0:2)...,
                                   crossing(r.lyap_errors, r.r), forced(r)] for r in sampled]; setting_columns = (2,)), 2);
                 sep = 5)

roundoff = load("roundoff")
roundoff_N = sort(unique(r.N for r in roundoff))
roundoff_error(N, naive) = only(r.error for r in roundoff if r.N == N && r.naive == naive)
supplement_table("roundoff",
                 raw"Completion audit using native RK4 maps. The production solver completes in $N$ fine batches; the deliberately unparenthesized full-interface negative control performs $N$ corrections. The negative control's error is not a contraction-roundoff floor.",
                 "l" * "c"^length(roundoff_N),
                 "& \\multicolumn{$(length(roundoff_N))}{c}{\$\\|U-U^*\\|_2\$}\\\\\\cmidrule(lr){2-$(length(roundoff_N) + 1)}\n" *
                 row([""; ["\$N=$N\$" for N in roundoff_N]]) * "\\\\\n",
                 join(row.(format([Any[label; [roundoff_error(N, naive) for N in roundoff_N]]
                                   for (label, naive) in (("Production solver", false), ("Negative control", true))])), "\\\\\n") * "\\\\\n")

# Refinement at H = ξh, one column per problem, with the least-squares slope of
# log(mismatch) against log(H) in the last row.
refinement = load("refinement")
refinement_problems = ("logistic", "lorenz")
refined(name) = sort([r for r in refinement if r.problem == name]; by = r -> r.xi)
refinement_ξ = only(unique([[r.xi for r in refined(name)] for name in refinement_problems]))
fine_steps = settings([only(unique(round(r.H / r.xi; sigdigits = 6) for r in refined(name))) for name in refinement_problems])
function slope(rs)
    x, y = log.([r.H for r in rs]), log.([r.source for r in rs])
    return sum((x .- sum(x) / length(x)) .* (y .- sum(y) / length(y))) / sum(abs2, x .- sum(x) / length(x))
end
mismatch = raw"$\max\|D\mathcal F-D\mathcal G\|$"
supplement_table("refinement",
                 "Actual derivative mismatch $mismatch under coarse refinement on identical sampled states, at \$H=\\xi h\$ with " *
                 "\$h=$(strip(fine_steps[1], '\$'))\$ (logistic) and \$h=$(strip(fine_steps[2], '\$'))\$ (Lorenz-63). " *
                 "The last row gives the least-squares slope of the logarithm of the mismatch against \$\\log H\$.",
                 "c"^3, "& \\multicolumn{2}{c}{$mismatch}\\\\\\cmidrule(lr){2-3}\n" *
                        row(["\$\\xi\$"; [problem_label[name] for name in refinement_problems]]) * "\\\\\n",
                 join(row.(format([Any[ξ; [only(r.source for r in refined(name) if r.xi == ξ) for name in refinement_problems]]
                                   for ξ in refinement_ξ])), "\\\\*\n") * "\\\\\n\\midrule\n" *
                 row(["Slope"; column([slope(refined(name)) for name in refinement_problems])]) * "\\\\\n")

# The numbers the text quotes.
println("Main predictions (s_pred, s_fr, s_hit; frozen factor; μ̂; one-look excess g in %):")
for r in predictions
    f = frozen(r)
    @printf("  %-9s ξ = %-3d  w = %-8.4g %s, %s, %s;  β_fr = %.3g;  μ = %.3g;  g = %.3g\n", problem_label[r.problem], r.xi,
            r.w, something(budget(r.beta, r.ehat, r.r), "--"), something(f.sweeps, "--"),
            something(crossing(r.errors, r.r), "--"), f.β, r.mu, 100 * (r.elook / r.ehat - 1))
end
l10 = only(r for r in predictions if r.problem == "lorenz" && r.xi == 10)
@printf("Lorenz-63, ξ = 10: μ̂ = %.3g, ν̂ = %.3g, Λ = %.3g, S_fr = %.3g, S_map = %.3g\n", l10.mu, l10.nu, l10.lambdaF,
        frozen(l10).source, l10.source)
l100 = only(r for r in predictions if r.problem == "lorenz" && r.xi == 100)
@printf("Lorenz-63, ξ = 100: w^-4 = %.2g\n", l100.w^-4)
println("ê₀ over the measured initial error; β̂_W over the measured reduction per sweep up to the crossing;")
println("budget from the measured initial error:")
for r in predictions
    s = crossing(r.errors, r.r)
    rate = s === nothing || s == 0 ? NaN : (r.errors[s + 1] / r.errors[1])^(1 / s)
    @printf("  %-9s ξ = %-3d  %.3g  %.3g  %s\n", problem_label[r.problem], r.xi, r.ehat / r.errors[1], r.beta / rate,
            something(budget(r.beta, r.errors[1], r.r), "--"))
end
println("Short windows (ξ, h: β_fr, s_fr; s_RK, s_hit,RK):")
for r in short
    println("  ", r.xi, ", ", r.h, ": ", @sprintf("%.3g", frozen(r).β), ", ", something(frozen(r).sweeps, "--"), "; ",
            something(budget(r.stage_beta, r.stage_ehat, r.r), "--"), ", ", something(crossing(r.stage_errors, r.r), "--"))
end
println("Linear checks (a, N: β_W, s_pred, s_hit):")
for r in sort(load("prediction_linear"); by = r -> (r.a, r.N))
    @printf("  %g, %d: %.3g, %d, %d\n", r.a, r.N, r.beta, r.kpred, r.kobs)
end
periodic = load("periodic_diagnostics")
@printf("Periodic orbit: period %.12g, return error %.3g, one-chunk amplification %.3g, largest multi-chunk amplification %.3g\n",
        periodic.period, periodic.return_error, periodic.growth[1], maximum(periodic.growth[2:end]))
println("Floquet multipliers: ", join((@sprintf("%.3g", real(z)) * (isreal(z) ? "" : @sprintf("%+.3gi", imag(z)))
                                       for z in periodic.multipliers), ", "))

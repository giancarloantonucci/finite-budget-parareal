# ConfirmationTables.jl
#
# The table of the independent Lorenz-96 confirmation (sec:functional_confirmation),
# from the records of ConfirmationRuns.jl and the h = 5e-5 long reference of
# FunctionalRuns.jl.
#   Supplement: tab:functional_confirmation.
#   Printed:    the numbers the text quotes: outcomes, fine batches, energy errors,
#               the primary contrast, the secondary intervals and the costs.
# Errors are mean absolute differences from the reference over the 32 starts. Intervals
# are bootstrap 95% intervals with 10,000 draws; each draw resamples the starts (the same
# draw for every method, so contrasts are paired) and, independently, pairs of
# consecutive 128-time-unit reference blocks within the reference chains.
# Cost models: C = (K_F + ζ K_G)/N and Q = (J_F + ζ K_G)/N with ζ = 0.01 for Parareal;
# C = Q = 1e-4/h for serial RK4 at step h.
#
# Needs Printf, Random, Serialization and Statistics.
# Run:     julia ConfirmationTables.jl
# Writes:  Tables/functional_confirmation.tex

using Printf, Random, Serialization, Statistics

records = deserialize(joinpath(@__DIR__, "Records", "confirmation.jls"))  # only load records you made yourself
reference = filter(r -> r.h == 5e-5, deserialize(joinpath(@__DIR__, "Records", "reference_lorenz96.jls")))

N, ζ, draws_count = 128, 0.01, 10000
method_order = ("coarse", "serial_half", "psi_lyap", "psi_lip")
method_label = Dict("coarse" => "coarse-only", "serial_half" => raw"serial RK4 ($h=0.005$)",
                    "psi_lyap" => raw"$\psi$ ($w=e^{\lambda\Delta T}$)", "psi_lip" => raw"$\psi$ (secant $w$)")
observable_name = ("mean", "energy", "neighbour product")
contrasts = (("coarse", ("psi_lyap", "psi_lip", "serial_half")), ("serial_half", ("psi_lyap", "psi_lip")))
primary = ("psi_lyap", "coarse", 2)  # method, baseline, observable: fixed before the starts were drawn

# The observables from pooled sufficient statistics of reference blocks (as in FunctionalRuns.jl).
function observables(ss)
    n = sum(s.n for s in ss)
    return [sum(s.sum1 for s in ss) / n, sum(s.energy for s in ss) / n, sum(s.neighbour for s in ss) / n]
end
centre = observables(reference)

# Bootstrap draws (one row each) of the reference, resampling pairs of consecutive blocks within each chain.
function reference_draws(rows; seed = 584)
    sorted = sort(rows; by = r -> (r.chain, r.block))
    blocks = [sorted[i:(i + 1)] for i in 1:2:length(sorted)]
    all(b[1].chain == b[2].chain for b in blocks) || error("a block pair crosses chains")
    rng, n = MersenneTwister(seed), length(blocks)
    return permutedims(stack(observables(reduce(vcat, blocks[rand(rng, 1:n, n)])) for _ in 1:draws_count))
end
draws = reference_draws(reference)

runs = Dict(m => sort(filter(r -> r.method == m, records); by = r -> r.rep) for m in method_order)
reps = length(runs["coarse"])
all([r.rep for r in runs[m]] == 1:reps for m in method_order) || error("unpaired confirmation records")
phi(m, j) = [getproperty(r, Symbol(:phi, j)) for r in runs[m]]
failed(m) = count(r -> !(isfinite(r.phi1) && isfinite(r.phi2) && isfinite(r.phi3)), runs[m])
error_of(m, j) = failed(m) == 0 ? mean(abs.(phi(m, j) .- centre[j])) : NaN
serial(m) = !startswith(m, "psi")
cost(m) = mean(serial(m) ? 1e-4 / r.h : (r.K + ζ * r.coarse_calls) / N for r in runs[m])
work(m) = mean(serial(m) ? 1e-4 / r.h : (r.fine_calls + ζ * r.coarse_calls) / N for r in runs[m])

# Bootstrap errors for every method and observable, all on the same resampled starts.
rng = MersenneTwister(912610)
indices = [rand(rng, 1:reps, reps) for _ in 1:draws_count]
boot = Dict((m, j) => failed(m) == 0 ? [mean(abs.(phi(m, j)[ii] .- draws[k, j])) for (k, ii) in enumerate(indices)] :
                      fill(NaN, draws_count) for m in method_order for j in 1:3)
paired(m, base, j) = (d = boot[(m, j)] - boot[(base, j)]; all(isfinite, d) ? quantile(d, [0.025, 0.975]) : [NaN, NaN])

# Three significant figures. A column whose entries include one below 0.01 writes its
# entries below 1 as m × 10^e; every other entry is a plain decimal. `column` formats a
# whole column, `num(x, scientific(xs))` one entry of the column xs; a lone value goes by
# its own size.
scientific(xs) = any(x -> isfinite(x) && 0 < abs(x) < 0.01, xs)
function decimal(x)
    iszero(x) && return raw"$0$"
    mantissa, exponent = split(@sprintf("%.2e", x), 'e')
    e = parse(Int, exponent)
    return "\$" * Printf.format(Printf.Format("%.$(max(0, 2 - e))f"), parse(Float64, mantissa) * 10.0^e) * "\$"
end
function num(x, sci = false)
    isfinite(x) || return "--"
    iszero(x) && return raw"$0$"
    mantissa, exponent = split(@sprintf("%.2e", x), 'e')
    e = parse(Int, exponent)
    e < -2 || (sci && e < 0) || return decimal(x)
    return "\$$mantissa\\times 10^{$e}\$"
end
column(xs) = (sci = scientific(xs); [num(x, sci) for x in xs])
# An interval, each end to three significant figures in plain decimals.
bracket((lo, hi)) = isfinite(lo) && isfinite(hi) ? "[$(decimal(lo)), $(decimal(hi))]" : "--"
function batches(m)
    serial(m) && return ""
    low, high = extrema(r.K for r in runs[m])
    return low == high ? "\$$low\$" : "\$$low\$--\$$high\$"
end

# A supplement table at the text width that may run over pages, starting only where its
# heading fits: an optional caption, a title above the rules, a repeated heading, and no
# rule in the page foot.
# longtable steps the table counter even without a caption, so an uncaptioned panel steps
# it back and keeps its table's number, with its own hyperref anchor.
same_number = "\\addtocounter{table}{-1}\\def\\theHtable{\\arabic{table}.\\number\\value{LT@tables}}"
function panel(ncols, title, head, body; caption = "")
    return "\\Needspace{20\\baselineskip}" * (isempty(caption) ? same_number : "") * "\\begin{longtable}{@{\\extracolsep{\\fill}}" * "c"^ncols * "@{}}\n" * caption *
           "\\multicolumn{$ncols}{@{}l}{\\textbf{$title}}\\\\\n\\toprule\n" * head * "\\midrule\\endfirsthead\n" *
           "\\multicolumn{$ncols}{l}{\\tablename~\\thetable\\ (continued)}\\\\\n" *
           "\\toprule\n" * head * "\\midrule\\endhead\n" *
           "\\multicolumn{$ncols}{r}{Continued on next page}\\\\\\endfoot\n" *
           "\\bottomrule\\endlastfoot\n" * body * "\\end{longtable}\n"
end

# The weighted rules share a row when their runs agree at every start, and the Failed
# column appears only when some run failed.
outcome(r) = (r.K, r.fine_calls, r.coarse_calls, r.phi1, r.phi2, r.phi3)
same_runs = all(isequal(outcome(a), outcome(b)) for (a, b) in zip(runs["psi_lyap"], runs["psi_lip"]))
shown = same_runs ? filter(!=("psi_lip"), method_order) : method_order
label(m) = same_runs && m == "psi_lyap" ? raw"$\psi$ (either $w$)" : method_label[m]
any_failed = any(failed(m) > 0 for m in method_order)

caption = "Independent Lorenz-96 confirmation at \$T = N = $N\$: $reps fresh matched starts. " *
          raw"Panel (a): mean absolute errors against long-reference estimates, fine batches $K_F$, modelled elapsed cost $\mathcal{C}$" *
          (any_failed ? raw", total work $\mathcal{Q}$, and failed runs. " : raw" and total work $\mathcal{Q}$. All runs are finite. ") *
          raw"Panel (b): paired 95\% intervals for error differences (method minus baseline); negative intervals favour the method. The starred energy contrast against coarse-only, prespecified for $\psi$ ($w=e^{\lambda\Delta T}$), is primary; all others are secondary pointwise comparisons. Resampling includes matched starts and long-reference uncertainty. " *
          (same_runs ? "Both weighted rules were run separately; they take the same fine batches and give identical estimates at every start, so they share one row." :
                       "Both acceptance rules are evaluated separately.")
outcome_head = join(["Method", raw"$K_F$", raw"$\mathcal{C}$", raw"$\mathcal{Q}$", (any_failed ? ["Failed"] : [])...,
                     "Mean", "Energy", "Neighbour"], " & ") * "\\\\\n"
outcome_columns = [column([value(m) for m in shown]) for value in (cost, work, (m -> error_of(m, j) for j in 1:3)...)]
outcomes = join((join([label(m), batches(m), outcome_columns[1][i], outcome_columns[2][i],
                       (any_failed ? ["$(failed(m))/$reps"] : [])..., (c[i] for c in outcome_columns[3:5])...], " & ")
                 for (i, m) in enumerate(shown)), "\\\\*\n") * "\\\\\n"
differences = join(map(contrasts) do (base, ms)
    rows = map(filter(in(shown), ms)) do m
        cells = [bracket(paired(m, base, j)) * ((m, base, j) == primary ? raw"$^{*}$" : "") for j in 1:3]
        join([label(m); cells], " & ")
    end
    "\\multicolumn{4}{c}{\\textit{Against $(label(base))}}\\\\*\n" * join(rows, "\\\\*\n") * "\\\\\n"
end, "\\addlinespace[3pt]\n")

tables = joinpath(@__DIR__, "Tables")
mkpath(tables)
write(joinpath(tables, "functional_confirmation.tex"),
      "% Generated by ConfirmationTables.jl.\n" *
      "\\begingroup\\small\\setlength{\\tabcolsep}{1.5pt}\\renewcommand{\\arraystretch}{1.15}" *
      "\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}\n" *
      panel(any_failed ? 8 : 7, "(a) Outcomes and cost", outcome_head, outcomes;
            caption = "\\caption{\\rev{$caption}}\\label{tab:functional_confirmation}\\\\\n") *
      panel(4, "(b) Paired error differences", "Method & Mean & Energy & Neighbour\\\\\n", differences) *
      "\\endgroup\n")

println("Runs with finite estimates: ", sum(reps - failed(m) for m in method_order), " of ", reps * length(method_order))
for m in filter(!serial, method_order)
    println("Fine batches, ", m, ": ", batches(m))
end
println("Weighted rules give identical estimates at every start: ", all(phi("psi_lyap", j) == phi("psi_lip", j) for j in 1:3))
println("\nErrors (mean, energy, neighbour product) and costs C, Q:")
for m in method_order
    e = [error_of(m, j) for j in 1:3]
    @printf("  %-12s  %.3g  %.3g  %.3g   C = %.3g  Q = %.3g\n", m, e[1], e[2], e[3], cost(m), work(m))
end
let (m, base, j) = primary, (lo, hi) = paired(m, base, j)
    @printf("\nPrimary contrast (%s minus %s, energy): %+.3g  [%.3g, %.3g]\n", m, base,
            error_of(m, j) - error_of(base, j), lo, hi)
end
println("Paired intervals, method minus baseline:")
for (base, ms) in contrasts, m in ms, j in 1:3
    lo, hi = paired(m, base, j)
    @printf("  %-12s  %-12s  %-17s  %+.3g  [%.3g, %.3g]%s\n", m, base, observable_name[j],
            error_of(m, j) - error_of(base, j), lo, hi, lo <= 0 <= hi ? "" : "  excludes 0")
end

# AdequacyTables.jl
#
# Tables of the Lorenz-63 stopping-adequacy study (sec:stopping_adequacy), from the
# records of AdequacyRuns.jl, and the article's summary of both independent functional
# tests, whose Lorenz-96 rows come from the records of ConfirmationRuns.jl.
#   Article:    tab:functional_main.
#   Supplement: tab:stopping_adequacy.
#   Printed:    the numbers the text quotes: the calibrated limits, the pilot and its
#               choice, the first confirmation, the feasibility screen and its choice,
#               and the second confirmation's errors, bounds, contrasts (against coarse-only
#               and, as a secondary check, the first fine batch) and costs.
# Errors are mean absolute differences from the h = 5e-5 long reference over the starts.
# Bounds and intervals come from 10,000 bootstrap draws; each resamples the starts (the
# same draw for every method, so contrasts are paired) and, independently, pairs of
# consecutive 128-time-unit reference blocks within the chains. L and U are one-sided 95%
# bounds. Cost models: C = (K_F + ζ K_G)/N and Q = (J_F + ζ K_G)/N for Parareal, and
# C = Q = 1e-4/h for serial RK4 at step h; ζ = 1e-4/H for RK4 coarse chunks and a
# quarter of that for explicit Euler, which has one stage to RK4's four. Lorenz-63
# energy errors keep three decimals, the precision of the limit they are tested against.
#
# Needs Printf, Random, Serialization and Statistics. Reads Records/reference_lorenz.jls
# and reference_lorenz96.jls (FunctionalRuns.jl) and confirmation.jls (ConfirmationRuns.jl).
# Run:     julia AdequacyTables.jl
# Writes:  Tables/stopping_adequacy.tex, Tables/functional_main.tex

using Printf, Random, Serialization, Statistics

load(name) = deserialize(joinpath(@__DIR__, "Records", name * ".jls"))  # only load records you made yourself
draws_count = 10000
ζ(method, h) = 1e-4 / h / (method == "euler" ? 4 : 1)
coarse_name = Dict("rk4" => "RK4", "euler" => "explicit Euler")
method_label = Dict("coarse" => "coarse-only", "first_batch" => "first fine batch", "fine" => "serial fine",
                    "serial_half" => raw"serial RK4 ($h=0.005$)", "psi_lyap" => raw"$\psi$ ($w=e^{\lambda\Delta T}$)",
                    "psi_lip" => raw"$\psi$ (secant $w$)", "standard" => "standard", "psi" => raw"$\psi$ (either $w$)")
method_order = ("coarse", "first_batch", "fine", "serial_half", "psi_lyap", "psi_lip", "standard")  # table order
phi(rs, j) = [getproperty(r, Symbol(:phi, j)) for r in rs]
finite(r) = isfinite(r.phi1) && isfinite(r.phi2) && isfinite(r.phi3)

# Observables from pooled sufficient statistics of reference blocks (as in FunctionalRuns.jl).
function lorenz_observables(ss)
    n = sum(s.n for s in ss)
    k, sa, sb = sum(s.pairs for s in ss), sum(s.sa for s in ss), sum(s.sb for s in ss)
    va, vb = sum(s.saa for s in ss) - sa^2 / k, sum(s.sbb for s in ss) - sb^2 / k
    return [sum(s.sum1 for s in ss) / n, sum(s.energy for s in ss) / n,
            (sum(s.sab for s in ss) - sa * sb / k) / sqrt(va * vb)]
end
lorenz96_observables(ss) = (n = sum(s.n for s in ss); [sum(s.sum1 for s in ss) / n, sum(s.energy for s in ss) / n,
                                                         sum(s.neighbour for s in ss) / n])

# Bootstrap draws (one row each) of a reference, resampling pairs of consecutive blocks within each chain.
function reference_draws(observables, rows, seed)
    sorted = sort(rows; by = r -> (r.chain, r.block))
    blocks = [sorted[i:(i + 1)] for i in 1:2:length(sorted)]
    all(b[1].chain == b[2].chain for b in blocks) || error("a block pair crosses chains")
    rng, n = MersenneTwister(seed), length(blocks)
    return permutedims(stack(observables(reduce(vcat, blocks[rand(rng, 1:n, n)])) for _ in 1:draws_count))
end

reference = filter(r -> r.h == 5e-5, load("reference_lorenz"))
centre, draws = lorenz_observables(reference), reference_draws(lorenz_observables, reference, 2026101010)
error_of(rs, j) = all(finite, rs) ? mean(abs.(phi(rs, j) .- centre[j])) : NaN

# Calibration: each limit is 1.25 times the fine runs' mean error.
calibration = load("adequacy_calibration")
limits = [1.25 * mean(abs(getproperty(r, Symbol(:phi, j)) - centre[j]) for r in calibration) for j in 1:3]

# Pilot: the cheapest coarse solver whose coarse-only runs are all finite and miss the
# energy limit. Feasibility: the cheapest such solver whose forced runs all reach exact
# completion and reproduce the serial fine interfaces bitwise.
pilot, feasibility = load("adequacy_pilot"), load("adequacy_feasibility")
candidates = unique((r.method, r.h) for r in pilot)
pilot_runs(c) = [r for r in pilot if (r.method, r.h) == c]
eligible(c) = all(finite, pilot_runs(c)) && error_of(pilot_runs(c), 2) > limits[2]
screened(c) = count(r -> (r.method, r.h) == c && r.exact && r.bitwise, feasibility)
feasible(c) = eligible(c) && screened(c) == length(pilot_runs(c))
cheapest(cs) = cs[argmin([ζ(c...) for c in cs])]
first_choice, second_choice = cheapest(filter(eligible, candidates)), cheapest(filter(feasible, candidates))

# The second confirmation: every method's runs in start order, and each forced iterate.
runs, forced = load("adequacy"), load("adequacy_forced")
H = only(unique(r.h for r in runs if r.method == "coarse"))
H == second_choice[2] || error("the second confirmation does not use the screened coarse solver")
N, zeta, reps = maximum(r.s for r in forced) + 1, ζ(second_choice...), maximum(r.rep for r in runs)
history(s) = sort([r for r in forced if r.s == s]; by = r -> r.rep)
group(m) = m == "first_batch" ? history(0) : sort([r for r in runs if r.method == m]; by = r -> r.rep)
all([r.rep for r in group(m)] == 1:reps for m in method_order) || error("unpaired adequacy records")

serial(r) = r.method in ("coarse", "fine", "serial_half")
calls(s) = (s + 1) * (N - 1) - s * (s + 1) ÷ 2, (s + 1) * N - s * (s + 1) ÷ 2  # K_G and J_F after s forced corrections
function cost(r)
    r.method == "coarse" && return zeta
    serial(r) && return 1e-4 / r.h
    r.method == "forced" && return (r.K + calls(r.s)[1] * zeta) / N
    return (r.K + zeta * r.coarse_calls) / N
end
function work(r)
    r.method == "coarse" && return zeta
    serial(r) && return 1e-4 / r.h
    r.method == "forced" && return (calls(r.s)[2] + calls(r.s)[1] * zeta) / N
    return (r.fine_calls + zeta * r.coarse_calls) / N
end

# Bootstrap errors on the same resampled starts for every group.
rng = MersenneTwister(2026101011)
idx = rand(rng, 1:reps, draws_count, reps)
function boot(rs, j)
    all(finite, rs) || return fill(NaN, draws_count)
    values = phi(rs, j)
    return [mean(abs.(values[idx[k, :]] .- draws[k, j])) for k in 1:draws_count]
end
boots = Dict((m, j) => boot(group(m), j) for m in method_order for j in 1:3)
lower(m, j) = quantile(boots[(m, j)], 0.05)
upper(m, j) = quantile(boots[(m, j)], 0.95)
paired(m, j, base = "coarse") = quantile(boots[(m, j)] - boots[(base, j)], [0.025, 0.975])
mean_cost(m) = mean(cost(r) for r in group(m))
ratio(m) = mean_cost(m) / mean_cost("standard")
function ratio_upper(m)
    a, b = cost.(group(m)), cost.(group("standard"))
    return quantile([mean(a[i] for i in idx[k, :]) / mean(b[i] for i in idx[k, :]) for k in 1:draws_count], 0.95)
end

# The Lorenz-96 confirmation, energy only (as in ConfirmationTables.jl).
reference96 = filter(r -> r.h == 5e-5, load("reference_lorenz96"))
centre96, draws96 = lorenz96_observables(reference96), reference_draws(lorenz96_observables, reference96, 584)
confirmation = load("confirmation")
N96, H96 = 128, only(unique(r.h for r in confirmation if r.method == "coarse"))
runs96(m) = sort([r for r in confirmation if r.method == m]; by = r -> r.rep)
reps96 = length(runs96("coarse"))
rng96 = MersenneTwister(912610)
indices96 = [rand(rng96, 1:reps96, reps96) for _ in 1:draws_count]
error96(m) = mean(abs.(phi(runs96(m), 2) .- centre96[2]))
boot96(m) = (values = phi(runs96(m), 2); [mean(abs.(values[ii] .- draws96[k, 2])) for (k, ii) in enumerate(indices96)])
cost96(m) = mean(startswith(m, "psi") ? (r.K + 0.01 * r.coarse_calls) / N96 : 1e-4 / r.h for r in runs96(m))
outcome96(r) = (r.K, r.fine_calls, r.coarse_calls, r.phi1, r.phi2, r.phi3)
same96 = all(isequal(outcome96(a), outcome96(b)) for (a, b) in zip(runs96("psi_lyap"), runs96("psi_lip")))

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
three(x) = isfinite(x) ? @sprintf("\$%.3f\$", x) : "--"
# An interval, each end to three significant figures in plain decimals.
bracket((lo, hi)) = isfinite(lo) && isfinite(hi) ? "[$(decimal(lo)), $(decimal(hi))]" : "--"
function batches(m)
    m in ("coarse", "fine", "serial_half") && return ""
    low, high = extrema(r.K for r in group(m))
    return low == high ? "\$$low\$" : "\$$low\$--\$$high\$"
end
setting(x) = @sprintf("%g", x)

tables = joinpath(@__DIR__, "Tables")
mkpath(tables)
row(cells) = join(cells, " & ")

# tab:functional_main: the article's float; its caption is in the paper.
limit = @sprintf("%.3f", limits[2])
order96 = same96 ? ("coarse", "serial_half", "psi_lyap") : ("coarse", "serial_half", "psi_lyap", "psi_lip")
costs = column([[mean_cost(m) for m in method_order]; [cost96(m) for m in order96]])  # one column, both blocks
lorenz63_rows = map(enumerate(method_order)) do (i, m)
    row([method_label[m], three(error_of(group(m), 2)), three(upper(m, 2)),
         startswith(m, "psi") ? "[$(three(paired(m, 2)[1])), $(three(paired(m, 2)[2]))]" : "", costs[i]])
end
lorenz96_rows = map(enumerate(order96)) do (i, m)
    interval = startswith(m, "psi") ? bracket(quantile(boot96(m) - boot96("coarse"), [0.025, 0.975])) : ""
    row([method_label[same96 && m == "psi_lyap" ? "psi" : m], num(error96(m)), "", interval,
         costs[length(method_order) + i]])
end
heading(text) = "\\multicolumn{5}{c}{\\textit{$text}}\\\\*\n"
write(joinpath(tables, "functional_main.tex"),
      "% Generated by AdequacyTables.jl.\n" *
      "\\setlength{\\tabcolsep}{10pt}\\renewcommand{\\arraystretch}{1.15}\n" *
      "\\begin{tabular}{@{}ccccc@{}}\\toprule\n" *
      raw"Method & $E$ & $U_{0.95}$ & Paired 95\% interval & $\mathcal{C}$" * "\\\\\n\\midrule\n" *
      heading("Lorenz-63: \$T = N = $N\$, \$H = $(setting(H))\$, \$\\zeta = $(setting(zeta))\$, limit $limit") *
      join(lorenz63_rows, "\\\\*\n") * "\\\\\n\\midrule\n" *
      heading("Lorenz-96: \$T = N = $N96\$, \$H = $(setting(H96))\$, \$\\zeta = 0.01\$") *
      join(lorenz96_rows, "\\\\*\n") * "\\\\\n\\bottomrule\\end{tabular}\n")

# tab:stopping_adequacy: a supplement table at the text width in three panels, each
# starting only where its heading fits, with a title above its rules, a repeated
# heading and no rule in the page foot.
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
all_finite = all(finite, runs) && all(finite, forced)
caption = "Independent Lorenz-63 confirmation after the feasibility screen: \$T=N=$N\$, " *
          "$(coarse_name[second_choice[1]]) coarse \$H=$(setting(H))\$, $reps fresh starts" *
          (all_finite ? ", all runs finite. " : ". ") *
          raw"Errors are means of absolute functional-estimation errors against the long reference. $L_{0.95},U_{0.95}$ are one-sided 95\% bootstrap bounds. The first fine batch is the forced run's first iterate: fine trajectories from coarse-pass starts, without correction. Panel (b) reports paired energy-error differences against coarse-only and ratios of mean elapsed costs against standard stopping, with a two-sided 95\% interval and one-sided upper bound, respectively. Secondary limits appear in panel (c)."
primary_costs = column([mean_cost(m) for m in method_order])
primary_work = column([mean(work(r) for r in group(m)) for m in method_order])
primary = map(enumerate(method_order)) do (i, m)
    row([method_label[m], three(error_of(group(m), 2)), three(lower(m, 2)), three(upper(m, 2)), batches(m),
         primary_costs[i], primary_work[i]])
end
comparisons = map(("psi_lyap", "psi_lip")) do m
    lo, hi = paired(m, 2)
    row([method_label[m], three(error_of(group(m), 2) - error_of(group("coarse"), 2)), "[$(three(lo)), $(three(hi))]",
         three(ratio(m)), three(ratio_upper(m))])
end
secondary_columns = [column([value(m) for m in method_order])
                     for value in (m -> error_of(group(m), 1), m -> upper(m, 1), m -> error_of(group(m), 3), m -> upper(m, 3))]
secondary = [row([method_label[m]; [c[i] for c in secondary_columns]]) for (i, m) in enumerate(method_order)]
write(joinpath(tables, "stopping_adequacy.tex"),
      "% Generated by AdequacyTables.jl.\n" *
      "\\begingroup\\small\\setlength{\\tabcolsep}{1.5pt}\\renewcommand{\\arraystretch}{1.15}" *
      "\\setlength{\\LTleft}{0pt}\\setlength{\\LTright}{0pt}\n" *
      panel(7, "(a) Primary energy error (limit $limit)",
            raw" & \multicolumn{3}{c}{Energy error} & \multicolumn{3}{c}{Work and cost}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}" *
            "\n" * raw"Method & Estimate & $L_{0.95}$ & $U_{0.95}$ & $K_F$ & $\mathcal{C}$ & $\mathcal{Q}$" * "\\\\\n",
            join(primary, "\\\\*\n") * "\\\\\n";
            caption = "\\caption{\\rev{$caption}}\\label{tab:stopping_adequacy}\\\\\n") *
      panel(5, "(b) Paired energy and elapsed-cost comparisons",
            raw" & \multicolumn{2}{c}{Energy: accepted minus coarse-only} & \multicolumn{2}{c}{Cost: accepted / standard}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}" *
            "\n" * raw"Rule & Difference & 95\% interval & Ratio & $U_{0.95}$" * "\\\\\n",
            join(comparisons, "\\\\*\n") * "\\\\\n") *
      panel(5, "(c) Secondary observables",
            " & \\multicolumn{2}{c}{\$\\langle z\\rangle\$ (limit $(num(limits[1])))} & " *
            "\\multicolumn{2}{c}{\$\\rho_x(0.5)\$ (limit $(num(limits[3])))}\\\\" *
            raw"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" * "\n" *
            raw"Method & Estimate & $U_{0.95}$ & Estimate & $U_{0.95}$" * "\\\\\n",
            join(secondary, "\\\\*\n") * "\\\\\n") *
      "\\endgroup\n")

# The numbers the text quotes.
@printf("Limits (mean z, energy, autocorrelation): %.6g  %.6g  %.6g\n", limits[1], limits[2], limits[3])
println("\nPilot (energy error, ζ, finite runs, eligible):")
for c in candidates
    @printf("  %-5s H = %-6s  %8.4g  %.4g  %d/%d  %s\n", c[1], c[2], error_of(pilot_runs(c), 2), ζ(c...),
            count(finite, pilot_runs(c)), length(pilot_runs(c)), eligible(c))
end
println("Pilot choice: ", first_choice)
if isfile(joinpath(@__DIR__, "Records", "adequacy_first.jls"))
    first_runs = load("adequacy_first")
    first_choice[2] == only(unique(r.h for r in first_runs if r.method == "coarse")) ||
        error("the first confirmation does not use the pilot's choice")
    @printf("First confirmation: coarse-only energy error %.5g\n",
            error_of(sort([r for r in first_runs if r.method == "coarse"]; by = r -> r.rep), 2))
    for m in ("psi_lyap", "psi_lip", "standard")
        rs = [r for r in first_runs if r.method == m]
        println("  ", m, ": ", count(!finite, rs), " of ", length(rs), " nonfinite")
    end
else
    println("First confirmation: Records/adequacy_first.jls is missing")
end
println("\nFeasibility (exact and bitwise of 8, eligible in the pilot):")
for c in candidates
    println("  ", rpad(c[1], 5), " H = ", rpad(c[2], 6), "  ", screened(c), "/", length(pilot_runs(c)), "  ", eligible(c))
end
println("Screened choice: ", second_choice)

println("\nSecond confirmation, all runs finite: ", all_finite)
println("Fine batches: ", join(("$m $(batches(m))" for m in ("psi_lyap", "psi_lip", "standard")), ", "))
println("Energy error, L, U, C, Q:")
for m in method_order
    @printf("  %-12s  %.3f  %.3f  %.3f   C = %.3g  Q = %.3g\n", m, error_of(group(m), 2), lower(m, 2), upper(m, 2),
            mean_cost(m), mean(work(r) for r in group(m)))
end
for m in ("psi_lyap", "psi_lip")
    lo, hi = paired(m, 2)
    @printf("%s minus coarse-only, energy: %+.3f  [%.3f, %.3f];  cost ratio to standard %.3f, U %.3f\n", m,
            error_of(group(m), 2) - error_of(group("coarse"), 2), lo, hi, ratio(m), ratio_upper(m))
    lo, hi = paired(m, 2, "first_batch")
    @printf("  minus first fine batch, energy: %+.3f  [%.3f, %.3f]\n",
            error_of(group(m), 2) - error_of(group("first_batch"), 2), lo, hi)
    for j in (1, 3)
        lo, hi = paired(m, j)
        @printf("  observable %d: U %.4g against limit %.4g; difference interval [%.3g, %.3g]\n", j, upper(m, j),
                limits[j], lo, hi)
    end
end
uppers = [quantile(boot(history(s), 2), 0.95) for s in 0:(N - 1)]
@printf("Smallest forced energy upper bound: %.3f (limit %.3f)\n", minimum(uppers), limits[2])
accepted_runs = [r for r in runs if startswith(r.method, "psi") || r.method == "standard"]
iterate_of(r) = only(x for x in forced if x.rep == r.rep && x.s == r.s)
mismatch = maximum(abs(getproperty(r, Symbol(:phi, j)) - getproperty(iterate_of(r), Symbol(:phi, j)))
                   for r in accepted_runs for j in 1:3)
@printf("Largest difference between an accepted estimate and its forced iterate: %.2g\n", mismatch)
# Where each start's forced energy error first meets the limit, and whether it leaves again.
inside(r) = abs(r.phi2 - centre[2]) <= limits[2]
crossings = map(1:reps) do rep
    hs = sort([r for r in forced if r.rep == rep]; by = r -> r.s)
    first_in = findfirst(inside, hs)
    (first_in === nothing ? nothing : hs[first_in].s,
     first_in !== nothing && any(!inside(r) for r in hs[(first_in + 1):end]))
end
println("Starts whose forced energy error meets the limit: ", count(c -> c[1] !== nothing, crossings), " of ", reps,
        "; of these, leaving it again later: ", count(c -> c[2], crossings))
for m in ("psi_lyap", "psi_lip", "standard")
    s = [r.s for r in group(m)]
    println("  ", m, " accepts before / at / after the first crossing: ",
            count(i -> crossings[i][1] !== nothing && s[i] < crossings[i][1], 1:reps), " / ",
            count(i -> crossings[i][1] !== nothing && s[i] == crossings[i][1], 1:reps), " / ",
            count(i -> crossings[i][1] !== nothing && s[i] > crossings[i][1], 1:reps))
end

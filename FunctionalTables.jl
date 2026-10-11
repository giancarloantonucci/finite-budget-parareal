# FunctionalTables.jl
#
# The long-reference table and the exploratory numbers quoted in the supplement
# (sec:functional_accuracy), from the records of FunctionalRuns.jl.
#   Supplement: tab:functional_reference.
#   Printed:    whether every step-halving interval contains zero; the change in
#               reference interval width from 256- to 128-time-unit blocks; the
#               exploratory errors with RK4 coarse chunks and their paired intervals
#               against coarse-only; the failed runs with Euler coarse chunks.
# Intervals are bootstrap 95% intervals with 4000 draws. Reference intervals resample
# pairs of consecutive stored blocks (256 time units) within the chains; the step-halving
# intervals draw the same blocks at both step sizes, so they are paired. Exploratory
# intervals resample whole starts and, independently, the h = 5e-5 reference blocks.
#
# Needs Printf, Random, Serialization and Statistics.
# Run:     julia FunctionalTables.jl
# Writes:  Tables/functional_reference.tex

using Printf, Random, Serialization, Statistics

records(kind, name) = deserialize(joinpath(@__DIR__, "Records", "$(kind)_$(name).jls"))  # only load records you made yourself

problems, draws_count = ("lorenz", "lorenz96"), 4000
problem_label = Dict("lorenz" => "Lorenz-63", "lorenz96" => "Lorenz-96")
observable_label = Dict("lorenz" => (raw"$\langle z\rangle$", raw"$\langle \|u\|^2/3\rangle$", raw"$\rho_x(0.5)$"),
                        "lorenz96" => (raw"$\langle \bar{x}\rangle$", raw"$\langle \|u\|^2/40\rangle$",
                                       raw"$\langle x_i x_{i-1}\rangle$"))
observable_name = Dict("lorenz" => ("mean z", "energy", "autocorrelation"),
                       "lorenz96" => ("mean", "energy", "neighbour product"))
rule_name = Dict("serial_rk4" => "coarse-only", "psi_lyap" => "psi (Lyapunov w)", "psi_lip" => "psi (secant w)")

# The three observables from pooled sufficient statistics (as in FunctionalRuns.jl).
function observables(name, ss)
    n = sum(s.n for s in ss)
    mean1, energy = sum(s.sum1 for s in ss) / n, sum(s.energy for s in ss) / n
    name == "lorenz96" && return [mean1, energy, sum(s.neighbour for s in ss) / n]
    k, sa, sb = sum(s.pairs for s in ss), sum(s.sa for s in ss), sum(s.sb for s in ss)
    va, vb = sum(s.saa for s in ss) - sa^2 / k, sum(s.sbb for s in ss) - sb^2 / k
    return [mean1, energy, (sum(s.sab for s in ss) - sa * sb / k) / sqrt(va * vb)]
end

# Bootstrap draws (one row each) of the observables, resampling runs of `group`
# consecutive stored blocks within each chain.
function reference_draws(name, rows; group = 2, seed = 174)
    sorted = sort(rows; by = r -> (r.chain, r.block))
    blocks = [sorted[i:(i + group - 1)] for i in 1:group:length(sorted)]
    all(length(unique(r.chain for r in b)) == 1 for b in blocks) || error("a block group crosses chains")
    rng, n = MersenneTwister(seed), length(blocks)
    return permutedims(stack(observables(name, reduce(vcat, blocks[rand(rng, 1:n, n)])) for _ in 1:draws_count))
end

ci(x) = quantile(x, [0.025, 0.975])
reference = Dict(map(problems) do name
    rows = records("reference", name)
    coarse, fine = filter(r -> r.h == 1e-4, rows), filter(r -> r.h == 5e-5, rows)
    name => (; coarse, fine, centre = observables(name, fine), dc = reference_draws(name, coarse),
             df = reference_draws(name, fine), df1 = reference_draws(name, fine; group = 1))
end)

tables = joinpath(@__DIR__, "Tables")
mkpath(tables)

# One row per observable: the reference (h = 5e-5) and the step-halving difference
# (h = 1e-4 minus 5e-5), each with its interval. Each observable gets the decimals that
# show its reference interval's half-width to two significant figures; three significant
# figures would hide the intervals.
function reference_rows(name)
    R = reference[name]
    rows = map(1:3) do j
        reference_ci, halving_ci = ci(R.df[:, j]), ci(R.dc[:, j] - R.df[:, j])
        decimals = 1 - floor(Int, log10(diff(reference_ci)[1] / 2))
        fixed(x) = "\$" * Printf.format(Printf.Format("%.$(decimals)f"), x) * "\$"
        bracket((lo, hi)) = "[$(fixed(lo)), $(fixed(hi))]"
        halving = observables(name, R.coarse)[j] - R.centre[j]
        join([observable_label[name][j], fixed(R.centre[j]), bracket(reference_ci), fixed(halving), bracket(halving_ci)],
             " & ")
    end
    return join(rows, "\\\\*\n") * "\\\\\n"
end

# A supplement table at its own width that may run over pages; the heading repeats and
# the page foot carries no rule of its own.
caption = raw"Long-reference functional estimates. Each step size uses four independently prepared trajectories of duration 8192 (Lorenz-63) or 2048 (Lorenz-96) after spin-up. The reference is the estimate at $h=5\times10^{-5}$; the halving difference is the estimate at $h=10^{-4}$ minus the reference. Intervals are 95\% block-bootstrap intervals with 256-time-unit blocks; the halving intervals resample the same blocks at both step sizes."
head = raw" & \multicolumn{2}{c}{Reference} & \multicolumn{2}{c}{Step halving}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}" *
       "\n" * raw"Observable & Estimate & 95\% interval & Difference & 95\% interval" * "\\\\\n"
body = join(("\\multicolumn{5}{c}{\\textit{$(problem_label[name])}}\\\\*\n" * reference_rows(name) for name in problems),
            "\\midrule\n")
write(joinpath(tables, "functional_reference.tex"),
      "% Generated by FunctionalTables.jl.\n" *
      "\\Needspace{20\\baselineskip}\\begingroup\\small\\setlength{\\tabcolsep}{10pt}\\renewcommand{\\arraystretch}{1.15}\n" *
      "\\begin{longtable}{@{}ccccc@{}}\n" *
      "\\caption{\\rev{$caption}}\\label{tab:functional_reference}\\\\\n" *
      "\\toprule\n" * head * "\\midrule\\endfirsthead\n" *
      "\\multicolumn{5}{l}{\\tablename~\\thetable\\ (continued)}\\\\\n" *
      "\\toprule\n" * head * "\\midrule\\endhead\n" *
      "\\multicolumn{5}{r}{Continued on next page}\\\\\\endfoot\n" *
      "\\bottomrule\\endlastfoot\n" * body * "\\end{longtable}\\endgroup\n")

halving_cis = [ci(reference[name].dc[:, j] - reference[name].df[:, j]) for name in problems for j in 1:3]
println("Every step-halving interval contains zero: ", all(lo <= 0 <= hi for (lo, hi) in halving_cis))
println("Interval-width ratio at h = 5e-5, 256- to 128-time-unit blocks:")
for name in problems, j in 1:3
    R = reference[name]
    @printf("  %-9s  %-17s  %.3f\n", problem_label[name], observable_name[name][j],
            diff(ci(R.df[:, j]))[1] / diff(ci(R.df1[:, j]))[1])
end

# Mean absolute errors against the h = 5e-5 reference for one method, and the paired
# interval for its difference from coarse-only; nothing if any start failed.
function exploratory(rows, name, N, method, j)
    pick(m) = sort([r for r in rows if r.N == N && r.method == m && r.coarse in ("rk4", "none") && r.h == 0.01];
                   by = r -> r.rep)
    runs, baseline_runs = pick(method), pick("serial_rk4")
    [r.rep for r in runs] == [r.rep for r in baseline_runs] || error("unpaired exploratory records")
    all(r -> isfinite(r.phi1) && isfinite(r.phi2) && isfinite(r.phi3), runs) || return nothing
    phi(rs) = [getproperty(r, Symbol(:phi, j)) for r in rs]
    vals, baseline = phi(runs), phi(baseline_runs)
    centre, df = reference[name].centre[j], reference[name].df[:, j]
    rng, n = MersenneTwister(831 + j + N), length(vals)
    differences = map(1:draws_count) do k
        ii = rand(rng, 1:n, n)
        mean(abs.(vals[ii] .- df[k])) - mean(abs.(baseline[ii] .- df[k]))
    end
    return mean(abs.(vals .- centre)), ci(differences)
end

println("\nExploratory errors with RK4 coarse chunks; intervals are for the method minus coarse-only:")
for name in problems
    rows = records("functionals", name)
    for N in (32, 128), method in ("serial_rk4", "psi_lyap", "psi_lip"), j in 1:3
        @printf("  %-9s  N = %-3d  %-16s  %-17s", problem_label[name], N, rule_name[method], observable_name[name][j])
        result = exploratory(rows, name, N, method, j)
        if result === nothing
            println("  failed")
        elseif method == "serial_rk4"
            @printf("  %.3g\n", result[1])
        else
            mae, (lo, hi) = result
            @printf("  %.3g  [%.3g, %.3g]%s\n", mae, lo, hi, hi < 0 || lo > 0 ? "  excludes 0" : "")
        end
    end
end

println("\nFailed runs with explicit Euler coarse chunks (nonfinite of 8):")
for name in problems
    rows = records("functionals", name)
    for N in (32, 128), method in ("psi_lyap", "psi_lip")
        rs = [r for r in rows if r.N == N && r.method == method && r.coarse == "euler"]
        @printf("  %-9s  N = %-3d  %-16s  %d of %d\n", problem_label[name], N, rule_name[method],
                count(r -> !isfinite(r.phi1), rs), length(rs))
    end
end

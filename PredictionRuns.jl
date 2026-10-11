# PredictionRuns.jl
#
# Sweep-budget predictions and the implementation checks of sec:prediction and the
# supplement (sec:implementation_checks, sec:frozen_prediction, sec:coarse_prediction,
# sec:numerical_checks, sec:periodic_table). PredictionTables.jl turns the records into
# tab:prediction and the supplement's prediction, refinement and roundoff tables, and
# prints the linear checks and periodic diagnostics the text quotes;
# PredictionFigures.jl draws fig:measured_vs_bound.
#
# A prediction samples the actual chunk maps' derivatives at 16 of the coarse-pass
# starts and inflates their maxima by 1.25, giving the transports Λ_F, Λ_G, the source S
# and the fixed weight w = max(1, Λ_F, Λ_G). It transports all first-batch defects into
# the initial-error estimate ê₀ and predicts the sweeps needed to reach the target r.
# A run forced past acceptance gives the observed error of every iterate in the
# stored-start norm with that weight. The records are:
#   prediction           logistic (implicit Euler, h = 1e-3) and Lorenz-63 and its
#                        periodic control (RK4, h = 1e-4) at N = 32, ΔT = 1, ξ = 10, 100;
#   prediction_short     Lorenz-63 short windows, N = 32, ξ = 10 (L = 20) and 100 (L = 2),
#                        h = 1e-4 and 5e-5, with the coarse-node RK4 stage diagnostic;
#   prediction_coarse    Lorenz-63 at N = 16, 128, H = 0.01, 0.02, RK4 or explicit Euler
#                        coarse chunks, with every iterate also in the norm w = e^{0.9056};
#   prediction_linear    u' = au, a = ±1, N = 8, 32, from closed-form chunk factors;
#   refinement           the derivative mismatch of fine and coarse unit chunks at
#                        ξ = 5, …, 100 on eight serial fine states;
#   roundoff             exact completion at N = 32, 64, natively and with the
#                        unparenthesized update applied to every interface;
#   periodic_diagnostics the periodic orbit's period, return error, Floquet multipliers
#                        and sampled amplification over 1, …, 32 unit chunks.
#
# Needs ForwardDiff, LinearAlgebra, Serialization, NSDERungeKutta and NSDETimeParallel.
# Run:     julia --threads=auto PredictionRuns.jl
#          Records that already exist are kept; only the missing ones are computed.
# Writes:  Records/prediction.jls, prediction_short.jls, prediction_coarse.jls,
#          prediction_linear.jls, refinement.jls, roundoff.jls, periodic_diagnostics.jls

using ForwardDiff, LinearAlgebra, Serialization
using NSDERungeKutta, NSDETimeParallel

start = Dict("logistic" => [0.1], "lorenz" => [2.0, 3.0, -14.0],
             "periodic" => [0.0, 3.6839592143970576, 73.30237399900534])
base = Dict("logistic" => Logistic(start["logistic"], (0.0, 1.0)), "lorenz" => Lorenz(start["lorenz"], (0.0, 1.0)),
            "periodic" => Lorenz(start["periodic"], (0.0, 1.0); ρ = 100.5))
whole(name, u0, T) = remake(base[name]; u0 = copy(u0), tspan = (0.0, T))
method(name, h) = name == "logistic" ? BackwardEuler(h = h, εᵣ = 1e-13, εₐ = 1e-15, Mₙ = 30) : RK4(h = h)
inflation = 1.25

# One chunk of length ΔT from u, and the map u ↦ its end state.
chunk(name, solver, ΔT, u) = solve(remake(base[name]; u0 = u, tspan = (0.0, ΔT)), chunk_solver(solver, ΔT))
flow(name, solver, ΔT, u) = chunk(name, solver, ΔT, u).u[end]

# The chunk map's Jacobian. For the logistic implicit Euler map it is the product of
# the exact step derivatives 1/(1 - h f′(v)) along the computed steps: differentiating
# the Newton iteration is unreliable at equilibria.
function jacobian(name, solver, h, ΔT, u)
    name == "logistic" || return ForwardDiff.jacobian(v -> flow(name, solver, ΔT, v), u)
    j = 1.0
    for v in chunk(name, solver, ΔT, u).u[2:end]
        j /= 1 - h * (1 - 2v[1])
    end
    return fill(j, 1, 1)
end

field_jacobian(name, u) = ForwardDiff.jacobian(v -> (dv = similar(v); base[name].rhs(dv, v, 0.0); dv), u)

# The serial interfaces U₀, …, U_N from u0.
function serial(name, solver, ΔT, u0, N)
    U = [u0]
    for _ in 1:N
        push!(U, flow(name, solver, ΔT, U[end]))
    end
    return U
end

sample_indices(N, k = 16) = unique(round.(Int, range(1, N; length = min(N, k))))
transport(x, N) = N == 1 ? zero(x) : sum(x^i for i in 0:(N - 2))  # Σ_{i ≤ N-2} x^i

# The smallest s with β^s ê₀ ≤ r; nothing when β ≥ 1 gives no contraction.
function budget(β, e0, r)
    e0 <= r && return 0
    β == 0 && return 1
    β >= 1 && return nothing
    return max(0, ceil(Int, log(e0 / r) / -log(β)))
end

# ‖(w^{-n}‖U_n - U★_n‖)_{n=0}^{N-1}‖₂: the stored starts, the terminal value excluded.
start_error(U, U★, w) = norm([exp(-n * log(w)) * norm(U[n + 1] - U★[n + 1]) for n in 0:(length(U) - 2)])

# The initial-error estimate: each weighted defect transported by Λ_F/w, then the ℓ² norm.
function transported(defects, λF, w)
    q, a = 0.0, Float64[]
    for (n, d) in enumerate(defects)
        q = (λF / w) * q + exp(-n * log(w)) * d
        push!(a, q)
    end
    return norm(a)
end

# The frozen-Jacobian model's leading source term; b = bᵀc is ½ for RK4 and 1 for implicit Euler.
frozen_source(h, ξ, L, μ, ν, b) = L * ξ * h^2 * (μ^2 * abs(b - ξ * b + (ξ - 1) / 2) + (ξ - 1) * ν / 2)

# Every iterate's starts in a run forced to exact completion or to K fine batches, and
# the serial fine interfaces it is measured against.
function forced(name, fine, coarse, ΔT, N, K)
    parareal = Parareal(fine, coarse; N, K, criterion = NoStop())
    sol = solve(whole(name, start[name], N * ΔT), parareal; history = :interfaces)
    return sol, reference(whole(name, start[name], N * ΔT), parareal)
end

# Sampled bounds from the coarse pass U0: the vector field's μ and drift ν over every
# coarse node, and the inflated chunk transports and source at the sampled starts.
function sampled(name, fine, coarse, h, H, ΔT, U0, N)
    nodes = [[U0[1]]; [v for n in 1:N for v in chunk(name, coarse, ΔT, U0[n]).u[2:end]]]
    J = [field_jacobian(name, v) for v in nodes]
    μ, ν = maximum(opnorm, J), maximum(opnorm(J[i + 1] - J[i]) / H for i in 1:(length(J) - 1))
    samples = sample_indices(N)
    JF = [jacobian(name, fine, h, ΔT, U0[n]) for n in samples]
    JG = [jacobian(name, coarse, H, ΔT, U0[n]) for n in samples]
    return (; μ, ν, samples, λF = inflation * maximum(opnorm, JF), λG = inflation * maximum(opnorm, JG),
            S = inflation * maximum(opnorm(a - b) for (a, b) in zip(JF, JG)))
end

# First-batch defects d_n = ‖U0_{n+1} - F(U0_n)‖; the terminal one is zero by reconstruction.
function first_defects(name, fine, ΔT, U0, N)
    defects = [norm(U0[n + 1] - flow(name, fine, ΔT, U0[n])) for n in 1:N]
    defects[end] = 0.0
    return defects
end

function prediction(name, ξ)
    N, ΔT, h = 32, 1.0, name == "logistic" ? 1e-3 : 1e-4
    H, r = ξ * h, name == "logistic" ? 1e-12 : 1e-9
    fine, coarse = method(name, h), method(name, H)
    U0 = serial(name, coarse, ΔT, start[name], N)
    s = sampled(name, fine, coarse, h, H, ΔT, U0, N)
    w = max(1.0, s.λF, s.λG)
    β = transport(s.λG / w, N) * s.S / w
    defects = first_defects(name, fine, ΔT, U0, N)
    ehat = transported(defects, s.λF, w)
    ψ0 = sum(exp(-n * log(w)) * defects[n] for n in 1:N) / N
    CN = N * norm([(s.λF / w)^j for j in 0:(N - 1)])
    kpred = budget(β, ehat, r)
    sweeps = kpred === nothing ? min(N, 8) : min(N, max(4, kpred + 2))  # enough to see the crossing
    sol, U★ = forced(name, fine, coarse, ΔT, N, min(N, sweeps + 1))
    return (; problem = name, xi = ξ, N, h, dt = ΔT, r, mu = s.μ, nu = s.ν, lambdaF = s.λF, lambdaG = s.λG,
            source = s.S, w, beta = β, ehat, eactual = start_error(U0, U★, w), psi0 = ψ0, CN, elook = CN * ψ0,
            kpred, derivative_samples = length(s.samples), defects,
            errors = [start_error(U, U★, w) for U in sol.diagnostics.interfaces])
end

# Halving h and H at fixed L, ξ, N shortens the chunks; eight forced sweeps show the crossing.
P4(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24  # the RK4 stability polynomial
function short_window(ξ, h)
    N, L, H, r = 32, ξ == 10 ? 20 : 2, ξ * h, 1e-9
    ΔT = L * H
    fine, coarse = RK4(h = h), RK4(h = H)
    U0 = serial("lorenz", coarse, ΔT, start["lorenz"], N)
    s = sampled("lorenz", fine, coarse, h, H, ΔT, U0, N)
    w = max(1.0, s.λF, s.λG)
    β = s.S / w * transport(s.λG / w, N)
    defects = first_defects("lorenz", fine, ΔT, U0, N)
    sol, U★ = forced("lorenz", fine, coarse, ΔT, N, 9)
    history = sol.diagnostics.interfaces
    # The coarse-node stage diagnostic: RK4 stage majorants from μ̂ and the frozen source.
    λF, λG = P4(h * s.μ)^(L * ξ), P4(H * s.μ)^L
    wRK = max(1.0, λF, λG)
    return (; problem = "lorenz", xi = ξ, N, h, H, L, dt = ΔT, r, mu = s.μ, nu = s.ν, w, source = s.S,
            lambdaF = s.λF, lambdaG = s.λG, beta = β, ehat = transported(defects, s.λF, w),
            derivative_samples = length(s.samples), defects,
            errors = [start_error(U, U★, w) for U in history],
            stage_lambdaF = λF, stage_lambdaG = λG, stage_w = wRK,
            stage_beta = frozen_source(h, ξ, L, s.μ, s.ν, 0.5) / wRK * transport(λG / wRK, N),
            stage_ehat = transported(defects, λF, wRK), stage_errors = [start_error(U, U★, wRK) for U in history])
end

# The prescribed Lorenz-63 start with matched RK4 and Euler coarse chunks, forced to exact
# completion or failure. Every iterate is also measured in one common norm, w = e^{0.9056}.
function coarse_start(N, H, coarse_method)
    h, ΔT, r = 1e-4, 1.0, 1e-9
    fine, coarse = RK4(h = h), coarse_method == "rk4" ? RK4(h = H) : Euler(h = H)
    U0 = serial("lorenz", coarse, ΔT, start["lorenz"], N)
    finite(U) = all(u -> all(isfinite, u), U)
    record = (; problem = "lorenz", method = coarse_method, N, H, h, xi = round(Int, H / h), L = round(Int, ΔT / H),
              dt = ΔT, r, derivative_samples = length(sample_indices(N)), lyap_w = exp(0.9056 * ΔT))
    finite(U0) || return (; record..., diagnostic = "nonfinite_initial", w = nothing, beta = nothing, ehat = nothing,
                           lambdaF = nothing, lambdaG = nothing, source = nothing, defects = Float64[],
                           errors = Float64[], lyap_errors = Float64[], status = "nonfinite")
    defects = first_defects("lorenz", fine, ΔT, U0, N)
    samples = sample_indices(N)
    JF = [jacobian("lorenz", fine, h, ΔT, U0[n]) for n in samples]
    JG = [jacobian("lorenz", coarse, H, ΔT, U0[n]) for n in samples]
    λF, λG, S = inflation * maximum(opnorm, JF), inflation * maximum(opnorm, JG),
                inflation * maximum(opnorm(a - b) for (a, b) in zip(JF, JG))
    w = max(1.0, λF, λG)
    sol, U★ = forced("lorenz", fine, coarse, ΔT, N, N)
    history = sol.diagnostics.interfaces
    return (; record..., diagnostic = "sampled", w, beta = S / w * transport(λG / w, N),
            ehat = transported(defects, λF, w), lambdaF = λF, lambdaG = λG, source = S, defects,
            errors = [start_error(U, U★, w) for U in history],
            lyap_errors = [start_error(U, U★, record.lyap_w) for U in history],
            status = string(sol.diagnostics.status))
end

# u' = au with RK4: the chunk factors are closed-form powers of the stability polynomial.
function linear(a, N)
    h, ξ, L, r = 0.01, 10, 1, 1e-10
    ff, gg = P4(a * h)^(ξ * L), P4(a * ξ * h)^L
    w = max(1.0, abs(ff), abs(gg))
    β = transport(abs(gg) / w, N) * abs(ff - gg) / w
    e0 = norm([w^(-n) * (gg^n - ff^n) for n in 0:(N - 1)])
    problem = IVP((du, u, t) -> (du .= a .* u), [1.0], (0.0, N * ξ * L * h))
    parareal = Parareal(RK4(h = h), RK4(h = ξ * h); N, K = min(N, min(N - 1, 4) + 1), criterion = NoStop())
    sol, U★ = solve(problem, parareal; history = :interfaces), reference(problem, parareal)
    hit = findfirst(U -> start_error(U, U★, w) <= r, sol.diagnostics.interfaces)
    return (; a, N, beta = β, kpred = budget(β, e0, r), kobs = hit === nothing ? nothing : hit - 1)
end

# The largest ‖DF - DG‖ over eight serial fine states, for coarse steps ξh.
function refinement(name, h)
    states = serial(name, method(name, h), 1.0, start[name], 8)[1:(end - 1)]
    return map([5, 10, 20, 50, 100]) do ξ  # a vector, so the two problems' rows concatenate
        mismatch = maximum(opnorm(jacobian(name, method(name, h), h, 1.0, u) -
                                  jacobian(name, method(name, ξ * h), ξ * h, 1.0, u)) for u in states)
        (; problem = name, xi = ξ, H = ξ * h, source = mismatch)
    end
end

# Exact completion with native Parareal, and with the update U_{n+1} = G(U_n) + F_n - G_n
# applied to every interface in that order instead (a negative control).
function roundoff(N)
    fine, coarse = RK4(h = 1e-4), RK4(h = 0.01)
    parareal = Parareal(fine, coarse; N, criterion = NoStop())
    U★ = reference(whole("lorenz", start["lorenz"], N), parareal)
    native = boundaryvalues(solve(whole("lorenz", start["lorenz"], N), parareal).lastiterate)
    U = serial("lorenz", coarse, 1.0, start["lorenz"], N)
    G = U[2:end]
    for _ in 1:N
        Fs = [flow("lorenz", fine, 1.0, u) for u in U[1:(end - 1)]]
        V, Gnew = copy(U), similar(G)
        for n in 1:N
            Gnew[n] = flow("lorenz", coarse, 1.0, V[n])
            V[n + 1] = Gnew[n] + Fs[n] - G[n]
        end
        U, G = V, Gnew
    end
    distance(U) = sqrt(sum(sum(abs2, U[n] - U★[n]) for n in eachindex(U★)))
    return [(; N, naive = false, error = distance(native)), (; N, naive = true, error = distance(U))]
end

# The periodic orbit: its period from a transverse return to x = 0 refined by bisection,
# the fixed-period map's Jacobian, and the largest sampled norm of products of up to 32
# unit-chunk Jacobians from 110 phases along the orbit.
function periodic_diagnostics(; h = 1e-4, phases = 110, nmax = 32)
    rk4(u, δ) = solve(remake(base["periodic"]; u0 = u, tspan = (0.0, δ)), RK4(h = δ)).u[end]  # one step of length δ
    steps = chunk("periodic", RK4(h = h), 10.0, start["periodic"]).u
    k = findfirst(j -> j > 5 && steps[j][1] < 0 <= steps[j + 1][1], 1:(length(steps) - 1))
    a, b = 0.0, h
    for _ in 1:50
        m = (a + b) / 2
        rk4(steps[k], m)[1] < 0 ? (a = m) : (b = m)
    end
    period = (k - 1) * h + (a + b) / 2
    nt = floor(Int, period / h)
    M = ForwardDiff.jacobian(u -> rk4(flow("periodic", RK4(h = h), nt * h, u), period - nt * h), start["periodic"])
    growth, v = zeros(nmax), start["periodic"]
    advance = round(Int, period / phases / h) * h
    for _ in 1:phases
        x, A = v, Matrix{Float64}(I, 3, 3)
        for n in 1:nmax
            A = jacobian("periodic", RK4(h = h), h, 1.0, x) * A
            x = flow("periodic", RK4(h = h), 1.0, x)
            growth[n] = max(growth[n], opnorm(A))
        end
        v = flow("periodic", RK4(h = h), advance, v)
    end
    return (; period, return_error = norm(rk4(steps[k], (a + b) / 2) - start["periodic"]), multipliers = eigvals(M),
            growth, phases, nmax)
end

records_dir = joinpath(@__DIR__, "Records")
files = ("prediction", "prediction_short", "prediction_coarse", "prediction_linear", "refinement", "roundoff",
         "periodic_diagnostics")
mkpath(records_dir)
todo = filter(f -> !isfile(joinpath(records_dir, f * ".jls")), files)
foreach(f -> println("Keeping Records/$f.jls"), setdiff(files, todo))

jobs = Dict(
    "prediction" => () -> [prediction(name, ξ) for name in ("logistic", "lorenz", "periodic") for ξ in (10, 100)],
    "prediction_short" => () -> [short_window(ξ, h) for ξ in (10, 100) for h in (1e-4, 5e-5)],
    "prediction_coarse" => () -> [coarse_start(N, H, m) for N in (16, 128) for H in (0.01, 0.02) for m in ("rk4", "euler")],
    "prediction_linear" => () -> [linear(a, N) for a in (-1.0, 1.0) for N in (8, 32)],
    "refinement" => () -> [refinement("logistic", 1e-3); refinement("lorenz", 1e-4)],
    "roundoff" => () -> [roundoff(32); roundoff(64)],
    "periodic_diagnostics" => periodic_diagnostics,
)
# Each record is written as soon as it is done, so one failure loses no other record.
@sync for f in todo
    Threads.@spawn begin
        record = jobs[f]()
        serialize(joinpath(records_dir, f * ".jls"), record)
        println("Wrote Records/$f.jls")
    end
end

# Convergence-order gate for SDC, mirroring qmat's `tests/test_qdelta/test_timestepping.py`.
#
# For each combination of nodes, quadrature type, sweeper, sweep count and step
# update: integrate the Dahlquist problem `u' = iu` on `[0, 2π]` at three step
# sizes, take the `L∞` error over the step solutions, fit `log(err)` against
# `log(dt)` by least squares, and require the slope to sit within 0.5 of the
# theoretical order with a regression residual below 0.05. Both the step-size
# triples and the tolerances are qmat's.

using OrdinaryDiffEqSDC
using SciMLBase
using Test

include("sdc_reference.jl")

const LAMBDA = 1im
const U0 = 1.0 + 0.0im
const TEND = 2 * π

dahlquist(u, p, t) = LAMBDA * u
dahlquist_jac(u, p, t) = LAMBDA

# The analytic Jacobian is supplied because ForwardDiff cannot differentiate a
# complex-valued right-hand side; it also makes each Newton solve exact, so the
# measured order reflects the sweep and nothing else.
const DAHLQUIST = ODEProblem(
    ODEFunction(dahlquist; jac = dahlquist_jac), U0, (0.0, TEND)
)

"""qmat's step-size triples, chosen so each order is measured before round-off."""
function nsteps_for_order(order)
    order == 1 && return [64, 128, 256]
    order == 2 && return [32, 64, 128]
    order == 3 && return [16, 32, 64]
    order in (4, 5) && return [8, 16, 32]
    order in (6, 7) && return [4, 8, 16]
    return [1, 2, 4]
end

"""Least-squares slope of `log10(err)` against `log10(dt)`, plus its residual."""
function numerical_order(nsteps, errors)
    x = log10.(1 ./ nsteps)
    y = log10.(errors)
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
    intercept = ym - slope * xm
    residual = sqrt(sum((y .- (intercept .+ slope .* x)) .^ 2) / length(x))
    return slope, residual
end

"""`L∞` error of the step solutions against `u₀exp(λt)`, optionally perturbing the tableau."""
function dahlquist_error(alg, nsteps; perturb! = nothing)
    integrator = init(
        DAHLQUIST, alg; dt = TEND / nsteps, adaptive = false,
        save_everystep = true, dense = false
    )
    perturb! === nothing || perturb!(integrator.cache.tab)
    solve!(integrator)
    sol = integrator.sol
    return maximum(abs.(sol.u .- U0 .* exp.(LAMBDA .* sol.t)))
end

function measured_order(alg, expected; perturb! = nothing)
    nsteps = nsteps_for_order(expected)
    errors = [dahlquist_error(alg, n; perturb!) for n in nsteps]
    return numerical_order(nsteps, errors)
end

sdc_alg(M, node_type, quad_type, sweeper, K, step_update) = SDC(
    num_nodes = M, node_type = node_type, quad_type = quad_type,
    num_sweeps = K, sweeper = sweeper, step_update = step_update
)

@testset "SDC Dahlquist convergence order" begin
    for (M, node_type, quad_type, sweeper, K, step_update, expected, bonus) in ORDER_CASES
        alg = sdc_alg(M, node_type, quad_type, sweeper, K, step_update)
        order, residual = measured_order(alg, expected)
        @testset "M=$M $node_type/$quad_type $sweeper K=$K $step_update" begin
            # `alg_order` implements the base rule, without qmat's table of
            # empirical bonus cases, so on those it is a strict lower bound.
            if bonus
                @test OrdinaryDiffEqSDC.alg_order(alg) < expected
            else
                @test OrdinaryDiffEqSDC.alg_order(alg) == expected
            end
            @test residual < 0.05
            @test abs(order - expected) < 0.5
        end
    end
end

# `:LU` (Weiser 2015) and `:MIN_SR_NS` (Čaklović et al. 2024) are known to gain
# more than one order per sweep on some node sets, and no closed-form predictor
# covers them, so they are gated as a lower bound rather than an equality.
@testset "SDC high-efficiency sweepers reach at least the predicted order" begin
    for sweeper in (:LU, :MIN_SR_NS), M in (3, 4), K in (1, 2, 3)
        alg = sdc_alg(M, :Legendre, :RadauRight, sweeper, K, :quadrature)
        expected = OrdinaryDiffEqSDC.alg_order(alg)
        order, residual = measured_order(alg, expected)
        @testset "$sweeper M=$M K=$K" begin
            @test residual < 0.05
            @test order > expected - 0.5
        end
    end
end

# ---------------------------------------------------------------------------
# Negative controls.
#
# A gate that can only pass proves nothing. These four checks establish that the
# order measurement responds to the coefficients it should respond to and not to
# the ones it should not.
# ---------------------------------------------------------------------------

@testset "SDC negative controls" begin
    M, K = 3, 3
    alg = sdc_alg(M, :Legendre, :RadauRight, :BE, K, :quadrature)
    expected = OrdinaryDiffEqSDC.alg_order(alg)
    @test expected == 4

    baseline, baseline_residual = measured_order(alg, expected)
    @test baseline_residual < 0.05
    @test abs(baseline - expected) < 0.5

    # (1) Q sets the collocation problem, so corrupting it must destroy the
    # order — even by a perturbation far smaller than the discretisation error.
    small_q, _ = measured_order(alg, expected; perturb! = tab -> (tab.Q[2, 1] += 1.0e-3))
    @test small_q < expected - 0.5

    # (2) A larger perturbation of Q must degrade it further, down to first order.
    large_q, _ = measured_order(alg, expected; perturb! = tab -> (tab.Q[2, 1] += 5.0e-2))
    @test large_q < small_q
    @test large_q < 1.5

    # (3) The step-update weights are the other half of the collocation rule.
    bad_w, _ = measured_order(
        alg, expected; perturb! = tab -> (tab.weights[1] += 1.0e-3)
    )
    @test bad_w < expected - 0.5

    # (4) The control on the controls: QΔ is only a preconditioner, so
    # perturbing it changes how fast the sweeps converge but not the fixed point
    # they converge to. The order must survive. If this one failed, the three
    # above would be telling us nothing more than "the test is sensitive to any
    # change at all".
    diag_qd, diag_residual = measured_order(
        alg, expected; perturb! = tab -> (tab.QΔ[2, 2] += 5.0e-2)
    )
    @test diag_residual < 0.05
    @test abs(diag_qd - expected) < 0.5

    lower_qd, lower_residual = measured_order(
        alg, expected; perturb! = tab -> (tab.QΔ[2, 1] += 5.0e-2)
    )
    @test lower_residual < 0.05
    @test abs(lower_qd - expected) < 0.5

    # (5) The gate tracks the sweep count: one fewer sweep, one lower order.
    fewer = sdc_alg(M, :Legendre, :RadauRight, :BE, K - 1, :quadrature)
    @test OrdinaryDiffEqSDC.alg_order(fewer) == expected - 1
    order, _ = measured_order(fewer, expected - 1)
    @test abs(order - (expected - 1)) < 0.5
end

# ---------------------------------------------------------------------------
# Non-linear problems: the linear gate above cannot see an error in how the
# right-hand side is assembled per node, only in the coefficients.
# ---------------------------------------------------------------------------

riccati(u, p, t) = -u^2
riccati_exact(t) = 1 / (1 + t)
const RICCATI = ODEProblem(riccati, 1.0, (0.0, 2.0))

# 2-D system whose exact solution is the unit circle: the extra term vanishes on
# the solution, so `[cos t, sin t]` solves it exactly while the Jacobian stays
# genuinely solution dependent.
function circle!(du, u, p, t)
    r = u[1]^2 + u[2]^2 - 1
    du[1] = -u[2] - r * u[1]
    du[2] = u[1] - r * u[2]
    return nothing
end
circle_exact(t) = [cos(t), sin(t)]
const CIRCLE = ODEProblem(circle!, [1.0, 0.0], (0.0, 2.0))

function nonlinear_order(prob, exact, alg, nsteps)
    errors = map(nsteps) do n
        sol = solve(
            prob, alg; dt = (prob.tspan[2] - prob.tspan[1]) / n,
            adaptive = false, save_everystep = true, dense = false
        )
        maximum(maximum(abs.(u .- exact(t))) for (u, t) in zip(sol.u, sol.t))
    end
    return numerical_order(nsteps, errors)
end

@testset "SDC non-linear convergence order" begin
    @testset "scalar Riccati, out-of-place, autodiff Jacobian" begin
        for K in 1:3
            alg = sdc_alg(3, :Legendre, :RadauRight, :BE, K, :quadrature)
            expected = OrdinaryDiffEqSDC.alg_order(alg)
            order, residual = nonlinear_order(
                RICCATI, riccati_exact, alg, [8, 16, 32]
            )
            @test residual < 0.05
            @test abs(order - expected) < 0.5
        end
    end

    @testset "2-D system, in-place, autodiff Jacobian" begin
        for (sweeper, K) in [(:BE, 2), (:BE, 3), (:Trapezoid, 2), (:LU, 3)]
            alg = sdc_alg(3, :Legendre, :RadauRight, sweeper, K, :quadrature)
            expected = OrdinaryDiffEqSDC.alg_order(alg)
            order, residual = nonlinear_order(
                CIRCLE, circle_exact, alg, [8, 16, 32]
            )
            @test residual < 0.05
            @test order > expected - 0.5
        end
    end

    @testset "explicit sweeper on a non-linear problem" begin
        # QΔ strictly lower triangular: every node is explicit, no solver is
        # built at all, and the order behaviour must be unchanged.
        alg = sdc_alg(3, :Legendre, :RadauRight, :FE, 3, :quadrature)
        expected = OrdinaryDiffEqSDC.alg_order(alg)
        order, residual = nonlinear_order(CIRCLE, circle_exact, alg, [16, 32, 64])
        @test residual < 0.05
        @test abs(order - expected) < 0.5
    end
end

@testset "SDC in-place and out-of-place agree" begin
    alg = sdc_alg(4, :Legendre, :RadauRight, :BE, 4, :quadrature)
    oop = solve(
        ODEProblem(
            (u, p, t) -> [
                -u[2] - (sum(abs2, u) - 1) * u[1],
                u[1] - (sum(abs2, u) - 1) * u[2],
            ], [1.0, 0.0], (0.0, 2.0)
        ),
        alg; dt = 0.25, adaptive = false
    )
    iip = solve(CIRCLE, alg; dt = 0.25, adaptive = false)
    @test oop.u[end] ≈ iip.u[end] rtol = 1.0e-10
end

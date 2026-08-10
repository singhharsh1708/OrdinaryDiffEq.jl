using OrdinaryDiffEqSDC
using OrdinaryDiffEqSDC: SDCTableau, sdc_collocation_order, sdc_solver_index,
    SDC_NODE_TYPES, SDC_QUAD_TYPES, SDC_SWEEPERS
using LinearAlgebra
using Test

include("sdc_reference.jl")

@testset "SDC coefficients match qmat" begin
    for case in QMAT_GOLDEN
        label = "M=$(case.M) $(case.node_type)/$(case.quad_type)"
        for (sweeper, QDelta) in case.QDelta
            tab = SDCTableau(
                Float64, case.M, case.node_type, case.quad_type, sweeper
            )
            @testset "$label $sweeper" begin
                @test tab.nodes ≈ case.nodes atol = 1.0e-13
                @test tab.weights ≈ case.weights atol = 1.0e-13
                @test tab.Q ≈ case.Q atol = 1.0e-13
                @test tab.QΔ ≈ QDelta atol = 1.0e-13
            end
        end
        @test sdc_collocation_order(case.M, case.node_type, case.quad_type) ==
            case.coll_order
    end
end

@testset "SDC coefficient invariants" begin
    for node_type in SDC_NODE_TYPES, quad_type in SDC_QUAD_TYPES
        minimum_nodes = quad_type in (:Lobatto, :RadauLeft) ? 2 : 1
        for M in max(2, minimum_nodes):6
            tab = SDCTableau(Float64, M, node_type, quad_type, :BE)
            τ = tab.nodes
            @testset "$node_type/$quad_type M=$M" begin
                @test issorted(τ)
                @test all(0 .<= τ .<= 1)
                quad_type in (:Lobatto, :RadauLeft) && @test τ[1] == 0
                quad_type in (:Lobatto, :RadauRight) && @test τ[end] == 1
                # Q ⋅ τ^{n} integrates monomials exactly up to degree M-1.
                for n in 0:(M - 1)
                    @test tab.Q * (τ .^ n) ≈ (τ .^ (n + 1)) ./ (n + 1) atol = 1.0e-13
                    @test sum(tab.weights .* τ .^ n) ≈ 1 / (n + 1) atol = 1.0e-13
                end
                # Backward Euler sweeper: lower triangular with row sums τ.
                @test tab.QΔ ≈ LowerTriangular(tab.QΔ)
                @test tab.QΔ * ones(M) ≈ τ atol = 1.0e-14
            end
        end
    end
end

@testset "SDC sweeper shapes" begin
    M = 4
    for sweeper in SDC_SWEEPERS
        tab = SDCTableau(Float64, M, :Legendre, :RadauRight, sweeper)
        @test tab.QΔ ≈ LowerTriangular(tab.QΔ)
        # The sweepers advertised as parallel-ready must actually be diagonal:
        # that is the property the whole phase-two design rests on.
        if sweeper in OrdinaryDiffEqSDC.SDC_DIAGONAL_SWEEPERS
            @test tab.QΔ ≈ Diagonal(tab.QΔ)
        else
            @test !(tab.QΔ ≈ Diagonal(tab.QΔ))
        end
        if sweeper === :FE
            @test all(iszero, diag(tab.QΔ))
        end
    end
    # Trapezoid is the average of the forward and backward Euler sweepers.
    be = SDCTableau(Float64, M, :Legendre, :RadauRight, :BE).QΔ
    fe = SDCTableau(Float64, M, :Legendre, :RadauRight, :FE).QΔ
    tr = SDCTableau(Float64, M, :Legendre, :RadauRight, :Trapezoid).QΔ
    @test tr ≈ (be .+ fe) ./ 2
end

@testset "SDC explicit-node bookkeeping" begin
    # τ₁ = 0 for Lobatto and Radau-left, so node 1 has no implicit solve.
    for quad_type in (:Lobatto, :RadauLeft), sweeper in (:BE, :BEpar, :MIN_SR_NS)
        tab = SDCTableau(Float64, 4, :Legendre, quad_type, sweeper)
        index = sdc_solver_index(tab.QΔ)
        @test index[1] == 0
        @test index[2:end] == collect(1:3)
    end
    tab = SDCTableau(Float64, 4, :Legendre, :RadauRight, :BE)
    @test sdc_solver_index(tab.QΔ) == collect(1:4)
    tab = SDCTableau(Float64, 4, :Legendre, :RadauRight, :FE)
    @test all(iszero, sdc_solver_index(tab.QΔ))
end

@testset "SDC coefficients are precision generic" begin
    small = SDCTableau(Float64, 5, :Legendre, :RadauRight, :LU)
    big = SDCTableau(BigFloat, 5, :Legendre, :RadauRight, :LU)
    @test Float64.(big.nodes) ≈ small.nodes atol = 1.0e-15
    @test Float64.(big.Q) ≈ small.Q atol = 1.0e-15
    τ = big.nodes
    # Extended precision exposes any conditioning loss in the Q construction.
    @test maximum(abs, big.Q * (τ .^ 3) - (τ .^ 4) ./ 4) < 1.0e-40
end

@testset "SDC argument validation" begin
    @test_throws ArgumentError SDC(node_type = :Chebyshev)
    @test_throws ArgumentError SDC(quad_type = :Nope)
    @test_throws ArgumentError SDC(sweeper = :NotASweeper)
    @test_throws ArgumentError SDC(step_update = :whatever)
    @test_throws ArgumentError SDC(num_sweeps = -1)
    @test_throws ArgumentError SDC(num_nodes = 1, quad_type = :Lobatto)
    @test_throws ArgumentError SDC(quad_type = :Gauss, step_update = :lastnode)
    @test SDC(quad_type = :RadauRight, step_update = :lastnode) isa SDC
end

@testset "SDC alg_order" begin
    # One order per sweep, plus one for the quadrature update, capped by the
    # collocation order (2M-1 for Radau).
    for K in 0:8
        alg = SDC(num_nodes = 3, quad_type = :RadauRight, num_sweeps = K)
        @test OrdinaryDiffEqSDC.alg_order(alg) == max(1, min(K + 1, 5))
    end
    # A second-order sweeper gains two orders on the first sweep.
    @test OrdinaryDiffEqSDC.alg_order(
        SDC(num_nodes = 4, quad_type = :Gauss, num_sweeps = 1, sweeper = :Trapezoid)
    ) == 3
    @test OrdinaryDiffEqSDC.alg_order(
        SDC(num_nodes = 4, quad_type = :Gauss, num_sweeps = 1, sweeper = :BE)
    ) == 2
    # Last-node updates lose the extra order the quadrature update provides.
    @test OrdinaryDiffEqSDC.alg_order(
        SDC(num_nodes = 4, quad_type = :RadauRight, num_sweeps = 3, step_update = :lastnode)
    ) == 3
end

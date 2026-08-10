# One `NLSolver` is built per implicit node, never one shared solver.
#
# Node `m`'s implicit solve is `u_m = tmp + QΔ[m,m] ⋅ dt ⋅ f(u_m)`, so each node
# has its own `γ_m = QΔ[m,m]` and therefore its own `W = M/(γ_m dt) - J`. A
# shared solver would have to refactorise `W` at every node of every sweep, and
# `do_newJW`'s `isfreshJ` branch would in fact skip that refactorisation once the
# Jacobian is current at this `t`, silently reusing the wrong `W`. Per-node
# solvers give each node a private `W` that is built once per step and reused
# across all `K` sweeps. `OrdinaryDiffEqPDIRK` uses the same construction for the
# same reason.
#
# Nodes whose `QΔ` diagonal entry vanishes are explicit and get no solver at all;
# `solver_index[m] == 0` marks them. This happens for `:FE` and `:Picard`
# everywhere, and for the first node of every sweeper when `τ₁ = 0`
# (`:Lobatto` and `:RadauLeft`).

struct SDCConstantCache{N, TabType} <: OrdinaryDiffEqConstantCache
    nlsolver::N
    tab::TabType
    solver_index::Vector{Int}
end

@cache mutable struct SDCCache{uType, rateType, N, TabType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    ubuf::uType
    k::rateType
    z::Vector{uType}
    znew::Vector{uType}
    nlsolver::N
    tab::TabType
    solver_index::Vector{Int}
end

# Non-FSAL: the sweep never reuses the previous step's final derivative.
get_fsalfirstlast(cache::SDCCache, u) = (nothing, nothing)

"""
    sdc_solver_index(QΔ)

Map each node to its position in the solver vector, or to `0` when the node is
explicit because `QΔ[m, m]` vanishes.
"""
function sdc_solver_index(QΔ::AbstractMatrix)
    M = size(QΔ, 1)
    index = zeros(Int, M)
    count = 0
    for m in 1:M
        if !iszero(QΔ[m, m])
            count += 1
            index[m] = count
        end
    end
    return index
end

function alg_cache(
        alg::SDC, u, rate_prototype,
        ::Type{uEltypeNoUnits}, ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
        uprev, uprev2, f, t, dt, reltol, p, calck,
        ::Val{true}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    tab = SDCTableau(
        constvalue(uBottomEltypeNoUnits), alg.num_nodes, alg.node_type,
        alg.quad_type, alg.sweeper
    )
    M = alg.num_nodes
    solver_index = sdc_solver_index(tab.QΔ)
    nlsolver = [
        build_nlsolver(
                alg, u, uprev, p, t, dt, f, rate_prototype, uEltypeNoUnits,
                uBottomEltypeNoUnits, tTypeNoUnits, tab.QΔ[m, m], tab.nodes[m],
                Val(true), verbose
            ) for m in 1:M if !iszero(solver_index[m])
    ]
    return SDCCache(
        u, uprev, zero(u), zero(u), zero(rate_prototype),
        [zero(u) for _ in 1:M], [zero(u) for _ in 1:M],
        nlsolver, tab, solver_index
    )
end

function alg_cache(
        alg::SDC, u, rate_prototype,
        ::Type{uEltypeNoUnits}, ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
        uprev, uprev2, f, t, dt, reltol, p, calck,
        ::Val{false}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    tab = SDCTableau(
        constvalue(uBottomEltypeNoUnits), alg.num_nodes, alg.node_type,
        alg.quad_type, alg.sweeper
    )
    solver_index = sdc_solver_index(tab.QΔ)
    nlsolver = [
        build_nlsolver(
                alg, u, uprev, p, t, dt, f, rate_prototype, uEltypeNoUnits,
                uBottomEltypeNoUnits, tTypeNoUnits, tab.QΔ[m, m], tab.nodes[m],
                Val(false), verbose
            ) for m in 1:(alg.num_nodes) if !iszero(solver_index[m])
    ]
    return SDCConstantCache(nlsolver, tab, solver_index)
end

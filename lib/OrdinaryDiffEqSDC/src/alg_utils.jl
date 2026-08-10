isfsal(::SDC) = false

"""
    sdc_iteration_order(alg)

Order reached by the SDC iteration itself, before the collocation cap.

One order is gained per sweep, two on the first sweep for a second-order
sweeper, and one more if the step update is the quadrature rule rather than the
last node value. This is `qmat.utils.sdc.getOrderSDC` without its table of
empirically found bonus cases, so for some node/sweeper combinations (notably
`:LU` and `:MIN_SR_NS`) the observed order is higher than this predicts.
"""
function sdc_iteration_order(alg::SDC)
    K = alg.num_sweeps
    order = 0
    if K > 0
        order += alg.sweeper in SDC_SECOND_ORDER_SWEEPERS ? 2 : 1
        order += K - 1
    end
    alg.step_update === :quadrature && (order += 1)
    return order
end

function alg_order(alg::SDC)
    coll = sdc_collocation_order(alg.num_nodes, alg.node_type, alg.quad_type)
    return max(1, min(sdc_iteration_order(alg), coll))
end

"""
    sdc_validate(num_nodes, node_type, quad_type, num_sweeps, sweeper, step_update)

Reject unusable parameter combinations at algorithm construction time.

This deliberately does not go through `prepare_alg`: the generic
`prepare_alg(::OrdinaryDiffEqImplicitAlgorithm, ::AbstractArray, …)` in
`OrdinaryDiffEqDifferentiation` does the autodiff preparation every implicit
solver needs, and adding an `SDC` method here would make the two ambiguous.
"""
function sdc_validate(
        num_nodes::Int, node_type::Symbol, quad_type::Symbol,
        num_sweeps::Int, sweeper::Symbol, step_update::Symbol
    )
    node_type in SDC_NODE_TYPES || throw(
        ArgumentError("SDC: `node_type` must be one of $(SDC_NODE_TYPES), got :$(node_type)")
    )
    quad_type in SDC_QUAD_TYPES || throw(
        ArgumentError("SDC: `quad_type` must be one of $(SDC_QUAD_TYPES), got :$(quad_type)")
    )
    sweeper in SDC_SWEEPERS || throw(
        ArgumentError("SDC: `sweeper` must be one of $(SDC_SWEEPERS), got :$(sweeper)")
    )
    step_update in SDC_STEP_UPDATES || throw(
        ArgumentError(
            "SDC: `step_update` must be one of $(SDC_STEP_UPDATES), got :$(step_update)"
        )
    )
    num_sweeps >= 0 ||
        throw(ArgumentError("SDC: `num_sweeps` must be ≥ 0, got $(num_sweeps)"))
    minimum_nodes = quad_type in (:Lobatto, :RadauLeft) ? 2 : 1
    num_nodes >= minimum_nodes || throw(
        ArgumentError(
            "SDC: `num_nodes` must be ≥ $(minimum_nodes) for `quad_type = :$(quad_type)`, " *
                "got $(num_nodes)"
        )
    )
    if step_update === :lastnode && !(quad_type in (:RadauRight, :Lobatto))
        throw(
            ArgumentError(
                "SDC: `step_update = :lastnode` needs the last node to be the right " *
                    "endpoint, so `quad_type` must be `:RadauRight` or `:Lobatto`, " *
                    "got :$(quad_type)"
            )
        )
    end
    return nothing
end

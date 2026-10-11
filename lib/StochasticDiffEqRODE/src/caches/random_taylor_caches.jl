struct RandomTaylorStage{T}
    halfwidth::Int
    map::Matrix{T}
    weights::Vector{T}
end

function RandomTaylorStage{T}(halfwidth, columns) where {T}
    return RandomTaylorStage{T}(
        halfwidth, zeros(T, 2halfwidth + 1, columns), zeros(T, 2halfwidth + 1)
    )
end

struct RandomTaylorTableau{T}
    holder::T
    lead::Vector{Int}
    depth::Vector{Int}
    prefix::Vector{Int}
    suffix::Vector{Int}
    singles::Int
    recip::Vector{T}
    coeffs::Vector{Vector{T}}
    moments::Vector{T}
    integrals::Vector{T}
    directions::Vector{Vector{T}}
    root::RandomTaylorStage{T}
    children::Vector{RandomTaylorStage{T}}
    grandchildren::Vector{Vector{RandomTaylorStage{T}}}
end

random_taylor_maxsum(order, holder, i) = ceil(Int, (order + 1 - i) / holder) - 1

function random_taylor_indices(order, holder)
    indices = Vector{Int}[]
    for i in 1:3
        smax = random_taylor_maxsum(order, holder, i)
        for index in Iterators.product(ntuple(_ -> 0:smax, i)...)
            sum(index) <= smax && push!(indices, collect(index))
        end
    end
    return indices
end

stencil_accuracy(n, k) = isodd(k) ? 2n + 1 - k : 2n + 2 - k

function stencil_halfwidth(need, ks)
    n = 0
    while any(k -> stencil_accuracy(n, k) < need - k, ks)
        n += 1
    end
    return n
end

function stencil_weights(::Type{T}, n, k) where {T}
    V = [(j // 1)^m for m in 0:(2n), j in big.(-n:n)]
    rhs = zeros(Rational{BigInt}, 2n + 1)
    rhs[k + 1] = 1
    return T.(V \ rhs)
end

function add_stencil!(stage, weights, column, factor)
    offset = stage.halfwidth - (length(weights) - 1) ÷ 2
    for j in eachindex(weights)
        stage.map[offset + j, column] += factor * weights[j]
    end
    return nothing
end

function moment_coefficients(::Type{T}, index) where {T}
    coeffs = zeros(Rational{BigInt}, sum(index) + 1)
    for js in Iterators.product((0:a for a in index)...)
        term = big(1) // 1
        tail = 0
        for r in length(index):-1:1
            tail += js[r] + 1
            term *= binomial(index[r], js[r]) // tail
        end
        coeffs[sum(js) + 1] += term
    end
    return T.(coeffs)
end

function RandomTaylorTableau(::Type{T}, order, holder) where {T}
    indices = random_taylor_indices(order, holder)
    columns = length(indices)
    K1, K2, K3 = ntuple(i -> random_taylor_maxsum(order, holder, i), 3)
    need = ntuple(i -> (order + 1 - i) / holder, 3)
    n1 = stencil_halfwidth(need[1], 1:K1)
    n2 = [stencil_halfwidth(need[2] - b, 1:(K2 - b)) for b in 0:K2]
    n3 = [stencil_halfwidth(need[3] - s, 1:(K3 - s)) for s in 0:K3]
    nd = [b == 0 ? 0 : stencil_halfwidth(need[2], b:b) for b in 0:K2]
    directions = [stencil_weights(T, nd[b + 1], b) for b in 0:K2]
    root = RandomTaylorStage{T}(n1, columns)
    children = map(0:K2) do b
        inner = maximum(nd[a + 1] for a in 1:(K3 - b); init = 0)
        return RandomTaylorStage{T}(max(n2[b + 1], inner), columns)
    end
    grandchildren = [
        [RandomTaylorStage{T}(n3[a + b + 1], columns) for a in 0:(K3 - b)] for b in 0:K2
    ]
    place(column, k) = add_stencil!(root, stencil_weights(T, n1, k), column, 1)
    function place(column, k, outer)
        weights = stencil_weights(T, n2[outer + 1], k)
        add_stencil!(root, weights, column, -3)
        add_stencil!(children[outer + 1], weights, column, 1)
        return nothing
    end
    function place(column, k, inner, outer)
        weights = stencil_weights(T, n3[inner + outer + 1], k)
        add_stencil!(root, weights, column, 9)
        add_stencil!(children[inner + 1], weights, column, -3)
        add_stencil!(children[outer + 1], weights, column, -3)
        add_stencil!(grandchildren[outer + 1][inner + 1], weights, column, 1)
        return nothing
    end
    for (column, index) in enumerate(indices)
        place(column, index...)
    end
    lead = first.(indices)
    depth = length.(indices)
    prefix = [something(findfirst(==(index[1:(end - 1)]), indices), 0) for index in indices]
    suffix = [something(findfirst(==(index[2:end]), indices), 0) for index in indices]
    recip = [one(T) / k for k in 1:(K1 + 1)]
    coeffs = [moment_coefficients(T, index) for index in indices]
    return RandomTaylorTableau{T}(
        T(holder), lead, depth, prefix, suffix, K1 + 1, recip, coeffs, ones(T, columns),
        zeros(T, columns), directions, root, children, grandchildren
    )
end

struct RandomTaylorConstantCache{T, rateType} <: StochasticDiffEqConstantCache
    tab::RandomTaylorTableau{T}
    rtmp1::Vector{rateType}
    rtmp2::Vector{rateType}
end

@cache struct RandomTaylorCache{uType, rateType, T} <: StochasticDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    tmp1::uType
    rtmp::rateType
    rtmp1::Vector{rateType}
    rtmp2::Vector{rateType}
    tab::RandomTaylorTableau{T}
end

function alg_cache(
        alg::RandomTaylor, prob, u, ΔW, ΔZ, p, rate_prototype,
        noise_rate_prototype, jump_rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, f, t, dt,
        ::Type{Val{false}}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    T = promote_type(tTypeNoUnits, eltype(prob.noise.W))
    tab = RandomTaylorTableau(T, alg.order, alg.holder)
    rtmp1 = Vector{typeof(rate_prototype)}(undef, 2 * tab.root.halfwidth + 1)
    rtmp2 = Vector{typeof(rate_prototype)}(
        undef, 2 * maximum(child -> child.halfwidth, tab.children; init = 0) + 1
    )
    return RandomTaylorConstantCache(tab, rtmp1, rtmp2)
end

function alg_cache(
        alg::RandomTaylor, prob, u, ΔW, ΔZ, p, rate_prototype,
        noise_rate_prototype, jump_rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, f, t, dt,
        ::Type{Val{true}}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    T = promote_type(tTypeNoUnits, eltype(prob.noise.W))
    tab = RandomTaylorTableau(T, alg.order, alg.holder)
    tmp = zero(u)
    tmp1 = zero(u)
    rtmp = zero(rate_prototype)
    rtmp1 = [zero(rate_prototype) for _ in 1:(2 * tab.root.halfwidth + 1)]
    rtmp2 = [
        zero(rate_prototype) for
            _ in 1:(2 * maximum(child -> child.halfwidth, tab.children; init = 0) + 1)
    ]
    return RandomTaylorCache(u, uprev, tmp, tmp1, rtmp, rtmp1, rtmp2, tab)
end

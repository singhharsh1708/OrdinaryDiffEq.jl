# Coefficient generation for SDC: collocation nodes, the quadrature matrix `Q`,
# the step-update weights, and the sweep preconditioner `QΔ`.
#
# Everything here depends only on `(M, node_type, quad_type, sweeper)` and the
# element type — never on `dt`, `t`, `u` or `p` — so it is evaluated once in
# `alg_cache` and stored in an `SDCTableau`.
#
# The conventions match `qmat` (https://github.com/Parallel-in-Time/qmat):
# nodes live on `[0, 1]` in increasing order, `Q[m, j] = ∫₀^{τₘ} ℓⱼ(s) ds` with
# `ℓⱼ` the Lagrange basis on the nodes, and `weights[j] = ∫₀¹ ℓⱼ(s) ds`.

const SDC_NODE_TYPES = (:Legendre, :Equidistant)
const SDC_QUAD_TYPES = (:Gauss, :RadauLeft, :RadauRight, :Lobatto)
const SDC_SWEEPERS = (:BE, :FE, :Trapezoid, :LU, :Picard, :BEpar, :MIN_SR_NS)
const SDC_STEP_UPDATES = (:quadrature, :lastnode)

# Sweepers whose `QΔ` is diagonal, so that a sweep decouples across the nodes.
const SDC_DIAGONAL_SWEEPERS = (:Picard, :BEpar, :MIN_SR_NS)

# Sweepers that are second order accurate on their own, and therefore gain two
# orders on the first sweep instead of one.
const SDC_SECOND_ORDER_SWEEPERS = (:Trapezoid,)

# Working precision for coefficient generation. The monomial Vandermonde used to
# build `Q` is ill-conditioned (cond ≈ 1e7 already at M = 8), so the whole
# derivation runs in `BigFloat` and is rounded down to the element type at the
# very end. It happens once per `alg_cache`, so the cost is irrelevant.
const SDC_COEFF_PRECISION = 256

struct SDCTableau{T}
    nodes::Vector{T}
    weights::Vector{T}
    Q::Matrix{T}
    QΔ::Matrix{T}
end

"""
    _jacobi(n, α, β, x)

Evaluate the Jacobi polynomial ``P_n^{(α,β)}(x)`` by its three-term recurrence.
`α` and `β` are `0` or `1` here; `P_n^{(0,0)}` is the Legendre polynomial.
"""
function _jacobi(n::Int, α::Int, β::Int, x::T) where {T}
    n <= 0 && return one(T)
    p1 = (α + 1) + (α + β + 2) * (x - one(T)) / 2
    n == 1 && return p1
    p0 = one(T)
    for k in 2:n
        a = T(2k * (k + α + β) * (2k + α + β - 2))
        b = T((2k + α + β - 1) * (2k + α + β) * (2k + α + β - 2))
        c = T((2k + α + β - 1) * (α^2 - β^2))
        d = T(2 * (k + α - 1) * (k + β - 1) * (2k + α + β))
        p0, p1 = p1, ((b * x + c) * p1 - d * p0) / a
    end
    return p1
end

"""
    _bisect(f, a, b)

Bisect `f` on a bracket `[a, b]` with `f(a)f(b) < 0` down to the resolution of
the element type. Derivative free, so it is generic in the precision.

The iteration cap is not decoration. `BigFloat` has no smallest normal, so if a
bracket endpoint sits exactly on the root the midpoint halves towards it through
the whole exponent range and `m == a` never fires. Exact roots are removed by
the caller before bisection, and this bound catches anything that slips through.
"""
function _bisect(f::F, a::T, b::T) where {F, T}
    fa = f(a)
    iszero(fa) && return a
    positive = fa > 0
    for _ in 1:(8 * precision(T) + 64)
        m = (a + b) / 2
        (m == a || m == b) && return m
        fm = f(m)
        iszero(fm) && return m
        if (fm > 0) == positive
            a = m
        else
            b = m
        end
    end
    return (a + b) / 2
end

"""
    _jacobi_roots(n, α, β, T)

All `n` roots of ``P_n^{(α,β)}`` in `(-1, 1)`, in increasing order. The roots are
simple and, for the small `n` used by SDC, well separated, so a uniform
bracketing grid followed by bisection is both robust and generic in `T`.
"""
function _jacobi_roots(n::Int, α::Int, β::Int, ::Type{T}) where {T}
    n <= 0 && return T[]
    ngrid = 20n + 50
    roots = T[]
    xprev = -one(T)
    fprev = _jacobi(n, α, β, xprev)
    for i in 1:ngrid
        x = -one(T) + 2 * T(i) / T(ngrid)
        fx = _jacobi(n, α, β, x)
        if iszero(fx)
            # A grid point landed exactly on a root; this is the common case,
            # not a freak one — odd-degree Legendre polynomials have a root at
            # x = 0 and the grid is symmetric. Note it directly, and let the
            # `iszero(fprev)` guard below skip the following interval so it is
            # not counted twice. Bracketing it would hand `_bisect` an endpoint
            # that is exactly the root.
            push!(roots, x)
        elseif !iszero(fprev) && (fprev > 0) != (fx > 0)
            push!(roots, _bisect(y -> _jacobi(n, α, β, y), xprev, x))
        end
        xprev, fprev = x, fx
    end
    length(roots) == n || throw(
        ArgumentError(
            "SDC: found $(length(roots)) of $n roots for the Jacobi polynomial " *
                "P_$n^($α,$β); node generation failed"
        )
    )
    return roots
end

"""
    _sdc_nodes_big(M, node_type, quad_type)

The `M` collocation nodes on `[0, 1]`, in increasing order, at extended
precision. `quad_type` selects which interval endpoints are nodes: `:Gauss`
(neither), `:RadauLeft` (`0`), `:RadauRight` (`1`), `:Lobatto` (both).
"""
function _sdc_nodes_big(M::Int, node_type::Symbol, quad_type::Symbol)
    T = BigFloat
    if node_type === :Equidistant
        # Closed form, matching qmat's EQUID convention.
        return if quad_type === :Gauss
            [T(m) / T(M + 1) for m in 1:M]
        elseif quad_type === :RadauLeft
            [T(m - 1) / T(M) for m in 1:M]
        elseif quad_type === :RadauRight
            [T(m) / T(M) for m in 1:M]
        else # :Lobatto
            [T(m - 1) / T(M - 1) for m in 1:M]
        end
    end
    # :Legendre — nodes on [-1, 1] from the Jacobi polynomials, then mapped.
    x = if quad_type === :Gauss
        _jacobi_roots(M, 0, 0, T)
    elseif quad_type === :RadauLeft
        vcat(-one(T), _jacobi_roots(M - 1, 0, 1, T))
    elseif quad_type === :RadauRight
        vcat(_jacobi_roots(M - 1, 1, 0, T), one(T))
    else # :Lobatto
        vcat(-one(T), _jacobi_roots(M - 2, 1, 1, T), one(T))
    end
    return (x .+ 1) ./ 2
end

"""
    _lagrange_integrals(τ, uppers)

`P[m, j] = ∫₀^{uppers[m]} ℓⱼ(s) ds` for the Lagrange basis `ℓⱼ` on the nodes `τ`.

Written in the monomial basis: the Lagrange coefficients are `V⁻¹` for the
Vandermonde matrix `V[k, i] = τₖ^{i-1}` (since `ℓⱼ(τₖ) = δⱼₖ`), and the monomial
integrals are `C[m, i] = uppers[m]^i / i`, so the result is `C V⁻¹`.
"""
function _lagrange_integrals(τ::Vector{BigFloat}, uppers::Vector{BigFloat})
    M = length(τ)
    V = [τ[k]^(i - 1) for k in 1:M, i in 1:M]
    C = [uppers[m]^i / i for m in 1:length(uppers), i in 1:M]
    return C / V
end

"""
    sdc_quadrature(τ) -> (weights, Q)

The step-update weights `∫₀¹ ℓⱼ` and the quadrature matrix `Q[m, j] = ∫₀^{τₘ} ℓⱼ`.
"""
function sdc_quadrature(τ::Vector{BigFloat})
    Q = _lagrange_integrals(τ, τ)
    weights = vec(_lagrange_integrals(τ, [one(BigFloat)]))
    return weights, Q
end

"""
    sdc_qdelta(T, sweeper, τ, Q)

The sweep preconditioner `QΔ ≈ Q`. Must be lower triangular for the sweep to
decouple into `M` successive `N`-sized solves; a diagonal `QΔ` decouples them
completely, which is what parallel-across-the-nodes SDC exploits.
"""
function sdc_qdelta(
        ::Type{T}, sweeper::Symbol, τ::Vector{BigFloat}, Q::Matrix{BigFloat}
    ) where {T}
    M = length(τ)
    B = BigFloat
    QΔ = zeros(B, M, M)
    # Distances between consecutive nodes, with the first measured from t₀.
    δ = [i == 1 ? τ[1] : τ[i] - τ[i - 1] for i in 1:M]
    if sweeper === :BE
        for i in 1:M, j in 1:i
            QΔ[i, j] = δ[j]
        end
    elseif sweeper === :FE
        for i in 1:M, j in 1:(i - 1)
            QΔ[i, j] = δ[j + 1]
        end
    elseif sweeper === :Trapezoid
        for i in 1:M
            for j in 1:i
                QΔ[i, j] += δ[j]
            end
            for j in 1:(i - 1)
                QΔ[i, j] += δ[j + 1]
            end
        end
        QΔ ./= 2
    elseif sweeper === :LU
        # Weiser's LU trick: QΔ = Uᵀ from the LU factorisation of Qᵀ.
        # `check = false` because Q is singular whenever τ₁ = 0 (Lobatto and
        # Radau-left), where its first row vanishes. The zero pivot is the right
        # answer there: it makes the first node explicit, which is correct since
        # u(τ₁) = uₙ. scipy's `lu`, which qmat uses, likewise does not check.
        QΔ = Matrix(transpose(LinearAlgebra.lu(transpose(Q); check = false).U))
    elseif sweeper === :Picard
        # QΔ stays zero: the plain (unpreconditioned) Picard iteration.
    elseif sweeper === :BEpar
        for i in 1:M
            QΔ[i, i] = τ[i]
        end
    elseif sweeper === :MIN_SR_NS
        for i in 1:M
            QΔ[i, i] = τ[i] / M
        end
    else
        throw(ArgumentError("SDC: unknown sweeper $(sweeper)"))
    end
    return T.(QΔ)
end

"""
    SDCTableau(T, M, node_type, quad_type, sweeper)

Build every coefficient array the sweep needs, in the element type `T`.
"""
function SDCTableau(
        ::Type{T}, M::Int, node_type::Symbol, quad_type::Symbol, sweeper::Symbol
    ) where {T}
    return setprecision(BigFloat, SDC_COEFF_PRECISION) do
        τ = _sdc_nodes_big(M, node_type, quad_type)
        weights, Q = sdc_quadrature(τ)
        QΔ = sdc_qdelta(T, sweeper, τ, Q)
        SDCTableau{T}(T.(τ), T.(weights), T.(Q), QΔ)
    end
end

"""
    sdc_collocation_order(M, node_type, quad_type)

Order of the underlying collocation method, which caps the order SDC can reach
no matter how many sweeps are taken.

Legendre nodes give the classical Gauss/Radau/Lobatto orders `2M`, `2M-1`,
`2M-2`. Equidistant nodes give the interpolatory order `M`, raised to `M+1` for
even `M` on the symmetric (Gauss and Lobatto) rules. Matches
`qmat.qcoeff.collocation.Collocation.order`.
"""
function sdc_collocation_order(M::Int, node_type::Symbol, quad_type::Symbol)
    if node_type === :Legendre
        return if quad_type === :Gauss
            2M
        elseif quad_type === :Lobatto
            2M - 2
        else # :RadauLeft, :RadauRight
            2M - 1
        end
    end
    if quad_type === :Gauss || quad_type === :Lobatto
        return M + (M % 2)
    end
    return M
end

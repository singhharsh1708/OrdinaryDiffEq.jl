# SDC in OrdinaryDiffEq.jl — phase-one design

Status: written before the phase-one implementation, revised only where the
implementation forced a change (revisions are marked).

## 0. Scope

Phase one delivers **serial SDC**, generic over

| axis | parameter | phase-one values |
|---|---|---|
| number of collocation nodes | `num_nodes` (`M`) | any `M ≥ 1` (`M ≥ 2` for Lobatto / Radau-left) |
| node distribution | `node_type` | `:Legendre`, `:Equidistant` |
| quadrature type | `quad_type` | `:Gauss`, `:RadauLeft`, `:RadauRight`, `:Lobatto` |
| number of sweeps | `num_sweeps` (`K`) | any `K ≥ 0` |
| sweeper / preconditioner | `sweeper` | `:BE`, `:FE`, `:Trapezoid`, `:LU`, `:Picard`, `:BEpar`, `:MIN_SR_NS` |
| step update | `step_update` | `:quadrature`, `:lastnode` |

Explicitly **not** in phase one: parallel-across-the-nodes execution, IMEX
splitting, adaptivity, PFASST, benchmarking. Parallel and IMEX are *designed
for* here (§6, §7) and the phase-one code is checked against those designs.

## 1. Where the family lives

New sub-package `lib/OrdinaryDiffEqSDC`, mirroring `lib/OrdinaryDiffEqMultirate`:

```
lib/OrdinaryDiffEqSDC/
  Project.toml
  src/OrdinaryDiffEqSDC.jl     # module, imports, exports
  src/algorithms.jl            # SDC algorithm struct + keyword constructor
  src/alg_utils.jl             # alg_order, isfsal, prepare_alg, issplit hook
  src/sdc_tableaus.jl          # nodes, Q, weights, QΔ  (all coefficient generation)
  src/sdc_caches.jl            # SDCCache / SDCConstantCache + alg_cache
  src/sdc_perform_step.jl      # initialize! + perform_step!
  test/
```

Not `OrdinaryDiffEqCollocation` and not folded into `OrdinaryDiffEqFIRK`:
FIRK solves the collocation system directly with a full Newton on the `M·N`
coupled system; SDC solves the *same* collocation system by preconditioned
iteration with `M` decoupled `N`-sized solves. Different cache, different
`perform_step!`, different failure modes. Separate package, and it keeps the
parallel-across-the-nodes work (which will add a `threading` field and a thread
pool of solvers) contained.

**Not re-exported from the umbrella `OrdinaryDiffEq` until the sub-package is
registered.** (Standing project rule.)

## 2. Algorithm type

```julia
struct SDC{AD, F, F2, CJ} <: OrdinaryDiffEqNewtonAlgorithm
    num_nodes::Int
    node_type::Symbol
    quad_type::Symbol
    num_sweeps::Int
    sweeper::Symbol
    step_update::Symbol
    linsolve::F
    nlsolve::F2
    autodiff::AD
    concrete_jac::CJ
end
```

Design points:

* **Runtime fields, not type parameters.** `M`, `K` and the symbols could be
  type parameters for compile-time specialisation, but every distinct
  `(M, K, node, quad, sweeper)` would then compile its own `perform_step!`.
  The library convention for parameterised families (`MRAB(k=…)`,
  `MREEF(order=…)`, `PDIRK44`, adaptive-order Radau) is runtime fields.
  `alg_order(alg)` is allowed to be a runtime function of the algorithm value —
  `alg_order(alg::MREEF) = alg.order` is the precedent.
* **`OrdinaryDiffEqNewtonAlgorithm`, not the `Adaptive` variant.** Phase one is
  fixed-step. `PDIRK44` is the precedent for a non-adaptive Newton algorithm.
  Adaptivity is a phase-three item (§8).
* **Validation in `prepare_alg`**, following `MRAB`/`MRIGARK*`: reject unknown
  symbols, `M < 2` for Lobatto/Radau-left, `:lastnode` when the last node is not
  `1`, and diagonal-`QΔ` + `:Picard` combinations that make the sweep explicit
  where the user asked for implicit.

### Order

```julia
alg_order(alg) = min(sdc_iteration_order(alg), collocation_order(alg))
```

with

```
iteration order = (sweeper is second order ? 2 : 1) + (K - 1) + (step_update == :quadrature)
collocation order = Legendre:  Gauss 2M, Radau 2M-1, Lobatto 2M-2
                    Equidist.: Gauss/Lobatto M + (M mod 2), Radau M
```

This is `qmat.utils.sdc.getOrderSDC` minus its ~143 hard-coded bonus cases (see
`sdc-notes/04-verification.md` for how the two were compared).

## 3. Coefficients: `Q`, weights, `QΔ`

All depend only on `(M, node_type, quad_type, sweeper)` and the element type —
**never on `dt`, `t`, `u` or `p`** — so they are computed exactly once, in
`alg_cache`, and stored in an immutable `SDCTableau`.

```julia
struct SDCTableau{T}
    nodes::Vector{T}     # τ ∈ [0,1], increasing
    weights::Vector{T}   # ∫₀¹ ℓⱼ
    Q::Matrix{T}         # Q[m,j] = ∫₀^{τₘ} ℓⱼ
    QΔ::Matrix{T}        # the sweeper
end
```

`QΔ` is stored as a **full `M×M` matrix even when it is diagonal.** A diagonal
sweeper is then just a matrix whose strictly-lower part is zero, and the sweep
loop's strict-lower accumulation is skipped by an `iszero` test rather than by a
different code path. This is the single most important phase-two decision: it
means adding a diagonal sweeper requires *no* change to `perform_step!`.

### Nodes

Computed natively rather than via `FastGaussQuadrature.jl` (trade-off recorded
in `sdc-notes/04-verification.md`). Nodes on `[-1,1]`, then mapped to `[0,1]`:

* `:Gauss` — roots of the Jacobi polynomial `P^{(0,0)}_M` (Legendre)
* `:RadauLeft` — `{-1}` ∪ roots of `P^{(0,1)}_{M-1}`
* `:RadauRight` — roots of `P^{(1,0)}_{M-1}` ∪ `{1}`
* `:Lobatto` — `{-1}` ∪ roots of `P^{(1,1)}_{M-2}` ∪ `{1}`

Interior roots are found by grid-bracketing plus bisection, which is generic in
the element type (works in `BigFloat`) and needs no derivative. `M` is small, so
the cost is irrelevant. Equidistant nodes are closed form, matching `qmat`'s
`EQUID` convention exactly.

One trap, recorded because it cost real time: odd-degree Legendre polynomials
have a root at exactly `x = 0`, the bracketing grid is symmetric so it lands
there, and `P₃(0)` evaluates to `-0.0` — so `signbit` reports a sign change
across a bracket whose endpoint *is* the root. `BigFloat` bisection towards an
exact zero then never terminates, because there is no smallest normal and the
midpoint halves through the whole exponent range. It hangs rather than erroring.
The root finder detects exact roots on the grid, and the bisection is bounded.

### `Q` and weights

`Q[m,j] = ∫₀^{τ_m} ℓ_j(s) ds` with `ℓ_j` the Lagrange basis on `τ`. Computed as
`Q = C · V⁻¹` with `V[i,j] = τ_j^{i-1}` and `C[m,i] = τ_m^i / i`, **evaluated in
`BigFloat` and converted down**. The monomial Vandermonde is ill-conditioned
(`cond ≈ 10⁷` at `M = 8`); doing the solve at extended precision costs
microseconds once per `alg_cache` and removes the conditioning question
entirely.

### Sweepers

With `δ₁ = τ₁`, `δ_m = τ_m − τ_{m-1}`:

| symbol | `QΔ` | shape |
|---|---|---|
| `:BE` | `QΔ[i,j] = δ_j`, `j ≤ i` | lower triangular, `Σ_j QΔ[i,j] = τ_i` |
| `:FE` | `QΔ[i,j] = δ_{j+1}`, `j < i` | strictly lower ⇒ explicit sweep |
| `:Trapezoid` | `(QΔ_BE + QΔ_FE)/2` | lower triangular |
| `:LU` | `Uᵀ` where `Qᵀ = LU` | lower triangular (Weiser's LU trick) |
| `:Picard` | `0` | plain Picard iteration |
| `:BEpar` | `diag(τ)` | **diagonal** — implicit Euler from `t₀` to each node |
| `:MIN_SR_NS` | `diag(τ)/M` | **diagonal** — Čaklović et al. 2024 |

`:BEpar` and `:MIN_SR_NS` are diagonal and therefore parallel-ready; they run
*serially* in phase one. Including them now is deliberate: it exercises the
diagonal code path under the phase-one verification gate, so phase two only has
to add threading, not correctness.

## 4. Cache

```julia
@cache mutable struct SDCCache{uType, rateType, N, TabType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType             # right-hand side being assembled for the current node
    ubuf::uType            # scratch for explicit nodes
    k::rateType            # scratch for f evaluations
    z::Vector{uType}       # dt·f at the nodes, sweep k
    znew::Vector{uType}    # dt·f at the nodes, sweep k+1
    nlsolvers::Vector{N}   # one per node
    tab::TabType
end
```

Two decisions worth stating:

**(a) The cache stores `dt·f` at the nodes, not `u` at the nodes.** The sweep
right-hand side and the quadrature step update are both linear combinations of
`dt·f`, and `nlsolve!` already returns `z ≈ dt·f(u_m)` as its solution variable.
So node values never need to be materialised or stored, saving `M` state vectors
and `M` divisions per sweep. Node values are reconstructed only where needed
(`u_m = tmp + γ·z`), which the nonlinear solver does internally anyway.

**(b) One `NLSolver` per node, not one shared solver.** This is forced, not an
optimisation — see `sdc-notes/02-nonlinear-solver-reuse.md`. Each node's implicit
solve has its own `γ_m = QΔ[m,m]`, hence its own `W = M/(γ_m dt) − J`. A single
shared solver either silently reuses a stale `W` (because `do_newJW`'s
`isfreshJ` branch declares `smallstepchange = true` once `J` is current at this
`t`) or must be forced to refactorise on every node of every sweep. `M` solvers
give each node a private `W` that is *intended* to be built once per step and
reused across all `K` sweeps. `PDIRK44` does exactly this with a `Vector` of
solvers.

Cost of (b): `M` copies of `J` and `W`.

**Revision, after measuring.** The reuse across sweeps does not currently happen,
for a reason outside SDC: `do_newJW`
(`lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl:534`) contains
`!integrator.opts.adaptive && return true, true`, so in fixed-step mode this
library rebuilds `J` and refactorises `W` on *every* nonlinear solve. Measured:
`M = 4`, `K = 4`, 16 steps gives `njacs = nw = 256 = 16 × 4 × 4`. The per-node
solver design is still correct and is still what makes the `W` right and the
phase-two threading safe — but the efficiency payoff only arrives once SDC is
adaptive. That moves adaptivity from "nice to have" to the top of §8.

## 5. `perform_step!`

```
u⁰_m = uprev,  z⁰_m = dt·f(uprev, t + τ_m dt)          # COPY initialisation
for k = 1..K
    for m = 1..M
        tmp = uprev + Σ_{j=1..M} (Q[m,j] − QΔ[m,j])·z_j
                    + Σ_{j<m}     QΔ[m,j]·znew_j
        if QΔ[m,m] == 0
            u_m = tmp;  znew_m = dt·f(u_m, t + τ_m dt)
        else
            solve  u_m = tmp + QΔ[m,m]·znew_m,  znew_m = dt·f(u_m, t + τ_m dt)
        end
    end
    swap(z, znew)
end
u = uprev + Σ_m weights[m]·z_m          # :quadrature
u = u_M                                 # :lastnode
```

This is the node-wise form from the qmint non-linear SDC tutorial, rearranged so
that `Σ_j (Q − QΔ)[m,j] z_j` is one pass and the `dt` factors are absorbed into
`z`. Equivalent to `u^{k+1} − Δt QΔ f^{k+1} = u₀ + Δt (Q − QΔ) f^k`.

`COPY` initialisation (`u⁰_m = u_n` at every node) is chosen because it is what
`qmat`'s order predictor assumes and what Čaklović et al. §3 use, so the
verification gate compares like with like.

Mapping onto the existing nonlinear solver — the whole point of §4(b):

| SDC quantity | `NLSolver` field |
|---|---|
| `tmp` (assembled RHS) | `nlsolver.tmp` |
| `QΔ[m,m]` | `nlsolver.γ` |
| `τ_m` | `nlsolver.c` |
| `dt·f(u_m)` | `nlsolver.z` (the returned solution) |
| `u_m = tmp + γ z` | `nlsolver`'s internal `ustep` |

because `nlsolve!` solves exactly `dt·f(tmp + γ·z, p, t + c·dt) = z`.

## 6. Where a diagonal `QΔ` slots in later

Nothing in §5 has to change. With `QΔ` diagonal the term `Σ_{j<m} QΔ[m,j]·znew_j`
is empty, so the body of the `m` loop depends only on `z` (sweep `k`) and never
on `znew` (sweep `k+1`). The `m` loop is then embarrassingly parallel.

Phase two is therefore:

1. add a `threading` field to `SDC` (`Sequential()` / `BaseThreads()` /
   `PolyesterThreads()`, using `OrdinaryDiffEqCore`'s existing `@threaded`
   macro — `PDIRK44` is the precedent);
2. wrap the `m` loop in `@threaded alg.threading`;
3. give each thread its own `tmp` scratch (currently one shared `tmp`) —
   promote `tmp` to `Vector{uType}` of length `M`;
4. reject non-diagonal `QΔ` when threading is on, in `prepare_alg`.

The `nlsolvers::Vector` is already per-node, which is what makes (2) safe.

**Phase-one constraints this imposes — things phase one must NOT do:**

* must not share one `NLSolver` across nodes (would be a data race and a `W`
  correctness bug);
* must not accumulate the sweep right-hand side into a single shared buffer
  across nodes in a way that carries node-to-node state (hence `tmp` is fully
  rebuilt per node, never incrementally updated from node `m-1`);
* must not write `znew` in place over `z` (the `k`-th iterate must stay intact
  for all nodes until the sweep ends) — hence the explicit swap;
* must not special-case "lower-triangular" in the algebra, e.g. by looping
  `j = 1:m` over `Q` and `j = 1:m-1` over `QΔ` and assuming a Gauss–Seidel
  structure.

All four are satisfied by the phase-one code as written.

Open design question for phase two, to take to Fynn/Thibaut: `MIN-SR-FLEX`
changes `QΔ` every sweep, so `γ_m` changes every sweep and `W` must be rebuilt
`M` times per sweep instead of `M` times per step. That is a real cost for large
systems and it is not obvious it pays for itself outside the stiff limit.
Mechanically it needs a `QΔ::Vector{Matrix{T}}` in the tableau and a forced
`update_W!(..., newJW = (false, true))` before each solve.

## 7. IMEX SDC (Minion 2003) — design

The split form is `u' = f₁(u,t) + f₂(u,t)` (SciML convention: `f₁` stiff/implicit,
`f₂` non-stiff/explicit), with two preconditioners: an implicit `QΔ^I` (lower
triangular, non-zero diagonal) and an explicit `QΔ^E` (strictly lower triangular).
The sweep becomes

```
u^{k+1} − Δt QΔ^I f₁^{k+1} − Δt QΔ^E f₂^{k+1}
      = u₀ + Δt Q (f₁^k + f₂^k) − Δt QΔ^I f₁^k − Δt QΔ^E f₂^k
```

so node `m` still solves `u_m = rhs_m + Δt QΔ^I[m,m] f₁(u_m)` — the *same* shape
as phase one, with `f₁` in place of `f`, because `QΔ^E` is strictly lower
triangular and contributes only known quantities.

**The existing split plumbing gives most of this for free.**
`OrdinaryDiffEqCore.nlsolve_f` (`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:969`)
is

```julia
function nlsolve_f(f, alg::OrdinaryDiffEqAlgorithm)
    return f isa SplitFunction && issplit(alg) ? f.f1 : f
end
```

and it is what `compute_step!` calls to get the function the Newton iteration
sees (`newton.jl:355`, `newton.jl:405`). So declaring `issplit(::SDC) = true`
makes every node solve implicit in `f₁` only, with `W = M/(γdt) − ∂f₁/∂u`,
without touching the nonlinear solver at all. `KenCarp3/4/5` and the IMEX SDIRK
schemes already rely on this exact mechanism.

What IMEX SDC still needs:

1. a second cache array `z₂` for `dt·f₂` at the nodes (the explicit rates);
2. a second preconditioner field on the algorithm (`sweeper_explicit`, default
   `:FE`);
3. the right-hand-side assembly to use `Q` against `z₁ + z₂` but `QΔ^I` against
   `z₁` and `QΔ^E` against `z₂`;
4. `nf2` statistics bookkeeping for the explicit evaluations.

None of that changes the phase-one structure — it widens `z` from one array of
rates to two. **Recorded consequence for phase one:** the sweep right-hand side
is assembled through a small helper that takes the rate arrays as arguments
rather than reading `cache.z` directly, so the IMEX version can pass two.

## 8. Known costs and deferred work

Ordered by how much they matter.

* **Adaptivity, which is also the fix for the `W` cost.** Fixed-step mode forces
  `M·K` Jacobian evaluations and `W` factorisations per step (§4, revision).
  Making SDC adaptive both removes that — `isfreshJ` then lets each node's `W`
  survive all `K` sweeps, taking the count to `M` per step — and makes the solver
  usable. The natural estimate is the collocation residual
  `‖u_n + Δt Q F(u^K) − u^K‖`, which SDC computes almost for free and which is the
  standard SDC stopping criterion; an embedded pair from sweeps `K−1` and `K` is
  the alternative. Baumann–Götschel–Lunet–Ruprecht–Speck (*Numerical Algorithms*,
  2024) is the reference for adaptive step selection in SDC.
* **`M` Jacobian evaluations per step, even once adaptive.** Each of the `M`
  `NLSolver`s owns its `J` and evaluates it independently. For a large system
  that is `M×` the Jacobian work of a DIRK. The fix is to let node 1 compute `J`,
  then `copyto!` it into the other solvers' caches, set their `J_t`, and call
  `update_W!(nlsolver_m, integrator, cache, γ_m*dt, repeat_step, (false, true))`
  to force a `W`-only rebuild. Deferred to keep phase one small and auditable; it
  is a pure optimisation, not a correctness fix.
* **Memory is `M×(J + W)`.** Unavoidable if each node keeps a factorisation;
  the alternative (one `W`, refactorised per node per sweep) trades `K×` more
  factorisations for `M×` less memory. Worth a switch later.
* **No dense output.** Falls back to the library default; the collocation
  polynomial through the node values would be the natural (and free) interpolant.
* **`M ≥ 1` only in the sense tested.** `M = 1` degenerates (Gauss → implicit
  midpoint, Radau-right → implicit Euler); allowed but not part of the gate.

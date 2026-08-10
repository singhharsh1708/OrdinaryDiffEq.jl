# Spectral Deferred Correction — what it is, in enough depth to argue about

Everything below is either derived here, taken from a source I read directly (cited
with the exact place), or measured by me and labelled as such. Anything I could not
confirm is called out.

---

## 1. The collocation formulation

Take one step of `u' = f(u, t)` from `t_n` to `t_n + Δt` with `u(t_n) = u_n`.
Write it in Picard (integral) form, which is where all of SDC lives:

```
u(t) = u_n + ∫_{t_n}^{t} f(u(s), s) ds
```

Pick `M` quadrature nodes `0 ≤ τ_1 < … < τ_M ≤ 1`, and set `t_m = t_n + τ_m Δt`.
Write `u_m ≈ u(t_m)`. Now replace the integrand `f(u(s), s)` by the unique
polynomial of degree `M-1` interpolating the `M` values `f(u_j, t_j)` at the
nodes. Integrating that polynomial from `0` to `τ_m` gives

```
u_m = u_n + Δt Σ_{j=1}^{M} q_{mj} f(u_j, t_j),        q_{mj} = ∫_0^{τ_m} ℓ_j(s) ds
```

with `ℓ_j` the Lagrange basis on the nodes. That is the **collocation problem**.
`Q = (q_{mj})` is the **quadrature matrix**: row `m` is the quadrature rule for
the sub-interval `[0, τ_m]` expressed in the node values.

Three things to notice, because all of SDC follows from them.

**(a) It is one big non-linear system, not a sequence of stages.** In vector form,
with `**u** = (u_1, …, u_M)` and `F(**u**) = (f(u_1,t_1), …, f(u_M,t_M))`,

```
**u** − Δt Q F(**u**) = **u**_n            ("all-at-once" system)
```

`Q` is *dense*. Solving this directly means a Newton iteration on `M·N` unknowns
coupled through `Q ⊗ I` — that is exactly what a fully implicit Runge–Kutta
solver like `RadauIIA5` does.

**(b) It *is* an implicit Runge–Kutta method.** `A = Q`, `c = τ`, and the step
update `u_{n+1} = u_n + Δt Σ_m w_m f(u_m, t_m)` with `w_j = ∫_0^1 ℓ_j`. Collocation
on Legendre–Gauss nodes *is* Gauss IRK; on Radau-right nodes it *is* RadauIIA.
This matters for the whole project: SDC is not a new method, it is a **cheap
iterative solver for a known, very high order implicit RK method.**

**(c) The order is set by the nodes, not by the iteration.** Standard collocation
theory (Hairer–Nørsett–Wanner I, §II.7, and Hairer–Wanner II, §IV.5): the
collocation method has order equal to the order of the underlying quadrature
rule:

| nodes | order `p*` |
|---|---|
| Legendre–Gauss (`M` interior) | `2M` |
| Legendre–Radau (one endpoint) | `2M − 1` |
| Legendre–Lobatto (both endpoints) | `2M − 2` |
| equidistant, Radau-type | `M` |
| equidistant, Gauss/Lobatto-type | `M + (M mod 2)` |

The last two rows I took from `qmat.qcoeff.collocation.Collocation.order`, and
the even-`M` bump for the symmetric equidistant rules is a symmetry effect that
`qmat`'s own source comments on with "why ? the node symmetry I guess ...". I
verified the whole table numerically (see `04-verification.md`); I have not found
a textbook statement of the equidistant rows.

`p*` is the **ceiling**. No amount of sweeping gets past it.

---

## 2. The Picard view, and why plain Picard is useless

The obvious iteration on the all-at-once system is Picard:

```
**u**^{k+1} = **u**_n + Δt Q F(**u**^k)
```

For the Dahlquist test problem `f = λu`, write `z = Δt λ`. The error against the
exact collocation solution obeys `e^{k+1} = z Q e^k`, so the iteration matrix is
`zQ` and Picard converges iff `|z| ρ(Q) < 1`.

Two consequences:

* Each iteration multiplies the error by `O(Δt)`, so **each iteration buys one
  order** — that part is fine and it is the seed of the whole idea.
* The iteration only converges for `Δt` small relative to the Lipschitz constant
  of `f`. For a stiff problem — which is exactly where you want an implicit
  method — `|z|` is enormous and Picard **diverges**. And even in the non-stiff
  case, the step size is bounded by the *iteration*, not by accuracy, which
  throws away the reason for using a high-order method.

So Picard has the right structure and the wrong convergence. Fix the convergence
without losing the structure, and you have SDC.

---

## 3. The sweep: what `Q_Δ` is and why it works

Precondition the Picard iteration. Choose `Q_Δ ≈ Q` that is **lower triangular**
and solve

```
(I − Δt Q_Δ F)(**u**^{k+1} − **u**^k) = **u**_n + Δt Q F(**u**^k) − **u**^k
```

The right-hand side is the **residual** of the collocation problem at iterate `k`.
Rearranged (this is the form you implement, and the form in the qmint tutorial):

```
**u**^{k+1} − Δt Q_Δ F(**u**^{k+1}) = **u**_n + Δt (Q − Q_Δ) F(**u**^k)      (†)
```

The fixed point of (†) is the solution of the collocation problem — `Q_Δ` cancels
completely at convergence — so **`Q_Δ` changes only the rate, never the answer.**
That is the single most useful fact about the method, and I turned it into a test
(see the negative controls in `04-verification.md`: perturbing `Q` destroys the
order, perturbing `Q_Δ` does not).

Because `Q_Δ` is lower triangular, (†) does not have to be solved all at once. It
unrolls node by node:

```
u_m^{k+1} = rhs_m + Δt q^Δ_{mm} f(u_m^{k+1}, t_m)
rhs_m = u_n + Δt Σ_{j=1}^{M} (q_{mj} − q^Δ_{mj}) f_j^k + Δt Σ_{j<m} q^Δ_{mj} f_j^{k+1}
```

Each node is a single `N`-dimensional implicit solve of exactly the shape a DIRK
stage solves: `u = known + γ Δt f(u)`. One sweep therefore costs about what one
`M`-stage DIRK step costs. If `Q_Δ` is *strictly* lower triangular there is no
solve at all and the sweep is explicit.

### The iteration matrix

For Dahlquist, subtracting the collocation solution from (†) gives

```
e^{k+1} = K(z) e^k,     K(z) = z (I − z Q_Δ)^{-1} (Q − Q_Δ)
```

(Čaklović, Lunet, Götschel & Ruprecht, arXiv:2403.18641, eq. 14 — I read this
directly.) Its two limits are what everybody designs against:

```
K_NS = lim_{|z|→0} K(z)/z = Q − Q_Δ           (non-stiff limit, eq. 15)
K_S  = lim_{|z|→∞} K(z)   = I − Q_Δ^{-1} Q    (stiff limit,     eq. 16)
```

`ρ(K_NS)` controls convergence for small `Δt`, `ρ(K_S)` for large `Δt`. Designing
a good `Q_Δ` means driving one or both of these down — ideally to **zero**, which
for a nilpotent matrix means the iteration terminates exactly after `M` sweeps
rather than merely converging.

### The classical sweepers

With `δ_1 = τ_1` and `δ_m = τ_m − τ_{m-1}`:

* **Implicit ("backward") Euler between the nodes**, `q^Δ_{mj} = δ_j` for `j ≤ m`.
  Row sums are `τ_m`, so it is a consistent approximation of `Q`. This is the
  original Dutt–Greengard–Rokhlin sweeper.
* **Explicit ("forward") Euler between the nodes**, `q^Δ_{mj} = δ_{j+1}` for `j < m`.
  Strictly lower triangular, so the sweep is explicit.
* **Trapezoid / Crank–Nicolson**, the average of the two.
* **LU trick** (Weiser, BIT 55, 2015): factor `Qᵀ = LU` and take `Q_Δ = Uᵀ`. This
  is not a time-stepping scheme at all — it is chosen so that `K_S` becomes
  *nilpotent* (`I − Q_Δ^{-1}Q` is strictly triangular by construction), which is
  why it converges so much faster than Euler sweeps on stiff problems. This is
  the single best-value change you can make to a serial SDC code.

### Why it is called "deferred correction"

Historically the sweep was written as a low-order (Euler) integration of the
*error* equation `δ' = f(u^k + δ) − f(u^k) + residual'`, i.e. you defer the
correction of the error to a cheap solver. The `Q_Δ` form above is algebraically
the same thing and is how modern implementations (pySDC, qmat) are written. If a
specialist says "sweeper" they mean `Q_Δ`.

---

## 4. Order behaviour: how many sweeps, and the ceiling

This is where I would be careful in conversation, because the clean folklore
statement ("one order per sweep") is true but under-specified, and the exact
count depends on four things: the sweeper, the node set, the step update, and
`K`.

**What is proved:**

* **One order per sweep for Euler sweepers.** Dutt, Greengard & Rokhlin,
  *BIT* 40 (2000) 241–266, Theorem 4.1 — for explicit and implicit Euler sweeps.
  (Cited as such in Čaklović et al. §3.1, which is where I checked the pointer.)
* **One order per sweep for higher-order correctors.** Christlieb, Ong & Qiu,
  *CAMCoS* 4 (2009) 27–56, Theorem 3.8.
* **For a diagonal `Q_Δ`, the order after `K` sweeps is exactly `min(K, p*)`.**
  van der Houwen, Sommeijer & Couzy, *Math. Comp.* 58 (1992) 135–159, Theorem 2.1
  — again via Čaklović et al. §3.1, which states the theorem in this form.

**What is not proved, and I should not claim:**

* Sweepers of order higher than one are **not** guaranteed to give more than one
  order per sweep, except on equidistant nodes (Čaklović et al. §3.1, explicitly).
* For LU-SDC there is numerical evidence of one order per sweep and, as of that
  paper, *no proof*.
* `MIN-SR-NS` empirically gains **more** than one order per sweep and the authors
  say they "do not yet have a theoretical explanation" (§3.1, on Figure 2). I
  reproduced this: `M = 3` Radau-right, `K = 2`, quadrature update gives a
  measured order of 4.006 where the naive count predicts 3.

**The working rule I use, and where it comes from.** `qmat`'s
`getOrderSDC` (`qmat/utils/sdc.py`, the function at line 142, which is the one
Thibaut pointed at) computes

```
order = 0
if K > 0:  order += (sweeper is 2nd order ? 2 : 1) + (K − 1)
if step update is the quadrature rule: order += 1
order = min(order, p*)
```

and then applies about 143 hard-coded bonus cases for particular
`(sweeper, K, M, node type, quad type)` combinations. So: the base rule is
"one order per sweep, plus one if you finish with the quadrature rule rather than
taking the last node value, capped at the collocation order", and the bonus table
is the honest admission that a general theorem does not exist. I implemented the
base rule and verified it against `qmat`'s predictor on 480 combinations; they
agree on 466 of them, and all 14 disagreements are `qmat` bonus cases where my
prediction is a *lower* bound (details in `04-verification.md`).

**The step update matters and is easy to miss.** Two choices:

* `u_{n+1} = u_n + Δt Σ_m w_m f(u_m^K, t_m)` — the quadrature rule. Works for any
  node set, and buys one extra order.
* `u_{n+1} = u_M^K` — the last node value. Only valid when `τ_M = 1`
  (Radau-right, Lobatto).

The quadrature update is strictly better on order and costs nothing extra (the
`f` values are already in hand). `qmat` tests both, so I support both.

**The ceiling in practice.** With Legendre–Radau-right and `M = 4` you can reach
order 7, but you need `K = 6` sweeps with the quadrature update — and each sweep
is 4 implicit solves. So a 7th-order SDC step is 24 implicit solves. That number
is the thing to keep in mind when anyone claims SDC is competitive.

---

## 5. What "parallel across the nodes" changes

Look again at the node-wise sweep:

```
rhs_m = u_n + Δt Σ_{j=1}^{M} (q_{mj} − q^Δ_{mj}) f_j^k + Δt Σ_{j<m} q^Δ_{mj} f_j^{k+1}
```

The **only** node-to-node coupling within a sweep is the second sum, and it is
weighted by the strictly-lower part of `Q_Δ`. Make `Q_Δ` **diagonal** and it
vanishes: every node's implicit solve depends only on the previous sweep's
values, so all `M` solves are independent and can run on `M` threads. This is
Speck, *Comput. Visual. Sci.* 19 (2018) 75–83 ("parallelisation across the
method").

The price is convergence rate. A lower-triangular `Q_Δ` is a Gauss–Seidel-like
preconditioner; a diagonal one is Jacobi-like, and it is a worse approximation of
the dense `Q`. Concretely, from Čaklović et al.:

* The naive diagonal choices — `Q_Δ = diag(Q)` or `Q_Δ = diag(τ)` (an implicit
  Euler step from `t_n` straight to each node, "IEpar"/`BEpar`) — "result in slow
  convergence of the SDC iteration, making it inefficient" (§2.1).
* **`MIN-SR-NS`**: `Q_Δ = diag(τ)/M`. Theorem 2.8 proves `(Q − Q_Δ)^M = 0`, i.e.
  `ρ(K_NS) = 0` — the non-stiff iteration matrix is *nilpotent*, so the iteration
  terminates exactly in `M` sweeps in the non-stiff limit. Optimal for non-stiff
  problems, **not A-stable** (§3.2), which the authors call unsurprising since it
  is optimised for the `z → 0` limit.
* **`MIN-SR-S`**: diagonal coefficients found by solving
  `det[(1−t)I + t Q_Δ^{-1} Q] − 1 = 0` at `t = τ_1, …, τ_M` (Definition 2.11),
  i.e. making `K_S` nilpotent. No proof that a solution exists for every `M`;
  found numerically with MINPACK's `hybrd`, seeded incrementally by fitting a
  power law `α t^β` through the previous `M`'s coefficients. At `M = 4`
  Radau-right this gives `ρ(K_S) = 2.4·10⁻⁴` (compare `0.42` for Speck's 2018
  `MIN` coefficients and `0.025` for van der Houwen–Sommeijer's, from their
  Table).
* **`MIN-SR-FLEX`**: change the preconditioner every sweep, `Q_Δ^{(k)} = diag(τ)/k`
  for `k = 1, …, M`. Theorem 2.13 proves the product of the stiff-limit iteration
  matrices `(I − Q_{Δ,M}^{-1}Q)⋯(I − Q_{Δ,1}^{-1}Q)` is exactly zero. For `k > M`
  the paper falls back to `MIN-SR-S`.

The headline claim of that paper is that `MIN-SR-S`/`MIN-SR-FLEX` recover
"stability domains and convergence order very similar to those of well
established serial SDC variants" — i.e. the diagonal penalty can be bought back
by choosing the coefficients well, rather than by taking more sweeps.

**Threading model in Julia.** No new machinery is needed. `OrdinaryDiffEqCore`
already ships `Sequential() / BaseThreads() / PolyesterThreads()` and a
`@threaded opt for …` macro (`lib/OrdinaryDiffEqCore/src/misc_utils.jl:143`), and
`OrdinaryDiffEqPDIRK` already uses it to run a 2-processor parallel DIRK, with a
**vector of `NLSolver`s, one per parallel stage**, each carrying its own `γ`,
Jacobian and `W`. That is structurally identical to parallel-across-the-nodes
SDC with `M` nodes. So phase two is: add a `threading` field, wrap the node loop,
give each node its own scratch buffer. See `03-design.md` §6 for the exact list
and for what phase one had to avoid so that this stays a small change.

For the shared-memory results at scale: Freese, Götschel, Lunet, Ruprecht &
Schreiber, *Int. J. High Perform. Comput. Appl.*, 2025 (arXiv:2403.20135) —
OpenMP implementations of parallel SDC inside ICON-O and SWEET, on the
shallow-water equations, with an IMEX splitting that integrates fast modes
implicitly and slow modes explicitly. I read the abstract and the metadata
directly; I did **not** get clean numbers out of the PDF for the speedups, so I
am not quoting any.

---

## 6. Where SDC actually competes

Fynn's expectation is IMEX and high order. Here is what the literature I read
does and does not support.

**Supported.**

* **IMEX.** Minion, *Comm. Math. Sci.* 1 (2003) 471–500, is the whole point:
  split `f = f_I + f_E`, use a lower-triangular `Q_Δ^I` for the stiff part and a
  strictly-lower-triangular `Q_Δ^E` for the non-stiff part, and you get an
  arbitrary-order IMEX method out of two first-order building blocks. Deriving a
  classical additive Runge–Kutta pair of order 5 means solving a large coupled
  order-condition system; with SDC you set `K = 5`. This is a real and specific
  advantage, and it is the strongest argument for the method.
* **High order on demand.** Order is a runtime parameter (`M`, `K`), not a
  tableau you have to find. Getting order 8+ from classical RK means Feagin-style
  tableaus with hundreds of coefficients; here it is `M = 5` Gauss and `K = 9`.
* **Cheap, meaningful error estimate.** The collocation residual
  `‖u_n + Δt Q F(u^K) − u^K‖` is available for free after every sweep and is the
  standard SDC stopping criterion. Adaptivity built on it: Baumann, Götschel,
  Lunet, Ruprecht & Speck, *Numerical Algorithms*, 2024.
* **As a building block for parallel-in-time.** MLSDC and PFASST are built on
  SDC sweeps. Gander & Lunet, *Time Parallel Time Integration*, SIAM, 2024, is
  the reference text (Thibaut is the second author).

**Not supported, or at least not by anything I read.**

* I found **no** claim that vanilla serial SDC beats a well-tuned explicit RK or
  ESDIRK on a general problem's work-precision diagram. The arithmetic is against
  it: `M·K` implicit solves per step versus `s` for an `s`-stage DIRK of the same
  order. Čaklović et al. §3 frames fixed-`K` SDC as an RK method precisely so it
  *can* be compared against classical RKM, which is an acknowledgement that the
  comparison is not a foregone conclusion.
* The parallel-across-the-nodes speedup is bounded by `M` (typically 3–5) and is
  a *shared-memory, small-scale* parallelism. It is a way to spend idle cores, not
  a way to beat a serial method by an order of magnitude.

**My reading of where this project should aim its benchmarks**, for discussion
with Fynn and Thibaut rather than as a conclusion: IMEX problems with a genuinely
stiff linear part and a cheap explicit part (advection–diffusion, shallow water,
the ICON-O/SWEET setting), and high-accuracy requests (`reltol ≤ 1e-10`) where the
collocation ceiling is high and classical high-order tableaus run out. Not
general-purpose non-stiff ODEs, where `Tsit5`/`Vern9` will win.

# Can SDC sweeps reuse OrdinaryDiffEq's nonlinear solver and `W` machinery?

**Short answer: yes, completely, with no changes to `OrdinaryDiffEqNonlinearSolve`
— but only if SDC builds one `NLSolver` per node instead of sharing one. Sharing
is not merely slower, it is wrong.** And there is a separate, real efficiency
problem that is *not* SDC's fault and that I could not design around: in
non-adaptive mode this library rebuilds the Jacobian and refactorises `W` on
every nonlinear solve, so phase-one SDC pays `M·K` Jacobians and factorisations
per step.

All line numbers are against the tree this was written on.

---

## 1. The interfaces line up exactly

`nlsolve!` documents the equation it solves
(`lib/OrdinaryDiffEqNonlinearSolve/src/nlsolve.jl:3`):

```
dt⋅f(innertmp + γ⋅z, p, t + c⋅dt) + outertmp = z
```

and for the `DIRK` method (the default) the residual is built as
(`newton.jl:507` and `newton.jl:548`):

```julia
function compute_ustep!(ustep, tmp, γ, z, method)
    if method === COEFFICIENT_MULTISTEP
        ustep = z
    else
        @.. ustep = tmp + γ * z          # u_stage = tmp + γ z
    end
    ...
    f(k, ustep, p, tstep)
    @.. ztmp = (dt * k - z) * invγdt     # residual: dt f(u_stage) - z
```

with, from `initialize!` (`newton.jl:10-11`):

```julia
cache.invγdt = inv(dt * nlsolver.γ)
cache.tstep  = integrator.t + nlsolver.c * dt
```

The SDC node solve is

```
u_m = rhs_m + Δt q^Δ_{mm} f(u_m, t_n + τ_m Δt)
```

so the mapping is one-to-one:

| SDC | `NLSolver` field |
|---|---|
| `rhs_m` | `nlsolver.tmp` |
| `q^Δ_{mm}` | `nlsolver.γ` |
| `τ_m` | `nlsolver.c` |
| `Δt f(u_m)` | `nlsolver.z`, the value `nlsolve!` returns |
| `u_m` | `tmp + γ z`, formed internally as `ustep` |

Nothing is missing, nothing has to be wrapped. `nlsolver.γ` and `nlsolver.c` are
plain mutable fields on `NLSolver` (`type.jl:204-213`), so they can be set per
node. The implementation does exactly this in
`lib/OrdinaryDiffEqSDC/src/sdc_perform_step.jl`, and the whole node solve is nine
lines.

There is a bonus: `nlsolve!` returns `z = Δt f(u_m)`, which is precisely the
quantity the next sweep's right-hand side and the quadrature step update need.
So SDC never has to store node *values* or divide by `dt` — the cache holds
`Δt f` at the nodes and nothing else.

## 2. Why one solver per node, and not one shared solver

Each node has its own `γ_m = q^Δ_{mm}`, hence its own
`W = M/(γ_m Δt) − J`. With a single shared solver you would set `nlsolver.γ`
before each node and rely on the library noticing that `W` is stale. It does not
reliably notice. `do_newJW`
(`lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl:519`) decides:

```julia
isfreshJ = isJcurrent(nlsolver, integrator) && !integrator.derivative_discontinuity
if isfreshJ
    jbad = false
    smallstepchange = true          # <-- asserted, not tested
else
    W_iγdt = inv(nlsolver.cache.W_γdt)
    iγdt   = inv(nlsolver.γ * integrator.dt)
    smallstepchange = abs(iγdt / W_iγdt - 1) <= get_new_W_γdt_cutoff(nlsolver)
    ...
end
wbad = (!smallstepchange) || (isfs && errorfail) || nlsolver.status === Divergence
```

and `isJcurrent(nlsolver, integrator) = integrator.t == nlsolver.cache.J_t`
(`OrdinaryDiffEqNonlinearSolve/src/utils.jl:45`). So once the Jacobian has been
evaluated at this `t` — which happens at the first node of the first sweep — the
`isfreshJ` branch declares `smallstepchange = true` *without comparing `γ` to the
`γ` the stored `W` was built with*. Every later node in the step would then be
solved with node 1's `W`. The Newton iteration would still converge (it is a
modified Newton, so a wrong `W` costs iterations, not correctness) — until it
does not, on a stiff problem, where the wrong `γ` is exactly the thing that makes
the iteration diverge.

The library's own answer to this is `OrdinaryDiffEqPDIRK`, which is a parallel
DIRK with two distinct `γ`s. It builds a **vector** of solvers
(`lib/OrdinaryDiffEqPDIRK/src/pdirk_caches.jl:52-80`) and assigns each its own
`γ` (`pdirk_perform_step.jl:16-23`):

```julia
@threaded alg.threading for i in 1:2
    nlsolver[i].z = zero(u)
    nlsolver[i].tmp = uprev
    nlsolver[i].γ = γs[i]
    nlsolver[i].c = cs[i]
    markfirststage!(nlsolver[i])
    k1[i] = nlsolve!(nlsolver[i], integrator, cache, repeat_step)
end
```

That is structurally the same thing SDC needs, and it is also the phase-two
threading pattern. Note the contrast with my own `MRIGARKIRK21a`/`ESDIRK34a`,
which share a single solver — they can, because
`_mrigark_impl_γ(tab) = tab.γ0[findfirst(!iszero, tab.γ0)]`
(`lib/OrdinaryDiffEqMultirate/src/multirate_caches.jl:179`) picks the *one* `γ`
those tableaux have. SDC is the other case.

Each `build_nlsolver` call allocates its own `J` and `W`
(`OrdinaryDiffEqNonlinearSolve/src/utils.jl:421`,
`J, W = build_J_W(alg, u, uprev, p, t, dt, f, jac_config, uEltypeNoUnits, Val(true))`),
so `M` solvers cost `M` copies of the Jacobian and of `W`. For SDC that is the
right trade: `M` is 3–5, and each node's `W` is then correct and, in principle,
reusable across all `K` sweeps of the step.

Nodes with `q^Δ_{mm} = 0` are explicit and get no solver at all. That is not an
edge case — it happens for `:FE` and `:Picard` at every node, and at the *first*
node of every sweeper whenever `τ_1 = 0` (Lobatto and Radau-left). The cache
carries a `solver_index` mapping nodes to solvers, with `0` meaning "explicit".

## 3. The problem I could not design around

Measured on `u' = [-u₂, u₁]`, `[0, 2π]`, 16 fixed steps:

| solver | steps | `njacs` | `nw` | `nsolve` |
|---|---|---|---|---|
| `ImplicitEuler`, `adaptive = false` | 16 | 16 | 16 | 32 |
| `SDC(num_nodes = 4, num_sweeps = 4)`, `adaptive = false` | 16 | **256** | **256** | 512 |

`256 = 16 steps × 4 nodes × 4 sweeps`: **one Jacobian evaluation and one `W`
factorisation per nonlinear solve.** The intended behaviour — build `W` once per
node per step and reuse it across the `K` sweeps — does not happen.

The cause is not SDC. It is `do_newJW`, `derivative_utils.jl:534`:

```julia
!integrator.opts.adaptive && return true, true # Not adaptive will always refactorize
```

In non-adaptive mode this library unconditionally rebuilds `J` and `W` on every
solve. `ImplicitEuler` pays the same policy but only has one solve per step, so
it is invisible there. Phase-one SDC is fixed-step by design, so it pays it `M·K`
times per step.

Two things follow.

* The per-node-solver design is still **correct and still the right structure**:
  the `isfreshJ` reasoning above is what would bite once the policy stops forcing
  a rebuild, and it is what makes the phase-two threaded version safe.
* The fix is **to make SDC adaptive**, not to work around the nonlinear solver.
  Once `integrator.opts.adaptive` is true, the `isfreshJ` branch does the right
  thing for per-node solvers: node `m`'s `J_t` is set on the first sweep, and
  every later sweep at the same `t` finds `isfreshJ = true`, `wbad = false`, and
  reuses the factorisation. That takes the count from `M·K` down to `M` per step.
  This is now the top item on the phase-three list, and it is worth raising with
  Chris independently: any fixed-step implicit solve in this library refactorises
  more than it needs to.

There is a further optimisation, orthogonal to the above: the `M` solvers each
evaluate the same Jacobian `∂f/∂u(u_n, t_n)`. Node 1 could compute it and the
rest `copyto!` it and force a `W`-only rebuild via
`update_W!(nlsolver_m, integrator, cache, γ_m*dt, repeat_step, (false, true))`.
That would take `M` Jacobians per step down to 1. I did not do it in phase one —
it reaches further into `OrdinaryDiffEqDifferentiation`'s internals than I want
in a first cut, and it is a pure optimisation.

## 4. IMEX comes for free from the same machinery

`nlsolve_f` (`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:969`):

```julia
function nlsolve_f(f, alg::OrdinaryDiffEqAlgorithm)
    return f isa SplitFunction && issplit(alg) ? f.f1 : f
end
```

and it is what `compute_step!` calls to obtain the function the Newton iteration
differentiates and evaluates (`newton.jl:355` out-of-place, `newton.jl:405`
in-place), and what `build_nlsolver` uses to build the Jacobian wrapper
(`utils.jl:403`). So for IMEX SDC, declaring `issplit(::SDCIMEX) = true` on a
`SplitODEProblem` makes every node solve implicit in `f₁` alone, with
`W = M/(γΔt) − ∂f₁/∂u`, without touching the nonlinear solver. `KenCarp3/4/5`,
`CFNLIRK3` and the IMEX SDIRK schemes already work exactly this way
(`lib/OrdinaryDiffEqSDIRK/src/alg_utils.jl`, the `issplit` block).

What IMEX SDC still has to add is bookkeeping, not machinery: a second rate array
for `Δt f₂` at the nodes, a second (strictly lower triangular) preconditioner
`Q_Δ^E`, and the `nf2` statistics. See `03-design.md` §7.

## 5. Verdict

Reuse it. Do not build SDC-specific nonlinear solver infrastructure. The one
structural requirement is a vector of solvers rather than a single solver, which
is a five-line difference in `alg_cache` and is what `PDIRK44` already does. The
outstanding efficiency question is about the library's non-adaptive `W` policy
and about sharing one Jacobian across the node solvers — both are optimisations
on top of a correct design, and both are worth raising upstream rather than
routing around.

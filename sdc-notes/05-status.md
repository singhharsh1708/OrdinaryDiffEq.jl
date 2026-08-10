# Verified / inferred / open

Deliberately separated. The open list is the one to take to Fynn and Thibaut.

---

## Verified — I ran it, or I read it in the source

### Code

* `lib/OrdinaryDiffEqSDC` builds, precompiles, and passes **2211 assertions** in
  12 s (`Pkg.test()` from the sub-package, exit 0).
* Nodes, weights, `Q` and seven `Q_Δ` matrices match `qmat`'s to `atol = 1e-13`
  across 6 node configurations (Legendre and equidistant × Gauss, Radau-left,
  Radau-right, Lobatto) × 7 sweepers = **42 `Q_Δ` matrices**.
* **470 convergence-order combinations pass**, using `qmat`'s own step-size
  triples and tolerances. Measured orders are tabulated in `04-verification.md`.
* Negative controls behave as they must: perturbing `Q` by `10⁻³` drops the order
  from 3.87 to 2.12; perturbing a step weight drops it to 0.39; perturbing `Q_Δ`
  leaves it at 3.84–3.87. The last one is the control on the controls and it
  matches the theory — `Q_Δ` cancels at the fixed point.
* Order survives on non-linear problems (scalar Riccati out-of-place; a 2-D
  system in-place; both through the automatic-differentiation Jacobian path), and
  on a fully explicit sweep where no nonlinear solver is built at all.
* In-place and out-of-place paths agree to `rtol = 1e-10`.
* **The in-place `perform_step!` allocates 0 bytes per step** (`@allocated`,
  `M = 4`, `K = 4`).
* Coefficients are correct in `BigFloat`: `Q τ³ = τ⁴/4` to better than `10⁻⁴⁰`.

### Design

* The existing nonlinear solver is reusable as is. `nlsolve!` solves
  `dt f(tmp + γz, p, t + c·dt) = z`, which is *exactly* the SDC node equation
  under the mapping `tmp = rhs_m`, `γ = QΔ[m,m]`, `c = τ_m`. Code evidence in
  `02-nonlinear-solver-reuse.md` §1.
* One `NLSolver` per node is required, not an optimisation: `do_newJW`'s
  `isfreshJ` branch (`derivative_utils.jl:542-571`) asserts
  `smallstepchange = true` once the Jacobian is current at this `t`, without
  comparing `γ` against the `γ` the stored `W` was built with. `PDIRK44` already
  does per-stage solvers for the same reason.
* IMEX needs no nonlinear-solver work: `nlsolve_f`
  (`integrator_utils.jl:969`) already routes `SplitFunction` problems to `f.f1`
  when `issplit(alg)`, and `compute_step!` calls it.
* Phase two needs no restructuring of `perform_step!`. With a diagonal `Q_Δ` the
  strictly-lower accumulation is empty, the node loop becomes independent, and
  the change is a `threading` field plus `@threaded` plus per-node scratch. The
  threading machinery (`Sequential`/`BaseThreads`/`PolyesterThreads`, `@threaded`)
  already exists in `OrdinaryDiffEqCore`.
* **Fixed-step implicit solves in this library rebuild `J` and refactorise `W` on
  every nonlinear solve** — `do_newJW`, `derivative_utils.jl:534`. Measured:
  `njacs = nw = 256` for 16 steps at `M = 4`, `K = 4`. Not SDC-specific;
  `ImplicitEuler` pays it too, but only once per step.

### Literature (each checked, none from memory)

| claim | status |
|---|---|
| Dutt, Greengard, Rokhlin, *BIT* **40** (2000) 241–266, DOI `10.1023/A:1022338906936` | correct as given |
| Minion, *Comm. Math. Sci.* **1**(3) (2003) 471–500 | correct as given |
| Weiser, *BIT* **55**(4) (2015) 1219–1241, DOI `10.1007/s10543-014-0540-y` | correct; note the DOI is a 2014 online-first, which is why `qmat` cites it as "[Weiser, 2014]" |
| Speck, *Comput. Visual. Sci.* **19** (2018) 75–83, DOI `10.1007/s00791-018-0298-x`, arXiv:1703.08079 | correct as given |
| Gander & Lunet, *Time Parallel Time Integration*, SIAM, 2024, ISBN 978-1-611978-01-8 | correct as given |
| Freese, Götschel, Lunet, Ruprecht, Schreiber, 2025 | correct; published in *Int. J. High Perform. Comput. Appl.*, DOI `10.1177/10943420251400406`, preprint arXiv:2403.20135 |

Nothing in your list was wrong. One addition you will need and did not list:
**Čaklović, Lunet, Götschel & Ruprecht, *Improving efficiency of parallel across
the method spectral deferred corrections*, arXiv:2403.18641** — this is where the
`MIN-SR-NS`/`MIN-SR-S`/`MIN-SR-FLEX` diagonal coefficients and their theorems
live, and it is the paper phase two is actually built on. Speck 2018 introduces
the idea; this one makes it work.

### `qmat`

* The function Thibaut pointed at is `getOrderSDC` at `qmat/utils/sdc.py:142` —
  it is the order *predictor*, not the test. The test that consumes it is
  `tests/test_qdelta/test_timestepping.py::testSDCConvergenceQUADRATURE` and
  `::testSDCConvergenceLASTNODE`. Line 142 is still correct on `main`.
* `getOrderSDC` carries roughly 143 hard-coded bonus cases on top of the base
  rule. I implemented the base rule; over the 470 gated combinations, 457 agree
  exactly and 13 are bonus cases where mine is a strict lower bound.
* I reproduced `qmat`'s own test in Python before porting: 470 of 480
  combinations pass its tolerance, and the 10 that do not are exactly the ones
  its test file special-cases with hand-written `nSteps` overrides.

### Ecosystem

* No registered Julia package implements SDC (all 12 249 General registry names
  checked). Nothing SDC-related exists in this repository.
* Four unregistered Julia SDC repositories exist on GitHub; the only live one is
  `lisawim/SpectralDeferredCorrections.jl` (last touched 2025-04-17), described
  by its own README as a playground for learning Julia. Lisa Wimmer is a pySDC
  contributor, so this is inside the parallel-in-time community, not outside it.

---

## Inferred — reasonable, not proven here

* **The equidistant collocation orders** (`M`, or `M + (M mod 2)` for the
  symmetric rules) come from `qmat`'s `Collocation.order`, whose own source
  comments the even-`M` bump with "why ? the node symmetry I guess ...". I
  verified it numerically for `M = 2…4`; I have not found a reference for it.
* **`M` solvers will reuse their `W` across sweeps once SDC is adaptive.** This
  follows from reading `do_newJW` — `isfreshJ` is true after the first sweep at a
  given `t`, `errorfail` is false in normal operation, so `wbad` is false — but I
  have not measured it, because phase-one SDC has no adaptivity to turn on.
* **Sharing one solver across nodes would produce wrong `W`s in practice.** The
  code path is clear, but I did not build the broken version to watch it fail.
* **`:LU` should decisively beat `:BE` on stiff problems.** That is the point of
  Weiser's construction and it is what the stiff-limit iteration matrix says.
  Every convergence measurement here is on non-stiff problems, so I have not seen
  it.
* **The per-node Jacobian sharing optimisation** (node 1 computes `J`, others
  `copyto!` and force a `W`-only rebuild) will work. The API exists
  (`update_W!(..., newJW)`); I have not written it.

---

## Open — things I would rather ask than guess

**For Fynn and Thibaut:**

1. **Which step update should be the default?** `qmat` tests both; the quadrature
   rule buys an extra order for free and works for any node set, so I defaulted
   to it. Is there a reason the group's codes use the last-node value (a
   stiffly-accurate / L-stability argument I am missing)?
2. **Which sweepers do you actually want?** I have BE, FE, Trapezoid, LU, Picard,
   BEpar, MIN-SR-NS. Missing and deliberately deferred: MIN-SR-S (needs a
   non-linear solve for the coefficients, or the tabulated values from the
   paper), MIN-SR-FLEX (needs a per-sweep `Q_Δ`, which changes `γ` every sweep
   and therefore forces a `W` rebuild every sweep — is that worth it outside the
   stiff limit?), and the Chebyshev node families.
3. **Initialisation.** I use `COPY` (`u⁰_m = u_n` at every node), matching
   `qmat`'s order predictor and Čaklović et al. §3. pySDC has other options
   (spread with the initial `f` evaluation, and so on). Does the choice matter
   for the comparisons you have in mind?
4. **What is the right first benchmark problem?** Not for a work-precision study
   — for deciding whether the phase-one implementation is doing something
   sensible on a problem the group recognises. Something IMEX and stiff would let
   me test what the current gate cannot.
5. **`MIN-SR-NS` gains two orders per sweep in my measurements** (`M = 3`
   Radau-right: `K = 2` gives 4.01, `K = 3` gives 5.07). Čaklović et al. §3.1
   report the same and say there is no theoretical explanation. Has anything
   changed since?
6. **How much does the ~143-entry bonus table in `getOrderSDC` matter to you?**
   I treat it as an empirical patch and implement the base rule as a lower bound.
   If those cases are load-bearing for anything, I should port the table.

**For Chris / upstream:**

7. **`do_newJW`'s `!adaptive → always refactorize` policy.** Every fixed-step
   implicit solve in the library rebuilds `J` and `W` on every nonlinear solve.
   For a one-solve-per-step method that is nearly free; for SDC it is `M·K` per
   step. Is that policy deliberate (a conservative default for a mode nobody
   profiles) or is it worth a `γ`-aware check?
8. **Whether `SDC` should be adaptive in the first PR.** It is the fix for (7)
   and it is what makes the method usable, but it widens phase one considerably.

**Mine to resolve before this is PR-ready:**

9. No stiff-problem coverage in the test suite. Largest gap.
10. No AllocCheck/JET/Aqua QA suite (`OrdinaryDiffEqMultirate` has one).
11. Dense output is whatever the core falls back to; the collocation polynomial
    through the node values is the free and correct interpolant and is not
    implemented.
12. Not re-exported from the umbrella `OrdinaryDiffEq` — deliberate, per the
    standing rule that a sub-package is not re-exported until it is registered.
13. `sdc-notes/` is branch-local scratch. Delete it before opening any PR.

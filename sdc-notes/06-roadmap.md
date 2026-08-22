# SDC in OrdinaryDiffEq.jl — how big is this, and in what order

Written 2026-08-23, after phase one landed. Everything below is counted from
primary sources, not from memory: the Parallel-in-Time community bibliography
(`Parallel-in-Time/parallel-in-time.github.io`, `_bibliography/pint.bib` and
`_bibliography/other/sdc.bib`), the `qmat` and `pySDC` source trees, and the
papers named. Where I am guessing, it says so.

---

## 1. How big the literature actually is

| source | entries |
|---|---|
| PinT community `pint.bib` | 727 total, **31** on SDC / MLSDC / PFASST |
| PinT community `other/sdc.bib` (SDC-specific, stops ~2017) | **38** |
| union, deduplicated, roughly | **~55 papers** |

That is the whole field, not a sample. It is a small literature — about 25 years
of work by maybe six groups. It is tractable to read the core of it.

The arc, with the papers that actually matter for us:

**Foundations (2000–2010)**
- Dutt, Greengard & Rokhlin, *BIT* 40 (2000) 241–266 — the original.
- Minion, *Comm. Math. Sci.* 1 (2003) 471–500 — IMEX (semi-implicit) SDC.
- Bourlioux, Layton & Minion (2003), Layton & Minion (2004) — multi-implicit SDC.
- Huang, Jia & Minion, *JCP* 214 (2006) — accelerating convergence (Krylov).
- Layton (2005, 2008, 2009) — node choice, corrector choice, efficiency.
- Christlieb, Ong & Qiu, *CAMCoS* 4 (2009) — high-order correctors, the
  one-order-per-sweep theorem for general correctors.

**Better preconditioners (2014–2018)**
- Weiser, *BIT* 55 (2015) 1219–1241 — the LU trick. Single biggest serial win.
- Speck, *Comput. Visual. Sci.* 19 (2018) 75–83 — diagonal `QΔ`, parallel across
  the method.
- Speck (2016) — inexact SDC.

**Parallel across the method, current (2024–2026)**
- Čaklović, Lunet, Götschel & Ruprecht, *SIAM J. Sci. Comput.* **47** (2025)
  A430–A453, doi `10.1137/24M1649800` — `MIN-SR-NS`/`-S`/`-FLEX`. **Note: this is
  now published; my earlier notes cite the arXiv preprint 2403.18641 and are
  out of date.**
- Freese, Götschel, Lunet, Ruprecht & Schreiber, *IJHPCA* (2025),
  doi `10.1177/10943420251400406` — shared-memory OpenMP performance in ICON-O
  and SWEET.
- Baumann, Götschel, Lunet, Ruprecht & Speck, *Numerical Algorithms* (2024),
  doi `10.1007/s11075-024-01964-z` — adaptive step selection for SDC.
- **Bolten & Wimmer (2026), unpublished — parallel-across-the-method SDC for
  index-1 DAEs.** This one is worth flagging: Lisa Wimmer is also the author of
  the standalone Julia SDC prototype I found, and DAEs are exactly Harsh's
  background.
- Saupe, Götschel, Lunet, Ruprecht & Speck (2026), Springer — resilience against
  soft faults through SDC adaptivity.

**Multilevel and PFASST (2012–2021)** — out of scope, listed so the boundary is
explicit: Emmett & Minion (2012), Speck et al. *BIT* (2015) MLSDC, Minion et al.
*SISC* (2015), Bolten, Moser & Speck (2017, 2018), Schöbel & Speck (2020)
PFASST-ER, Benedusi et al. (2021).

**Reference text**: Gander & Lunet, *Time Parallel Time Integration*, SIAM, 2024.

---

## 2. What "an SDC method" means, and how many there are

This is the thing to get straight before estimating, because SDC does not come
as a list of tableaux. It factorises:

```
method = (node set) × (preconditioner QΔ) × (sweep count K) × (step update) × (problem splitting)
```

The first four are runtime parameters of one algorithm type. Only the last one —
how the right-hand side is split — forces a genuinely new solver type.

### Counted from qmat, which is the reference implementation for the coefficients

| axis | qmat has | we have |
|---|---|---|
| node distributions | 6 (`EQUID`, `LEGENDRE`, `CHEBY-1…4`) | **2** |
| quadrature types | 4 (`GAUSS`, `RADAU-LEFT`, `RADAU-RIGHT`, `LOBATTO`) | **4** ✓ |
| `QΔ` generators | **26** | **9** |

The 26 `QΔ` generators break down as:

- *time-stepping based* (6): `BE`, `FE`, `TRAP`, `BEPAR`, `TRAPAR`, `SOE`
- *algebraic* (7): `PIC`, `Exact`, `LU`, `LU2`, `QPar`, `GS`, `LDU`
- *diagonal / minimisation* (13): `MIN`, `MIN3`, `MIN_VDHS`, `MIN_SR_NS`,
  `MIN_SR_S`, `MIN_SR_FLEX`, `Jumper`, `FlexJumper`, `DNODES`–`DNODES5`

We have `BE`, `FE`, `Trapezoid`, `LU`, `Picard`, `BEpar`, `MIN_SR_NS`,
`MIN_SR_S`, `MIN_SR_FLEX`. Most of the missing 17 are cheap: `TRAPAR`, `LU2`,
`QPar`, `GS`, `LDU`, `Exact` and the five `DNODES` are one to three lines each.
The ones with real content are `SOE` (Lagrange derivative matrix), `MIN`
(Nelder–Mead spectral-radius fit), `MIN3` and `MIN_VDHS` (published tables), and
the two `Jumper` variants.

### Counted from pySDC, which is the reference implementation for the methods

Sweeper classes, which is the honest list of *method families*:

| pySDC sweeper | what it is | needs a new type here |
|---|---|---|
| `generic_implicit` | vanilla SDC | done |
| `generic_implicit_MPI` | parallel across the nodes | no — a `threading` field |
| `imex_1st_order` | IMEX SDC (Minion) | **yes** |
| `imex_1st_order_MPI` | parallel IMEX | no — same field |
| `imex_1st_order_mass` | IMEX with a mass matrix | no |
| `multi_implicit` | MISDC, two implicit terms | **yes** |
| `explicit` | fully explicit SDC | no — `FE`/`Picard` already give this |
| `verlet` | second-order problems | **yes**, and it is a `DynamicalODE` |
| `boris_2nd_order` | Boris-SDC, charged particles | **yes**, niche |
| `Runge_Kutta`, `Runge_Kutta_Nystrom` | RK dressed as a sweeper | no, not our concern |
| `Multistep` | multistep as a sweeper | no |
| `ParaDiagSweepers` | ParaDiag | separate family, out of scope |

So: **three solver types that matter to us** (vanilla, IMEX, multi-implicit),
plus two niche ones (Verlet/Boris) that are separate projects.

---

## 3. Where we are

Landed in PR #4208 and the two stacked branches:

- one algorithm type, `SDC`, fully parameterised
- 2 node distributions × 4 quadrature types × 9 sweepers × 2 step updates
- adaptive step size control, with the Jacobian-reuse win measured (188× fewer
  Jacobian evaluations than fixed-step)
- 2322 test assertions, including direct verification of Theorem 2.8,
  Definition 2.11 and Theorem 2.13 from Čaklović et al.
- coefficients checked against qmat to 1e-13

Honest position: **phase one of three is done, and the coefficient groundwork for
phase two is done.** In terms of the whole arc that Fynn described, this is
maybe a third of the way — but it is the third that everything else sits on.

---

## 4. Roadmap

Sizing is in *sessions of focused work*, calibrated against what phase one
actually took, not guessed from nothing. Each item is a separate PR.

### Phase 2 — parallel across the nodes

The point of the collaboration. Coefficients already exist; this is execution.

| item | size | notes |
|---|---|---|
| `threading` field, `@threaded` node loop, per-node scratch | 1 | `PDIRK44` is the precedent; core machinery exists |
| reject non-diagonal `QΔ` when threaded, validation | small | |
| shared-memory benchmark on a real problem | 1–2 | needs a problem the group recognises |
| `MIN3`, `MIN_VDHS` tables for comparison against the literature | 1 | published tables, straight port |

**Blocker to resolve first**: with `M` threads each holding its own `NLSolver`,
memory is `M×(J + W)`. For a large system that may dominate. Worth asking Fynn
and Thibaut what problem sizes they care about before committing to the design.

### Phase 3 — IMEX SDC

Fynn's stated expectation for where SDC wins, and the strongest argument for the
method existing in this library at all.

| item | size | notes |
|---|---|---|
| `SDCIMEX` type, split cache, `issplit`, second `QΔ^E` | 2 | `nlsolve_f` gives the hard part free |
| convergence gate against qmat's IMEX ordering | 1 | qmat can generate the reference |
| stiff/non-stiff split test problems | 1 | advection–diffusion, shallow water |

### Phase 4 — completing the coefficient space

Cheap, mechanical, high value for cross-checking against the group's results.

| item | size |
|---|---|
| `CHEBY-1…4` node distributions | 1 |
| the 12 easy missing `QΔ` generators | 1 |
| `SOE`, `MIN`, `Jumper`, `FlexJumper` | 1 |

### Phase 5 — things the papers say we will want

| item | size | reference |
|---|---|---|
| residual-based error estimate as an alternative to the sweep difference | 1 | Baumann et al. 2024 |
| collocation-polynomial dense output | 1 | free from the node values |
| inexact SDC (loose inner tolerances) | 1–2 | Speck 2016 |
| index-1 DAE support | 2–3 | Bolten & Wimmer 2026 — plays to Harsh's DAE background |

### Explicitly out of scope

MLSDC, PFASST, ParaDiag, Boris-SDC, Verlet-SDC, and the full work-precision
study. Each is a project in its own right.

---

## 5. Total, and what I would actually do

Roughly **15–20 focused sessions** to get through phases 2–5, of which phases 2
and 3 (about 8) are the ones that make the joint paper possible. Phase 4 is
padding that can be done any time or skipped. Phase 5 is optional except for the
DAE item, which I would argue is the most interesting thing on the list given
who is working on it and what Harsh already knows.

If I had to cut it to the minimum that produces a paper: **phase 2 + phase 3 +
one benchmark problem**. That is about 8 sessions.

The sequencing is not mine to change — Fynn set it as vanilla, then parallel,
then IMEX, and phase one confirmed the design holds up.

## 6. Open questions that block estimating further

1. What problem sizes matter for the shared-memory work? Decides whether
   `M×(J + W)` memory is acceptable.
2. Which of the 17 missing `QΔ` generators do they actually use? I would rather
   port the four they run than all seventeen.
3. Is the DAE direction (Bolten & Wimmer 2026) in scope for this collaboration or
   a separate thread?
4. Do they want `CHEBY-*` nodes, or is Legendre plus equidistant enough for the
   comparisons they plan?

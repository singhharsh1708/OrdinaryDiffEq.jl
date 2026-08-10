# SDC phase-one notes

Branch-local working notes for the SDC implementation in `lib/OrdinaryDiffEqSDC`.
**Delete this directory before opening a PR** — it is not package content.

| file | what it is |
|---|---|
| `01-sdc-explained.md` | the method itself: collocation, Picard, the sweep, order behaviour, parallel across the nodes, where SDC competes |
| `02-nonlinear-solver-reuse.md` | can SDC reuse the existing nonlinear solver and `W` machinery, with the code evidence |
| `03-design.md` | the design, written before the code, including the phase-two parallel and IMEX designs |
| `04-verification.md` | what was measured: coefficient comparison against `qmat`, the 470-case order gate, negative controls, non-linear problems, ecosystem check |
| `05-status.md` | verified / inferred / open, separated |
| `tools/gen_reference.py` | regenerates `lib/OrdinaryDiffEqSDC/test/sdc_reference.jl` from `qmat` |

Reproducing the reference data:

```bash
git clone --depth 1 https://github.com/Parallel-in-Time/qmat.git
python -m venv venv && ./venv/bin/pip install numpy scipy
./venv/bin/python sdc-notes/tools/gen_reference.py   # writes qmat_reference.json
```

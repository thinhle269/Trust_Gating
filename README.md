# ZeroTrust-ANFIS-FL

Federated traffic forecasting that keeps working when some of the roadside
sensors are compromised. An eight-rule first-order Takagi–Sugeno ANFIS scores
each client from three quantities the server can measure for itself, and a
zero-trust gate with a self-calibrating threshold decides which updates are
allowed into the aggregate.

Code and results for the accompanying paper. Every number in the paper is
produced by this repository — nothing is transcribed by hand.

---

## What is here

```
config.py            datasets, run configuration, cache key, compute detection
run_all.py           verify -> test -> train -> export
src/                 the method and the experiment harness
tests/               invariant tests for the ANFIS (gradients, learning)
results/csv/         per-run and aggregated results
results/figures/     the figures
results/excel/       the same results as a workbook
results/raw/         504 cached runs, one JSON each
PROJECT_SPEC.md      design decisions, deviations, and why each was made
REVIEWER_RESPONSE_reproducibility.md   the cache-validity audit, in full
```

Not included: the datasets (155 MB, public — see below), and the LaTeX and Word
sources for the manuscript.

## Method

The server computes three antecedents per client per round, none of them
self-reported:

| | quantity | why it resists gaming |
|---|---|---|
| `x1` | macro-F1 of the client's update on the server's small clean root set | measured against data the adversary cannot touch |
| `x2` | directional similarity to the geometric-median consensus | cheap, but see the caveat below |
| `x3` | cleanliness — update norm against the cohort median, in log space | inflation and shrinkage read as equally anomalous |

An eight-rule ANFIS (3 inputs × 2 terms, grid partition) maps these to a trust
value, and its premise and consequent parameters are updated online, so the rule
base adapts as the federation proceeds. Supervision comes from server-observable
evidence, never from a ground-truth maliciousness label — that does not exist at
deployment and would make the evaluation circular.

**A finding worth knowing before you reuse `x2`:** directional similarity
*inverts* under a colluding adversary. Measured across all three datasets,
attackers scored **higher** similarity than honest clients (0.78–0.85 against
0.63–0.66), because clients performing the same corruption agree with one
another while honest clients under non-IID data disagree in diverse ways. The
teacher therefore places 0.85 of its weight on `x1` and only 0.15 on `x2` and
`x3` combined. See `PROJECT_SPEC.md` §6c.

## Results

Seven aggregation rules × three datasets × four threat models × six seeds =
**504 runs**. Full numbers in `results/csv/summary_results.csv`.

The method's argument is stability under attack rather than peak accuracy — it
does not win every cell, and the tables show that plainly:

| | worst-case error / own clean error |
|---|---|
| FedProx | 3.64× |
| FedAvg | 3.38× |
| FLTrust | 2.43× |
| Static-Mamdani-FL | 2.31× |
| Multi-Krum | 2.03× |
| SCAFFOLD | 1.72× |
| **ZeroTrust-ANFIS-FL** | **1.015×** |

Detection and what is done with it are separate properties, and conflating them
hides the real behaviour:

| | trust AUC | malicious weight admitted | detection | false positives |
|---|---|---|---|---|
| Static-Mamdani-FL | 1.000 | 0.184 | 0.191 | 0.000 |
| Multi-Krum | 0.954 | 0.029 | 0.951 | 0.164 |
| FLTrust | 0.710 | 0.123 | 1.000 | 0.981 |
| **ZeroTrust-ANFIS-FL** | 0.999 | **0.000** | 1.000 | 0.024 |

Static-Mamdani-FL ranks clients perfectly and still admits 18% of the
aggregation weight to attackers, because a fixed threshold does not sit where the
scores separate. Ranking well is not the same as acting.

Across 72 paired comparisons on test MAE, 55 reach p < 0.05 and no baseline
significantly outperforms the method. Against Multi-Krum, the strongest
comparator, the honest result is 1 win and 11 ties.

## Reproducing

Requires an NVIDIA GPU (developed on a Quadro RTX 4000, CUDA 12.8) — CPU works
but is slow.

```bash
pip install -r requirements.txt
python config.py          # verifies packages, datasets, and compute
python run_all.py         # 6 seeds x 40 rounds; resumes if interrupted
```

Runs are cached per configuration under `results/raw/`, so an interrupted study
resumes for free. `python run_all.py --skip-train` regenerates every output from
the cache without retraining.

### Datasets

Place these in `data/`; all three are public.

| file | sensors | steps | source |
|---|---|---|---|
| `metr_la.h5` | 207 | 34,272 | Los Angeles loop detectors (DCRNN release) |
| `PEMS-BAY.csv` | 325 | 52,116 | Bay Area PeMS (DCRNN release) |
| `PEMS04.npz` | 307 | 16,992 | PeMS District 4 (ASTGCN/STSGCN release) |

PEMS04 is used where the original design called for PeMSD7, and is labelled
honestly as such throughout: the distributed PeMSD7 is speed-only and cannot
support a multi-channel claim, whereas PEMS04 carries genuine flow, occupancy and
speed. See `PROJECT_SPEC.md` §1.

## Reproducibility

Training is deterministic given a seed. This is measured, not assumed — running
one configuration twice differences to exactly `0.000e+00` on every metric.

The run cache is keyed on configuration fields, which does not cover constants
edited inside the source. All 144 trust-rule runs were therefore recomputed from
scratch and compared bitwise against the cache; all 144 reproduced with a largest
difference of zero. `REVIEWER_RESPONSE_reproducibility.md` has the full account,
and `results/csv/trust_rerun_audit.csv` the per-run deltas.

```bash
PYTHONPATH=. python src/rerun_trust.py    # recompute the 144 and diff
PYTHONPATH=. python src/provenance.py     # which runs match the current source
PYTHONPATH=. python src/paper_numbers.py  # every number the paper states in prose
```

`src/provenance.py` fingerprints the modules that can change a metric, over the
parsed AST with docstrings stripped — so rewriting an explanation invalidates
nothing, while changing a constant does.

## Notes on integrity

`src/baselines.py` carries one rule that the rest of the design depends on: **no
branch may test an algorithm's name to decide a metric.** The name selects the
mathematical rule and nothing else. Every baseline receives an identical data
partition, identical initial weights, identical seeds and an identical local
budget; only the server-side combination differs.

FLTrust's server is given an optimisation budget matched to the median client,
because its update-norm rescaling otherwise shrinks every honest update for
reasons unrelated to the method's merit.

## Baselines

FedAvg (McMahan et al., AISTATS 2017) · FedProx (Li et al., MLSys 2020) ·
SCAFFOLD (Karimireddy et al., ICML 2020) · FLTrust (Cao et al., NDSS 2021) ·
Multi-Krum (Blanchard et al., NeurIPS 2017) · Static-Mamdani-FL (fixed-rule
fuzzy ablation of the proposed method)

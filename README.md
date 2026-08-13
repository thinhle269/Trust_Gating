# ZeroTrust-ANFIS-FL

Federated traffic forecasting that keeps working when some of the roadside
sensors are compromised. An eight-rule first-order Takagi–Sugeno ANFIS scores
each client from three quantities the server can measure for itself, and a
zero-trust gate with a self-calibrating threshold decides which updates are
allowed into the aggregate.

 

---
 

```
config.py            datasets, run configuration, cache key, compute detection
run_all.py           verify -> test -> train -> export
src/                 the method and the experiment harness
tests/               invariant tests for the ANFIS (gradients, learning)
results/csv/         per-run and aggregated results
results/figures/     the figures
results/excel/        
results/raw/         504 cached runs, one JSON each
 
```
 

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
base adapts as the federation proceeds.  

 

## Results

Seven aggregation rules × three datasets × four threat models × six seeds =
**504 runs**. Full numbers in `results/csv/summary_results.csv`.

 
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

 

## Reproducibility

Training is deterministic given a seed. This is measured, not assumed — running
one configuration twice differences to exactly `0.000e+00` on every metric.

 

 
```
 

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

# Clean, mark or ignore?

When an imperfect error detector flags cells in a table, which strategy actually improves a downstream model, and how good must detection be for cleaning to pay off?

This repository holds a small, reproducible study. A **simulated detector** flags cells with an exactly controlled recall and false positive rate, so detection quality can be varied on its own. Five strategies are then compared on the same splits and the same injected errors:

| Code | Strategy |
|---|---|
| S0 | Do nothing |
| S1 | Mark flagged cells as missing (native handling in LightGBM; mean imputation plus a missing indicator for the linear model) |
| S2 | Replace them with the training mean or mode |
| S3 | Predict them from the other numeric columns (scikit-learn `IterativeImputer`) |
| S4 | Drop training rows with a flagged cell |
| S5 | Restore their true value (oracle) |

Four real detectors (standard deviation, interquartile range, Isolation Forest, SQL rules with DuckDB) are also placed on the same map. Models: LightGBM and a linear model (logistic or least squares).

## Main results

Five hypotheses and a verdict rule were frozen after a pilot on two tables, then tested on five tables never analysed before. Comparisons use a paired *t*-test over 10 random splits, with the Benjamini-Yekutieli correction over all comparisons of a table. A clear win or loss must also exceed 0.002 of AUROC or 0.5 % of the clean RMSE. The splits overlap, so the *p*-values are optimistic. Classification and regression are reported apart.

- With LightGBM, any cleaning starts to hurt between 1 % and 5 % of false positives. At recall 0.95 and 20 % of false positives, marking loses clearly to doing nothing in 36 of 42 settings across the five tables.
- With LightGBM, marking does as well as simple repair in most settings. The exception is unit errors (values multiplied by 1000), where marking lost clearly in 12 settings: the hypothesis that marking is never worse is **not confirmed** (3 of 5 tables).
- Conditional repair does not track how predictable the affected column is: this hypothesis is **not confirmed** either. The test is poorly posed on tables without predictable columns.
- The gain of the oracle comes from the important columns (confirmed for both models).
- Deleting rows almost never pays off.
- On the one table with real errors (Movies), the errors cost almost nothing, and even the oracle has no significant gain.

These results come from injected errors and two models; they do not say how often real errors reach the zone where cleaning pays off.

## Hypotheses and verdicts

A hypothesis is confirmed if its rule holds on at least 4 of the 5 confirmation tables (Wine, Diamonds, Houses, Adult, Default Credit). Where a rule names no model, each model gets its own vote; this reading was validated after the confirmation results.

| Hypothesis | Rule (summary) | Model | Tables | Verdict |
|---|---|---|---|---|
| H1. On a tree model, marking does at least as well as simple repair | S1 has fewer than 5 % clear losses against S2 | LightGBM | 3/5 | not confirmed |
| H2. Beyond some false positive rate, cleaning hurts | S1 loses clearly to S0 in at least 80 % of settings at recall 0.95 and 20 % of false positives | LightGBM | 5/5 | confirmed |
| | | linear | 1/5 | not confirmed |
| H3. Conditional repair beats marking only on predictable columns | S3 has no clear win over S1 on the least predictable columns, and at least one on the most predictable | LightGBM | 2/5 | not confirmed |
| | | linear | 1/5 | not confirmed |
| H4. The gain comes mostly from important columns | Median gain of S5 over S0 on the most important columns is at least twice that on the least important | LightGBM | 4/5 | confirmed |
| | | linear | 5/5 | confirmed |
| H5. Marking helps more when the error destroys information | At the perfect detector, S1 gains more over S0 on in-domain and swapped categories than on unit errors and typos | LightGBM | 4/5 | confirmed |

The second part of H2 (marking tolerates false positives better than simple repair) is weak and not tested. H4 measures the oracle, not a real repair. H5 came out of a short trial on the pilot tables.

## Limitations

- Errors are injected, except on Movies, a single table with few errors.
- The verdict rules and the negligible-effect thresholds have no theoretical basis.
- Hyperparameters are tuned once on clean data and kept fixed; an AutoML system may absorb errors that a fixed model does not.
- Ten overlapping splits make the *p*-values optimistic. Wine has about 18 % duplicate rows. Wine and Houses have no categorical column.
- The real detectors are simple, and none is learned. The SQL rules were changed once after a first measure.
- Two models only. LightGBM reads a missing value that appears only at prediction time as 0, so marking at prediction time only is a repair by 0, and those results do not carry over to other tree libraries.

## Install and test

    uv sync
    uv run pytest

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). Everything runs on a CPU.

## Reproduce

Tune each table once on clean data (writes `cmi/tuned/<table>.json`, already committed):

    uv run python -m cmi.tuning --table wine-quality

Run the frozen grid of a table on 10 splits. Results go to `results/pilot/<table>/`, and an interrupted run resumes where it stopped:

    uv run python -m cmi.pilot run --table wine-quality --splits 10 --jobs 4
    uv run python -m cmi.pilot report --table wine-quality

On an 8-core laptop with 4 splits in parallel, a table took between about 1 and 3.5 hours (Movies: about 1 minute). Tables: `online-shoppers` and `bike-sharing` (pilot), `wine-quality`, `diamonds`, `houses`, `adult`, `default-credit` (confirmation), `MagicTelescope` and `bank-marketing` (seen in early trials, reported apart), `movies` (real errors).

Draw one figure, or all of them (to `figures/`):

    uv run python -m cmi.figures map
    uv run python -m cmi.figures all

## Repository layout

| Module | Role |
|---|---|
| `cmi/data.py` | Tables, row cap, splits, and the format decoding of Movies |
| `cmi/injection.py` | Error types and mechanisms, with every parameter taken from clean training data |
| `cmi/detector.py` | Simulated detector with exact recall and false positive rate |
| `cmi/real_detectors.py` | Real detectors, fitted on the training part only |
| `cmi/repairs.py`, `cmi/strategies.py` | Strategies S0 to S5 |
| `cmi/models.py`, `cmi/tuning.py` | Models and their one-time tuning |
| `cmi/pilot.py` | The frozen grid, statistics and per-table report |
| `cmi/verdicts.py` | The frozen verdict rules |
| `cmi/figures.py` | Figures |
| `cmi/week1.py`, `cmi/week2.py`, `cmi/grid.py` | Earlier, smaller experiments that led to the grid |

## Data

Tables are fetched at run time and are not stored in this repository. They are capped at 20 000 rows.

| Table | Source | Licence | Citation |
|---|---|---|---|
| MAGIC Gamma Telescope | OpenML 1120 | CC BY 4.0 | Bock, R. (2004), https://doi.org/10.24432/C52C8B |
| Bank Marketing | OpenML 1461 | CC BY 4.0 | Moro, S., Rita, P., & Cortez, P. (2014), https://doi.org/10.24432/C5K306 |
| Online Shoppers Purchasing Intention | OpenML 46947 | CC BY 4.0 | Sakar, C., & Kastro, Y. (2018), https://doi.org/10.24432/C5F88Q |
| Wine Quality | OpenML 287 | CC BY 4.0 | Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009), https://doi.org/10.24432/C56S3T |
| Bike Sharing | OpenML 42712 | CC BY 4.0 | Fanaee-T, H. (2013), https://doi.org/10.24432/C5W894 |
| Adult | OpenML 45068 | CC BY 4.0 | Becker, B., & Kohavi, R. (1996), https://doi.org/10.24432/C5XW20 |
| Default of Credit Card Clients | OpenML 42477 | CC BY 4.0 | Yeh, I.-C., & Lien, C.-H. (2009), https://doi.org/10.24432/C55S3H |
| Diamonds | OpenML 42225 | MIT (ggplot2) | Wickham, H. (2016), *ggplot2: Elegant Graphics for Data Analysis*, Springer |
| California Housing | OpenML 537 | StatLib, no explicit licence found | Pace, R. K., & Barry, R. (1997), "Sparse Spatial Autoregressions", *Statistics & Probability Letters* |
| Movies (real errors) | Raha repository, `datasets/movies_1`, commit `7be1334` | Unclear, see below | |

Dropped columns: the call duration of Bank Marketing (`V12`), known only after the outcome, and the two components of the Bike Sharing target (`casual`, `registered`).

**Movies.** Its clean and dirty versions come from the [Raha repository](https://github.com/BigDaMa/raha) (Apache-2.0 for the repository). The data themselves come from IMDb and Rotten Tomatoes, whose terms of use for this copy are unclear. The files are therefore downloaded at run time from a fixed commit, checked by their SHA-256 hash, and never stored here.

## Licence

Code under the MIT licence (see `LICENSE`). The data keep their own licences, listed above.

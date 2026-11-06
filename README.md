# Clean, mark or ignore?

When an imperfect error detector flags cells in a table, which strategy actually improves a downstream model, and how good must detection be for cleaning to pay off?

Work in progress. Nothing here is a finding yet.

## Usage

Install:

    uv sync

Run the tests:

    uv run pytest

Run the week 1 experiment (it resumes if interrupted), then build its report:

    uv run python -m cmi.week1 run
    uv run python -m cmi.week1 report

Run the strategies against a simulated detector, on a provisional grid (about 15 minutes on a laptop CPU, also resumable):

    uv run python -m cmi.week2 run
    uv run python -m cmi.week2 report

Run them on any table, with a short check first (`--smoke`):

    uv run python -m cmi.grid run --table bike-sharing --splits 3 --smoke
    uv run python -m cmi.grid report --table bike-sharing --smoke

Add `--detector sd`, `iqr`, `iforest` or `rules` to use a real detector, and `--error` to choose the error type. Measure the quality of the real detectors on every table:

    uv run python -m cmi.real_detectors

## Data

Tables are fetched from OpenML and are not stored in this repository. The first five come from the UCI Machine Learning Repository and are licensed under CC BY 4.0.

| Table | OpenML id | Citation |
|---|---:|---|
| MAGIC Gamma Telescope | 1120 | Bock, R. (2004), https://doi.org/10.24432/C52C8B |
| Bank Marketing | 1461 | Moro, S., Rita, P., & Cortez, P. (2014), https://doi.org/10.24432/C5K306 |
| Online Shoppers Purchasing Intention | 46947 | Sakar, C., & Kastro, Y. (2018), https://doi.org/10.24432/C5F88Q |
| Wine Quality | 287 | Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009), https://doi.org/10.24432/C56S3T |
| Bike Sharing | 42712 | Fanaee-T, H. (2013), https://doi.org/10.24432/C5W894 |
| Diamonds | 42225 | Distributed with ggplot2: Wickham, H. (2016), *ggplot2: Elegant Graphics for Data Analysis*, Springer |
| California Housing | 537 | From StatLib: Pace, R. K., & Barry, R. (1997), "Sparse Spatial Autoregressions", *Statistics & Probability Letters* |

Dropped columns: the call duration of Bank Marketing (`V12`), and the two components of the Bike Sharing target (`casual`, `registered`).

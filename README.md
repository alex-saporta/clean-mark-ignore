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

## Data

MAGIC Gamma Telescope: Bock, R. (2004), UCI Machine Learning Repository, https://doi.org/10.24432/C52C8B, licensed under CC BY 4.0. Fetched from OpenML (id 1120).

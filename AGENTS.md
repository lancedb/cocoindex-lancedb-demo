# Repository Guidelines

- Goal: Combine the usage of CocoIndex and LanceDB
- Dataset: Multimodal data (images, text) for recipes of food/drink

The scope of this project is evolving, so more items will be added here later.

## Build, Test, and Development Commands

We will be using `uv` to manage dependencies and Python versions.

- `uv sync`: install Python dependencies for the project.
- `uv run data_generator.py --start 0 --end 5`: create sample JSON records and images in `data/`.
- `uv run ingest.py -o`: ingest data into LanceDB, overwriting the existing database.
- `uv run ingest.py`: append new records to the existing database (upsert).
- `uv run query.py`: run sample text and image similarity queries.

> NOTE: When running code in your sandbox, avoid using `uv run` directly, as you have issues running
> it in your sandbox. Activate the local uv environment via `source .venv/bin/activate`.

## Coding Style & Naming Conventions

- Python code uses 4 spaces for indentation; avoid tabs.
- Prefer `snake_case` for variables/functions and `PascalCase` for classes.
- Keep scripts self-contained and focused; avoid adding heavy framework abstractions.
- Formatting/linting: Use ruff
  - `ruff check --fix --select I *.py`
  - `ruff format --line-length 100 *.py`

## Testing Guidelines

- No automated test suite is present today.
- If you add tests, use `pytest` and place them under `tests/` with `test_*.py` naming.

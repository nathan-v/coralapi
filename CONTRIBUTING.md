# Contributing to CoralAPI

## Code Style

This project uses `ruff` for linting and formatting. Please check your changes before submitting PRs:

```bash
uv run ruff check .
uv run ruff format .
```

## Python Versions

Python 3.11+ is supported. 3.11 is the floor because it is the newest interpreter `tflite-runtime` publishes wheels for, and it is the version the container image ships.

## Developer Setup

```bash
git clone https://github.com/nathan-v/coralapi.git
cd coralapi
uv sync
```

## Testing

The test suite is hardware-independent — the Edge TPU delegate is mocked, so no Coral device or `libedgetpu` is needed to run it:

```bash
uv run pytest
```

Running the service itself (`uv run uvicorn coralapi.main:app`) does require an attached Coral TPU; it exits at startup otherwise, by design.

## Pull Request Process

1. Ensure your code passes `uv run ruff check .` and `uv run ruff format --check .`
2. Ensure `uv run pytest` passes
3. Update documentation (README) if your changes affect configuration or behavior
4. Reference any related issues in your PR description

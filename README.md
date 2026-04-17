# Regression Predictor Agent

A multi-agent system that predicts which CI tests are most likely to break before merging a PR, using semantic diff similarity and historical CI failure data.

## How it works

1. **Data Collection** — fetches recent commits, diffs, and CI results from GitHub
2. **Embedding** — converts each diff into a 384-dim vector using a local sentence-transformer model
3. **Retrieval** — finds the most semantically similar historical diffs for a new diff
4. **Ranking** — scores each test by its weighted failure rate across similar diffs
5. **Explanation** — uses Claude to generate a one-sentence reason per flagged test

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add secrets to .env
cp .env.example .env   # then fill in your tokens
# GITHUB_TOKEN=...
# ANTHROPIC_API_KEY=...
# TARGET_REPO=owner/repo
```

## Usage

```bash
# Build the vector DB from historical CI data
python main.py build

# Predict failing tests for the current branch diff
python main.py predict

# Predict from a specific commit SHA
python main.py predict --sha <commit-sha>

# Predict from a saved diff file
python main.py predict --diff path/to/changes.diff

# Save the prediction report to JSON
python main.py predict --output report.json
```

## Evaluation

```bash
# Run leave-one-out backtesting (20 commits by default)
python evaluate.py

# Evaluate a larger sample
python evaluate.py --sample 50
```

Results are saved to `storage/eval_results.json`.

## Configuration

Edit `config.py` to tune:

| Parameter | Default | Description |
|---|---|---|
| `TOP_K_SIMILAR` | 20 | Similar diffs retrieved per prediction |
| `TOP_N_TESTS_TO_REPORT` | 12 | Max tests included in each prediction |
| `MIN_FAILURE_SCORE_THRESHOLD` | 0.1 | Minimum score to include a test |
| `CLAUDE_MODEL` | claude-opus-4-6 | Model used for explanations |

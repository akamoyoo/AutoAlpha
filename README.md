# AutoAlpha

AutoAlpha is a workflow repository for generating and evaluating quantitative factors.

Its core goals are:
- Let AI agents write new factors using a consistent template
- Automatically evaluate factor quality with `factors/evaluate_factor.py`
- Route factors into `factors/passed` or `factors/rejected` based on results
- Produce evaluation reports in `factors/reports`

## How To Use

You can use `Codex` or `Claude Code` as your agent and give it the prompt below:

```text
Please refer to factors/README.md, factors/factors.example, and factors/reports. Try to write a new factor, then write a report based on the results from evaluate_factor.py.
```

## Directory Guide

- `factors/README.md`: factor evaluation workflow and rules
- `factors/factors.example`: factor template
- `factors/inbox`: new factors waiting for evaluation
- `factors/passed`: factors that passed evaluation
- `factors/rejected`: factors that failed evaluation
- `factors/reports`: JSON/Markdown evaluation reports

## Upload Rules

The following are intentionally excluded from GitHub:
- `Data/`
- all `*.parquet` files
- all `__pycache__/` directories

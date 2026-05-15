# causal-stuff

A playground for experimenting with causal inference and treatment effect estimation methods on open datasets. Built with Claude Code to speed up prototyping and testing different approaches.

## What's here

Testing various causal methods (propensity score matching, Mahalanobis matching, double ML, metalearners) on benchmark datasets like LaLonde/NSW to see how well they recover true treatment effects.

Using frameworks like DoWhy, EconML, and CausalML to compare approaches and evaluate bias.

## Setup

```bash
source .venv/bin/activate
uv sync
jupyter lab
```

Check out `CLAUDE.md` for more details on structure and conventions.

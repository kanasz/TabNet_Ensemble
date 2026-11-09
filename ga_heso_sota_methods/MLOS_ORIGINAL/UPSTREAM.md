# Vendored upstream: MLOS

Unmodified copy of https://github.com/ytyancp/MLOS at commit
`2f0a4b0ccbcbbd308faf6a20e28d67b2d79b6cb4` ("Add files via upload",
2024-07-28), cloned 2026-11-09 with its `.git` removed: `MLOS.py`,
`README.md` and the paper's datasets in `Dataset.rar`.

Do not edit anything in this directory except this file. Upstream is an
oversampler only (no training or evaluation code). It is used by
`optimization/no_ga_mlos_original_runner.py`, which loads `MLOS.py` by file
path and calls it exactly as upstream's `__main__` does,
`MLOS(len=1).fit_Sample(x_train, y_train, n_majority - n_minority)`, with the
module defaults `k = 5` and `aer = -0.5`, on the training part of each fold.
Upstream expects majority = 1 and minority = 0 (README); the runner maps the
labels accordingly. Nothing is generated in this directory at run time.

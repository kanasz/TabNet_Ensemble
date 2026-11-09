# Vendored upstream: CCO

Unmodified copy of https://github.com/priyomondal/CCO at commit
`8064bca3b20a18d971be84ce44973d8546595d42` ("Delete colabs directory",
2025-11-18), cloned 2026-11-06 with its `.git` removed.

Do not edit anything in this directory except this file. It is driven by
`optimization/no_ga_cco_original_runner.py`, which runs upstream's own
`main.py::main()` (no GA, no re-implemented training loop). The only runtime
adaptation is the network width (upstream's `Net` is fixed at 34 inputs /
6 classes for the bundled dermatology set).

Generated at run time (gitignored): `data/<name>/` holding X, Y, TEST.txt,
model*.pt and the runner's `_runner_records.pkl`.

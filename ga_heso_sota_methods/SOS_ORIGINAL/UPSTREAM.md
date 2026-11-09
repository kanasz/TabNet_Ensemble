# Vendored upstream: SOS

Unmodified copy of https://github.com/JayoungKim408/SOS at commit
`f1b450cf199d5e0013b401e380008446ef8757f7` ("Delete default_tabular_configs.py",
2023-02-10), cloned 2026-11-06 with its `.git` removed.

Do not edit anything in this directory except this file. It is driven by
`optimization/no_ga_sos_original_runner.py`, which runs upstream's own
`main.py` per fold exactly as `run.sh` does (`--mode train`, then
`--mode fine_tune`) with upstream's `configs/ve/default.py`, setting only the
dataset name and `image_size`.

Upstream pins numpy 1.23, pandas 1.5, torch 1.12 and pomegranate 0.14. The
runner restores that behaviour at runtime where the installed versions differ
(see its docstring); no file here is patched.

Generated at run time (gitignored): `tabular_datasets/<name>_exp<k>.{npz,json}`
for the project's folds, and `runs/<name>/exp<k>/` (config.py, checkpoints,
tensorboard logs, upstream's stdout/fine-tune logs, the runner's records).

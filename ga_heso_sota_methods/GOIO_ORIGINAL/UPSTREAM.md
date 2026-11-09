# Vendored upstream: GOIO

Unmodified copy of https://github.com/MLneuers/GOIO at commit
`4551ca6164f86c3f7c8964fe935713c5ea6862cf` ("initial commit", 2025-01-20),
cloned 2026-11-08 with its `.git` and committed `__pycache__` folders removed.

Do not edit anything in this directory except this file. It is driven by
`optimization/no_ga_goio_original_runner.py`, which runs upstream's own
`main.py` exactly as the README does:

    python main.py --dataname <name> --method data  --mode split
    python main.py --dataname <name> --method MLVAE --mode train  --exp <k>
    python main.py --dataname <name> --method CLDM  --mode train  --exp <k>
    python main.py --dataname <name> --method CLDM  --mode sample --exp <k>
    python main.py --dataname <name> --method CLDM  --mode eval   --exp <k>

with all of upstream's default arguments.

Generated at run time (gitignored): `data/datasets/<name>/` (the csv and
upstream's CLASSIC/SOS/TEST/GOIO fold folders), `ckpt_1_1_1/<name>/exp<k>/`
(MLVAE + CLDM checkpoints), `synthetic/<name>/exp<k>/CLDM/` (synthetic data,
upstream's evaluation csv, the runner's records and logs).

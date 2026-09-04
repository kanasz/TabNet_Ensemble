"""
Smoke test: run the UNMODIFIED DGOT clone (ga_heso_sota_methods/DGOT_ORIGINAL)
on one wine dataset.

This does not touch ga_heso_sota_methods/DGOT - it drives the pristine clone,
using the paper's own default hyperparameters, its own train() and its own
scripts/evaluate_binary.DGOT(). The only project code involved is
prepare_dgot_data(), which just writes the .npy folder layout DGOT expects; it
imports nothing but numpy/sklearn, so no modified DGOT module is pulled in.

    python test_dgot_original_wine.py                 # exp0 only, 150 epochs
    python test_dgot_original_wine.py --epochs 800    # the paper's full budget
    python test_dgot_original_wine.py --exp exp1

NOTE on epochs: upstream saves netG.pth only inside

    if epoch % args.save_ckpt_every == 0 and epoch > 100:   (train.py:328)

so anything <= 100 epochs trains and then writes no model at all, and the
evaluation step has nothing to load. 150 is the smallest round number that
produces a checkpoint; it is a smoke test, not a result.

NOTE on the protocol: upstream picks which generator to keep by scoring
candidates on the TEST fold (val_data_process reads datasets/{name}/TEST/{exp}),
so the numbers this prints are optimistic. That test-set selection is exactly
what our adapted copy removes.
"""

import argparse
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_ORIGINAL_DGOT = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'DGOT_ORIGINAL', 'DGOT')

# Build the dataset folders before touching the clone: prepare_dgot_data only
# needs numpy/sklearn, and importing it now avoids doing so after the chdir.
# base_functions pulls in torch, so it is imported inside main() instead -
# that keeps --help usable on a machine without it.
from ga_heso_sota_methods.DGOT.prepare_data import prepare_dgot_data  # noqa: E402


def _build_args(dataset, exp, feature_len, epochs, device):
    """The paper's own defaults, read off DGOT_ORIGINAL/DGOT/train.py's argparse."""
    return argparse.Namespace(
        # diffusion
        use_geometric=False,
        beta_min=0.1,
        beta_max=20.0,
        num_timesteps=4,
        # training
        seed=666,
        batch_size=512,
        num_epoch=epochs,
        device=device,
        exp=exp,
        save_content=False,
        save_content_every=50,
        save_ckpt_every=5,
        resume=False,
        # optimiser
        lr_d=2e-3,
        lr_g=5e-3,
        beta1=0.8,
        beta2=0.9,
        # regularisation
        r1_gamma=0.05,
        lazy_reg=None,
        # dataset
        dataset=dataset,
        class_num=2,
        # loss
        pw1=1.0,
        pw2=1.0,
        # generator
        init_ch=16,
        ch_mult=[1, 2, 2],
        feature_len=feature_len,
        nz=50,
        rbg=4,
        # discriminator
        num_channels=1,
        t_emb_dim=32,
        ngf=32,
        # configs
        save_configs=True,
        use_configs=False,
        configs_file=os.path.join('.', 'configs', 'configs_binary.yaml'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--epochs', type=int, default=150,
                        help='must exceed 100 or upstream never saves netG.pth')
    parser.add_argument('--exp', default='exp0', help='which of the 5 folds to run')
    parser.add_argument('--dataset-name', default='wine_red_3_vs_5_original')
    args_cli = parser.parse_args()

    if not os.path.isdir(_ORIGINAL_DGOT):
        raise SystemExit(f"Original DGOT clone not found at {_ORIGINAL_DGOT}")
    if args_cli.epochs <= 100:
        print(f"WARNING: --epochs {args_cli.epochs} <= 100; upstream will not "
              f"save netG.pth and evaluation will fail (train.py:328)")

    from base_functions import get_wine_quality_red_3_vs_5_data

    print(f"Preparing {args_cli.dataset_name} under {_ORIGINAL_DGOT} ...")
    data = get_wine_quality_red_3_vs_5_data()
    feature_len = prepare_dgot_data(
        data, args_cli.dataset_name, base_dir=_ORIGINAL_DGOT,
        numerical_cols=list(data[0].columns.values), categorical_cols=None,
    )
    print(f"  feature_len = {feature_len}")

    # Upstream uses './datasets/...', './saved_log/...' and top-level
    # 'from models... import', so it only runs from inside its own directory.
    os.chdir(_ORIGINAL_DGOT)
    sys.path.insert(0, _ORIGINAL_DGOT)

    import torch
    from sklearn.ensemble import RandomForestClassifier
    from train import train as dgot_train
    from scripts.evaluate_binary import DGOT as dgot_evaluate

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_args = _build_args(args_cli.dataset_name, args_cli.exp, feature_len,
                             args_cli.epochs, device)

    print(f"Training original DGOT: dataset={args_cli.dataset_name} "
          f"{args_cli.exp} epochs={args_cli.epochs} device={device}")
    start = time.time()
    dgot_train(train_args)
    print(f"--- training: {time.time() - start:.1f} seconds ---")

    model_dir = f'./saved_log/DGOT/{args_cli.dataset_name}/{args_cli.exp}'
    test_dir = f'./datasets/{args_cli.dataset_name}/TEST/{args_cli.exp}'
    checkpoint = os.path.join(model_dir, 'netG.pth')
    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"No netG.pth in {model_dir} - upstream only saves one when "
            f"epoch % save_ckpt_every == 0 and epoch > 100."
        )
    print(f"Checkpoint: {checkpoint}")

    print("Evaluating with the original scripts/evaluate_binary.DGOT() ...")
    results = dgot_evaluate(
        filepath=model_dir, testpath=test_dir,
        classifiers=RandomForestClassifier(n_estimators=100, random_state=42),
        oversample_rate=1.2, repetitions=5, devices=device,
    )
    # Upstream's indicator_cls reports accuracy / macro_f1 / mcc only - it has
    # no gmean, which is why our adapted copy adds one.
    print(results)


if __name__ == '__main__':
    main()
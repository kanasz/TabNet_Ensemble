"""
One-shot diagnostic for GOIO's "AttributeError: 'float' object has no attribute
'item'" (MLVAE/main.py:198).

That line does `dist_loss.item()`, where dist_loss is the 6th value returned by
Model_VAE.forward -> calc_sim_scores (MLVAE/model.py:363):

    sim_scores = 0.0                       # a plain Python float
    for i in torch.unique(labels):
        ...
        sim_scores += torch.max(...)       # only here does it become a tensor
    return sim_scores

so it stays a float exactly when that loop body never runs. The prepared data
is known to be fine (552 rows / 11 features / 552 labels), which rules out the
obvious cause, so this wraps calc_sim_scores to print what it actually receives
on the first few calls and what it returns. GOIO's own code is untouched - the
wrapper is installed at runtime and delegates to the original.

Run it once on the cluster:

    python test_goio_debug_wine.py

It will print a few diagnostic blocks and then fail exactly as the real run
does; the blocks are the answer.
"""

import argparse
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    from base_functions import get_wine_quality_red_3_vs_5_data

    # Importing the runner performs GOIO's required chdir + sys.path setup.
    import optimization.no_ga_goio_runner as runner
    import torch
    from ga_heso_sota_methods.GOIO.data.dataprocessing import Default_processing
    from ga_heso_sota_methods.GOIO.MLVAE import model as goio_model
    from ga_heso_sota_methods.GOIO.MLVAE.main import main as train_mlvae

    dataset_name = 'wine_red_3_vs_5'
    data = get_wine_quality_red_3_vs_5_data()

    print('Preparing data ...')
    runner._prepare_csv(data, dataset_name)
    Default_processing(argparse.Namespace(dataname=dataset_name))
    runner._check_prepared_folds(dataset_name)

    # Wrap calc_sim_scores: same class object main.py imported, so patching the
    # attribute is enough. The original is still what computes the value.
    original = goio_model.Model_VAE.calc_sim_scores
    seen = {'calls': 0}

    def traced(self, z, labels):
        result = original(self, z, labels)
        if seen['calls'] < 3:
            seen['calls'] += 1
            uniq = torch.unique(labels)
            print(f'--- calc_sim_scores call #{seen["calls"]}')
            print(f'    labels : shape={tuple(labels.shape)} dtype={labels.dtype} '
                  f'device={labels.device}')
            print(f'    unique : n={uniq.numel()} values={uniq.flatten().tolist()[:10]}')
            print(f'    z      : shape={tuple(z.shape)}')
            print(f'    proto  : shape={tuple(self.proto.shape)}')
            print(f'    RETURNED: type={type(result).__name__} value={result}')
            if not torch.is_tensor(result):
                print('    ^^ this is the bug: the loop body never ran, so '
                      'sim_scores stayed the initial float 0.0')
        return result

    goio_model.Model_VAE.calc_sim_scores = traced

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args = runner._build_args(dataset_name, 0, device, dist=1, proto=1, kld=1,
                              condition=True, max_beta=1e-2, min_beta=1e-5,
                              lambd=0.7, threshold=0.2)

    print(f'Training MLVAE on exp0 (device={device}) ...')
    train_mlvae(args)


if __name__ == '__main__':
    main()
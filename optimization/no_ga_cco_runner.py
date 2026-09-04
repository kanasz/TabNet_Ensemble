"""
Fixed-baseline (no GA) runner for CCO — reusable across datasets.

CCO is imported and run exactly as published: ga_heso_sota_methods/CCO stays
untouched and Cluster(), synthetic_generation(), FocalLoss() and
CustomDataset() are used as-is. This module only replaces CCO's main.py, which
is a CLI script hardwired to one dataset (it torch.load()s a pickled X/Y pair
and its Net is fixed at 34 features in / 6 classes out), with a function that
takes the (X_df, y_series) pairs the rest of this project uses. CCO's train()
is the one function reimplemented rather than called - see the comment on the
epoch loop for why.

run_cco_baseline() reproduces CCO's own protocol once with the paper's
hyperparameters — no hyperparameter search — and writes the standard
{'fitness', 'true_values', 'predicted_values'} result file that
predictions/parse_results.py reads.

Two properties of the published protocol are reproduced by default and are
worth knowing when this is used as a baseline:

  - epoch_selection='test_best' keeps the best epoch measured on the *test*
    fold (CCO's models.py::model_train does exactly this, and there is no
    validation split), so the reported score is a maximum over epochs.
  - global_scaling=True MinMax-scales the whole dataset before the CV split,
    as CCO's utils.py::load_data does, on top of the per-split scaling.

Both leak test-fold information and both are what the paper's code does. Pass
epoch_selection='final' and global_scaling=False for a leakage-free run whose
numbers are comparable to GA-HESO's; the defaults stay faithful to CCO.

NOTE: CCO's Cluster() stops splitting once fewer than 100 points are
unassigned (utils.py), so on training folds smaller than ~100 samples every
point stays in one cluster and the method degenerates into plain global
oversampling. That is CCO's behaviour, not a bug here.
"""

import os

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.optim as optim
from collections import Counter
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from ga_heso_sota_methods.CCO.cco_common import build_cco_net, encode_categorical
from ga_heso_sota_methods.CCO.utils import (
    Cluster,
    CustomDataset,
    FocalLoss,
    set_seeds,
    synthetic_generation,
)

# CCO's own default: StratifiedKFold(n_splits=5) in utils.py::load_data.
N_FOLDS = 5


def _metrics(y_true, y_pred):
    """The four metrics CCO's utils.py::test reports, computed the same way."""
    return {
        'bacc': balanced_accuracy_score(y_true, y_pred),
        'mcc': sklearn.metrics.matthews_corrcoef(y_true, y_pred),
        'f1': sklearn.metrics.f1_score(y_true, y_pred, average='weighted'),
        'gmean': geometric_mean_score(y_true.tolist(), y_pred.tolist()),
    }


def _predict(net, x_test, batch_size, device):
    """Ordered predictions for the test fold.

    CCO's own test() shuffles its loader and returns metrics only, discarding
    the predictions. Metrics are order-invariant so the values match, but this
    project stores per-fold predictions, which need to stay aligned with y.
    """
    net.eval()
    preds = []
    loader = torch.utils.data.DataLoader(
        CustomDataset(x_test, torch.zeros(len(x_test))),
        batch_size=batch_size, shuffle=False,
    )
    with torch.no_grad():
        for inputs, _ in loader:
            preds.extend(net(inputs.to(device).float()).argmax(dim=1).cpu().numpy())
    net.train()
    return np.array(preds, dtype=int)


def run_cco_baseline(data, dataset_name, results_file, k=0.3, beta=0.5, t=0.5,
                     epochs=100, batch_size=32, gamma=1, lr=0.001, seed=42,
                     D=None, categorical_cols=None, global_scaling=True,
                     epoch_selection='test_best'):
    """Runs CCO once over 5 stratified folds and writes the result file.

    k, beta, t          CCO's own hyperparameters (see its README).
    D                   feature dimensionality used in CCO's density term;
                        defaults to the actual number of columns, which is what
                        its --D flag is meant to be set to.
    epochs, batch_size, gamma, lr
                        training budget; CCO's main.py uses gamma=1, lr=0.001.
    global_scaling      reproduce load_data()'s pre-split MinMax pass.
    epoch_selection     'test_best' (CCO's protocol) or 'final'.
    categorical_cols    names of true categorical columns in data[0].
    """
    if epoch_selection not in ('test_best', 'final'):
        raise ValueError("epoch_selection must be 'test_best' or 'final'")

    set_seeds(seed, torch.cuda.is_available())

    x_df, y_series = data
    x_df = encode_categorical(x_df, categorical_cols)
    x_all = x_df.to_numpy(dtype=np.float32)
    y_all = np.asarray(y_series).astype(int)
    num_classes = len(np.unique(y_all))
    D = x_all.shape[1] if D is None else D

    if global_scaling:
        # CCO's utils.py::load_data scales the full dataset before splitting.
        x_all = MinMaxScaler().fit_transform(x_all).astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CCO baseline on {dataset_name}: n={len(x_all)}, D={D}, "
          f"classes={num_classes}, k={k}, beta={beta}, t={t}, device={device}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    true_values, predicted_values, fold_metrics = [], [], []

    for fold, (train_index, test_index) in enumerate(skf.split(x_all, y_all)):
        x_train, x_test = x_all[train_index], x_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]

        # per-split scaling, as CCO's utils.py::scaling does
        scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

        # Cluster() and synthetic_generation() run on CPU tensors (both pin
        # device='cpu' internally).
        x_train_t = torch.tensor(x_train)
        y_train_t = torch.tensor(y_train.astype(np.float32))

        cluster_labels = Cluster(x_train_t, k, D, t, beta)
        x_syn, y_syn = synthetic_generation(cluster_labels, x_train_t, y_train_t, t)

        # The training step casts labels to int64, which would silently round
        # anything fractional, so fail loudly instead. synthetic_generation
        # builds its labels from cluster-local class values, so a non-integer
        # here means those got mixed up with something else.
        if len(x_syn) != len(y_syn):
            raise ValueError(
                f"CCO returned {len(x_syn)} samples but {len(y_syn)} labels "
                f"on fold {fold} of {dataset_name}"
            )
        uniq = torch.unique(y_syn)
        if not torch.equal(uniq, uniq.round()):
            raise ValueError(
                f"CCO returned non-integer labels on fold {fold} of "
                f"{dataset_name}: {uniq.tolist()[:10]}"
            )

        counts = Counter(y_train.tolist())
        per_cls_weights = torch.tensor(
            [1.0 / counts[c] for c in range(num_classes)], dtype=torch.float32
        ).to(device)
        criterion = FocalLoss(weight=per_cls_weights, gamma=gamma, reduction='none')

        net = build_cco_net(D, num_classes).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        train_loader = torch.utils.data.DataLoader(
            CustomDataset(x_syn.to(device), y_syn.to(device)),
            batch_size=batch_size, shuffle=True,
        )
        x_test_t = torch.tensor(x_test)

        # CCO's utils.py::train() ends by computing a train-set balanced
        # accuracy that its own models.py::model_train discards, and that call
        # raises "ValueError: continuous is not supported" on label dtypes
        # other than the pickled int64 its dermatology set ships with. The loop
        # below is its gradient step with that dead metric left out - same
        # zero_grad / forward / int64 cast / FocalLoss / backward / step.
        best = None
        for epoch in range(epochs):
            net.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).type(torch.int64)
                optimizer.zero_grad()
                loss = criterion(net(inputs), labels)
                loss.backward(retain_graph=True)
                optimizer.step()
            preds = _predict(net, x_test_t, batch_size, device)
            scores = _metrics(y_test, preds)
            if epoch_selection == 'final' or best is None or scores['bacc'] > best[1]['bacc']:
                best = (epoch, scores, preds)

        epoch_best, scores, preds = best
        scores = dict(scores, epoch=epoch_best)
        fold_metrics.append(scores)
        true_values.append(y_test)
        predicted_values.append(preds)
        print("  fold {}/{}  bacc={bacc:.4f} mcc={mcc:.4f} f1={f1:.4f} "
              "gmean={gmean:.4f} (epoch {epoch})".format(fold + 1, N_FOLDS, **scores))

    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)

    fold_df = pd.DataFrame(fold_metrics)
    summary = pd.DataFrame({
        'mean_across_folds': fold_df.mean(axis=0),
        'std_across_folds': fold_df.std(axis=0),
    })
    summary.to_csv(results_file + '_summary.csv')
    print(summary)

    result = {
        'fitness': float(fold_df['gmean'].mean()),
        'true_values': true_values,
        'predicted_values': predicted_values,
    }
    with open(results_file + '.txt', 'w') as f:
        f.write(str(result))
    print(f"Saved result file to {results_file}.txt")
    print(f"evaluated fitness (mean gmean): {result['fitness']:.6f}")

    return result
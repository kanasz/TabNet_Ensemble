"""
Shared helpers for the two CCO entry points (optimization/no_ga_cco_runner.py,
optimization/ga_cco_tuner.py).

Both drive the code in this directory, which is a clone of the published CCO
repository and stays exactly as published; these two helpers only cover what it
cannot do on datasets other than the dermatology set it ships with.

This file is the one addition to the clone - it adds nothing to CCO's own
modules and none of them import it, so re-cloning upstream only ever overwrites
files that are not this one.
"""

import inspect

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from ga_heso_sota_methods.CCO.utils import Net as _UpstreamNet


class _Net(nn.Module):
    """CCO's Net with the input/output widths made arguments.

    Upstream's Net is `Linear(34, 20) -> Linear(20, 12) -> Linear(12, 6)`,
    i.e. hardwired to the 34-feature, 6-class dermatology set it ships with.
    Hidden widths and activations here are identical; only the two ends follow
    the dataset.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 12)
        self.fc3 = nn.Linear(12, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_cco_net(input_dim, num_classes):
    """Prefers the vendored Net when it is parameterised, else the local copy.

    A pristine clone of CCO has `Net.__init__(self)`; an adapted copy in this
    repo has `Net.__init__(self, input_dim, num_classes)`. This works with
    either, so the vendored directory never needs editing.
    """
    try:
        params = list(inspect.signature(_UpstreamNet.__init__).parameters)
    except (TypeError, ValueError):
        params = []
    if len(params) >= 3:  # self, input_dim, num_classes
        return _UpstreamNet(input_dim, num_classes)
    return _Net(input_dim, num_classes)


def encode_categorical(x_df, categorical_cols=None):
    """Makes every column float-castable.

    CCO measures raw pairwise distances, so string columns (e.g. abalone's
    'Sex') have to be label encoded first - the same treatment
    no_ga_goio_runner.py gives the same datasets.
    """
    x_df = x_df.copy()
    for col in x_df.columns:
        if col in (categorical_cols or []):
            x_df[col] = pd.factorize(x_df[col].astype(str).str.strip())[0].astype(float)
        else:
            x_df[col] = x_df[col].astype(float)
    return x_df
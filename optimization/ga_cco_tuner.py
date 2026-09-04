import contextlib
import io
import os
import random
import time
import numpy as np
import torch
import torch.optim as optim

from collections import Counter
from imbalanced_ensemble.metrics import geometric_mean_score
from joblib import Parallel, delayed
from pygad import pygad
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from constants import genes_cco
from ga_heso_sota_methods.CCO.cco_common import build_cco_net, encode_categorical
from ga_heso_sota_methods.CCO.utils import Cluster, synthetic_generation, FocalLoss, CustomDataset

seed = 42
pygad.random.seed(42)

# ── Tuner ─────────────────────────────────────────────────────────────────────

class GaCCOTuner:

    def __init__(self, num_generations, num_parents=10, population=20, input_dim=8,
                 categorical_cols=None):
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.population = population
        self.input_dim = input_dim  # D: number of features, fixed per dataset
        # Names of true categorical columns (e.g. ['Sex'] for abalone). CCO
        # works on raw pairwise distances, so they are label encoded before the
        # frame is cast to float32 in eval_func.
        self.categorical_cols = categorical_cols
        self.X_orig = None
        self.y_orig = None
        self.train_indices = []
        self.test_indices = []

    def parallel_fit(self, index, train_index, test_index, X, y, k, beta, t, gamma, epochs, batch_size, D):
        # Reset RNG state per fold so a given solution evaluates
        # deterministically (same result during GA search and in the
        # on_stop re-evaluation) even though folds now run concurrently in
        # separate worker processes — a shared, once-per-call seed wouldn't
        # reach every worker.
        torch.manual_seed(seed + index)
        np.random.seed(seed + index)
        random.seed(seed + index)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # per-split scaling: fit on train, transform both
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        X_train_t = torch.tensor(X_train)
        y_train_t = torch.tensor(y_train)
        X_test_t  = torch.tensor(X_test).to(device)
        y_test_t  = torch.tensor(y_test).to(device)

        try:
            # Suppress CCO's internal debug prints (RADIUS, Counter, etc.)
            with contextlib.redirect_stdout(io.StringIO()):
                CC = Cluster(X_train_t, k, D, t, beta)
                X_syn, Y_syn = synthetic_generation(CC, X_train_t, y_train_t, t)
        except Exception as e:
            print("CCO failed for this solution:", e)
            return y_test.astype(int), np.zeros(len(y_test), dtype=int), 0.0

        X_syn = X_syn.to(device)
        Y_syn = Y_syn.to(device)

        ct = Counter(y_train.astype(int))
        per_cls_weights = torch.tensor(
            [1.0 / ct[0], 1.0 / ct[1]], dtype=torch.float32
        ).to(device)
        criterion = FocalLoss(weight=per_cls_weights, gamma=gamma, reduction='none')

        # build_cco_net works against a pristine CCO clone too, whose Net is
        # hardwired to 34 features / 6 classes and takes no arguments.
        net = build_cco_net(D, 2).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        fold_generator = torch.Generator()
        fold_generator.manual_seed(seed + index)
        train_loader = torch.utils.data.DataLoader(
            CustomDataset(X_syn, Y_syn), batch_size=batch_size, shuffle=True,
            generator=fold_generator,
        )
        for _ in range(epochs):
            net.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()
                loss = criterion(net(inputs), labels)
                loss.backward(retain_graph=True)
                optimizer.step()

        # evaluate once on the test fold after all epochs — no test-set leakage
        net.eval()
        preds = []
        with torch.no_grad():
            for inputs, _ in torch.utils.data.DataLoader(
                CustomDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
            ):
                preds.extend(net(inputs.to(device)).argmax(dim=1).cpu().numpy())
        preds = np.array(preds)
        fold_gmean = geometric_mean_score(y_test.astype(int), preds)
        print(f"  fold {index + 1}/5  ep={epochs}  gmean={fold_gmean:.4f}")

        return y_test.astype(int), preds, fold_gmean

    def eval_func(self, ga_instance, solution, solution_idx):
        k          = float(solution[0])
        beta       = float(solution[1])
        t          = float(solution[2])
        gamma      = float(solution[3])
        epochs     = 50
        batch_size = int(solution[4])
        D          = self.input_dim

        X = self.X_orig.values.astype(np.float32)
        y = self.y_orig.to_numpy().astype(np.float32)

        results = Parallel(n_jobs=5)(
            delayed(self.parallel_fit)(
                index, self.train_indices[index], self.test_indices[index],
                X, y, k, beta, t, gamma, epochs, batch_size, D,
            ) for index in range(len(self.train_indices))
        )

        true_values      = [r[0] for r in results]
        predicted_values = [r[1] for r in results]
        gmeans            = [r[2] for r in results]

        return np.mean(gmeans), true_values, predicted_values

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        gm_mean, _, _ = self.eval_func(ga_instance, solution, solution_idx)
        elapsed = time.time() - start_time
        print(
            "gmean: {:.6f}  k={:.3f} beta={:.3f} t={:.3f} gamma={:.3f} "
            "batch={}  ({:.1f}s)".format(
                gm_mean, solution[0], solution[1], solution[2],
                solution[3], solution[4], elapsed,
            )
        )
        return gm_mean

    def run_experiment(self, data, fname, log_dir=None):
        """Runs the GA and writes two kinds of output to two places.

        fname is the final result: '{fname}.txt' (metrics + predictions) and
        '{fname}.pkl' (the finished GA state holding the best solution) are the
        only files written there, so a results/{method}/ folder ends up with
        exactly one .txt and one .pkl per dataset.

        log_dir takes the transient per-generation resume checkpoint. It
        defaults to fname's own folder, which keeps the old single-directory
        behaviour for callers that have not been migrated yet.
        """
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        filename = fname
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        log_dir = log_dir if log_dir is not None else os.path.dirname(os.path.abspath(filename))
        os.makedirs(log_dir, exist_ok=True)
        # Resume state lives with the transient output, not with the results.
        checkpoint = os.path.join(log_dir, 'checkpoint')

        self.X_orig, self.y_orig = data
        self.X_orig = encode_categorical(self.X_orig, self.categorical_cols)
        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        def callback_generation(ga_instance):
            print("Generation : {}".format(ga_instance.generations_completed))
            print("Fitness    : {}".format(
                ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            ))
            print("Solution   : {}".format(ga_instance.best_solutions[-1]))
            # Per-generation resume checkpoint - transient, so it goes to the
            # log dir and is overwritten in place.
            ga_instance.save(filename=checkpoint)

        def on_stop(ga_instance, last_population_fitness):
            print('------------------------------------------------')
            new_fitness, true_values, predicted_values = self.eval_func(
                ga_instance, ga_instance.best_solutions[-1], None
            )
            result = {
                'fitness':          new_fitness,
                'true_values':      true_values,
                'predicted_values': predicted_values,
            }
            with open(filename + '.txt', 'w') as f:
                f.write(str(result))
            # The final model: one .pkl next to the one .txt in results/.
            ga_instance.save(filename=filename)
            print('evaluated fitness: {:.6f}'.format(new_fitness))
            print('------------------------------------------------')

        if os.path.exists(checkpoint + '.pkl'):
            ga_instance = pygad.load(checkpoint)
        else:
            ga_instance = pygad.GA(
                num_generations=self.num_generations,
                random_seed=42,
                mutation_type='random',
                parallel_processing=['thread', 1],
                num_parents_mating=self.num_parents,
                crossover_type='single_point',
                parent_selection_type='sss',
                fitness_func=self.fitness_func,
                sol_per_pop=self.population,
                num_genes=len(genes_cco['types']),
                gene_type=genes_cco['types'],
                gene_space=genes_cco['spaces'],
                save_best_solutions=True,
                save_solutions=True,
                mutation_probability=0.1,
                mutation_percent_genes=0.1,
                on_generation=callback_generation,
                on_stop=on_stop,
            )

        ga_instance.run()

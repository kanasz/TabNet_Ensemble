import argparse
import os
import random
import sys
import time
import numpy as np
import torch

from joblib import Parallel, delayed
from pygad import pygad
from sklearn.ensemble import RandomForestClassifier
from constants import genes_dgot
from ga_heso_sota_methods.DGOT.train import train
from ga_heso_sota_methods.DGOT.scripts.evaluate_binary import DGOT as dgot_evaluate
from ga_heso_sota_methods.DGOT.prepare_data import prepare_dgot_data

# Import DGOT components directly from the repository so the method is used
# exactly as published — no modifications.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DGOT_PATH = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'DGOT')
os.chdir(_DGOT_PATH)
sys.path.insert(0, _DGOT_PATH)                       # noqa: E402

seed = 42
pygad.random.seed(42)

# num_epoch is reduced from the original 800 to keep each GA fitness
# evaluation feasible. Use the full 800 in prediction_yeast_dgot.py.
_GA_NUM_EPOCH = 400


class GaDGOTTuner:

    def __init__(self, num_generations, num_parents=10, population=20, dataset_name='yeast3', feature_len=8,
                 numerical_cols=None, categorical_cols=None):
        self.num_generations  = num_generations
        self.num_parents      = num_parents
        self.population       = population
        self.dataset_name     = dataset_name
        self.feature_len      = feature_len
        self.numerical_cols   = numerical_cols
        self.categorical_cols = categorical_cols
        self.data             = None

    def _build_args(self, solution, exp):
        lr_d, lr_g, beta1, beta2, r1_gamma, pw1, num_timesteps, nz = solution
        return argparse.Namespace(
            # diffusion
            use_geometric=False,
            beta_min=0.1,
            beta_max=20.0,
            num_timesteps=int(num_timesteps),
            # training
            seed=seed,
            batch_size=512,
            num_epoch=_GA_NUM_EPOCH,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            exp=exp,
            save_content=False,
            save_content_every=50,
            save_ckpt_every=5,
            resume=False,
            # optimiser
            lr_d=float(lr_d),
            lr_g=float(lr_g),
            beta1=float(beta1),
            beta2=float(beta2),
            # regularisation
            r1_gamma=float(r1_gamma),
            lazy_reg=None,
            # dataset
            dataset=self.dataset_name,
            class_num=2,
            # loss
            pw1=float(pw1),
            pw2=1.0,
            # generator
            init_ch=16,
            ch_mult=[1, 2, 2],
            feature_len=self.feature_len,
            nz=int(nz),
            rbg=4,
            # discriminator
            num_channels=1,
            t_emb_dim=32,
            ngf=32,
            # configs
            save_configs=True,
            use_configs=False,
            configs_file=r'.\configs\configs_binary.yaml',
        )

    def parallel_fit(self, k, solution):
        exp = f'exp{k}'
        args = self._build_args(solution, exp)

        try:
            train(args)
        except Exception as e:
            print(f"DGOT train failed on {exp}: {e}")
            return 0.0, [], []

        model_dir = f'./saved_log/DGOT/{self.dataset_name}/{exp}'
        test_dir  = f'./datasets/{self.dataset_name}/TEST/{exp}'
        train_dir = f'./datasets/{self.dataset_name}/DGOT/{exp}'

        if not os.path.exists(os.path.join(model_dir, 'netG.pth')):
            print(f"No checkpoint saved for {exp} — skipping evaluation")
            return 0.0, [], []

        fold_gmean = 0.0
        try:
            clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            # Reset RNG state right before dgot_evaluate — its internal
            # sample_evaluate() draws unseeded torch.randn/np.random.shuffle,
            # so without this the same solution scores differently every time
            # it's re-evaluated (e.g. GA search vs. the on_stop re-run).
            torch.manual_seed(seed + k)
            np.random.seed(seed + k)
            random.seed(seed + k)
            results = dgot_evaluate(filepath=model_dir, testpath=test_dir, classifiers=clf, oversample_rate=1.2,
                                    repetitions=5, devices=device_str)
            fold_gmean = results['gmean'].iloc[:-2].mean()
        except Exception as e:
            print(f"DGOT evaluate failed on {exp}: {e}")

        # Load fold data and run a supplementary RF to get per-sample predictions.
        # dgot_evaluate is a black-box so we re-run inference here on original
        # (non-augmented) training data — G-mean from preds may differ from fitness.
        true_values, pred_values = [], []
        try:
            X_tr = np.load(os.path.join(train_dir, 'xtrain.npy'))[:, 0, :]
            y_tr = np.load(os.path.join(train_dir, 'ytrain.npy'))
            X_te = np.load(os.path.join(test_dir,  'xtest.npy'))
            y_te = np.load(os.path.join(test_dir,  'ytest.npy'))
            rf_pred = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf_pred.fit(X_tr, y_tr)
            fold_preds = rf_pred.predict(X_te)
            true_values = y_te.astype(int)
            pred_values = fold_preds.astype(int)
        except Exception as e:
            print(f"DGOT per-sample pred failed on {exp}: {e}")

        print(f"  fold {k + 1}/5  gmean={fold_gmean:.4f}")
        return fold_gmean, true_values, pred_values

    def eval_func(self, ga_instance, solution, solution_idx):
        # NOTE: runs 5 DGOT trainings concurrently on whatever device is
        # available. On a single GPU this means 5x the VRAM footprint at
        # once — if you hit a CUDA OOM, lower n_jobs below.
        results = Parallel(n_jobs=5)(
            delayed(self.parallel_fit)(k, solution) for k in range(5)
        )

        geometric_mean_scores = [r[0] for r in results]
        true_values_all       = [r[1] for r in results]
        pred_values_all       = [r[2] for r in results]

        return np.mean(geometric_mean_scores), true_values_all, pred_values_all

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        gm_mean, _, _ = self.eval_func(ga_instance, solution, solution_idx)
        elapsed = time.time() - start_time
        print(
            "gmean: {:.6f}  lr_d={:.5f} lr_g={:.5f} beta1={:.3f} beta2={:.3f} "
            "r1={:.3f} pw1={:.2f} T={} nz={}  ({:.1f}s)".format(
                gm_mean, solution[0], solution[1], solution[2], solution[3],
                solution[4], solution[5], solution[6], solution[7], elapsed,
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
        behaviour for callers that have not been migrated yet. Pass an absolute
        path - this module chdir()s into the DGOT directory at import time, so
        a relative one would resolve there.
        """
        filename = fname
        self.data = data
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        log_dir = log_dir if log_dir is not None else os.path.dirname(os.path.abspath(filename))
        os.makedirs(log_dir, exist_ok=True)
        # Resume state lives with the transient output, not with the results.
        checkpoint = os.path.join(log_dir, 'checkpoint')

        # prepare dataset folder structure; encoded_len reflects OHE expansion
        encoded_len = prepare_dgot_data(
            data, self.dataset_name, base_dir=_DGOT_PATH,
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
        )
        if encoded_len is not None:
            self.feature_len = encoded_len

        def callback_generation(ga_instance):
            print("Generation: {}".format(ga_instance.generations_completed))
            print("Fitness: {}".format(ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
            print("Solution: {}".format(ga_instance.best_solutions[-1]))
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

            # pygad's run() executes num_generations MORE generations on a
            # loaded instance instead of treating it as a total budget, so a
            # job that stopped at generation 31 would otherwise finish at 81.
            # Clamp to what is left of the configured budget.
            done = ga_instance.generations_completed
            remaining = self.num_generations - done
            print("Resuming {}.pkl at generation {}/{} ({} to go)".format(
                checkpoint, done, self.num_generations, max(remaining, 0)))

            # save() pickles the callbacks together with the tuner they were
            # bound to, so a resumed run would otherwise keep using the
            # previous run's paths and settings. Re-bind them to this instance.
            ga_instance.fitness_func = self.fitness_func
            ga_instance.on_generation = callback_generation
            ga_instance.on_stop = on_stop

            if remaining <= 0:
                print("Configured generations already completed - writing the "
                      "final result without running further generations.")
                on_stop(ga_instance, getattr(ga_instance, 'last_generation_fitness', [0.0]))
                return

            ga_instance.num_generations = remaining
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
                num_genes=len(genes_dgot['types']),
                gene_type=genes_dgot['types'],
                gene_space=genes_dgot['spaces'],
                save_best_solutions=True,
                save_solutions=True,
                mutation_probability=0.1,
                mutation_percent_genes=0.1,
                on_generation=callback_generation,
                on_stop=on_stop,
            )

        ga_instance.run()

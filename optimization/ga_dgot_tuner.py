import argparse
import os
import sys
import time
import numpy as np
import torch

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

    def __init__(self, num_generations, num_parents=10, population=20, dataset_name='yeast3', feature_len=8):
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.population = population
        self.dataset_name = dataset_name
        self.feature_len = feature_len
        self.data = None

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

    def eval_func(self, ga_instance, solution, solution_idx):
        geometric_mean_scores = []
        auc_scores = []

        for k in range(5):
            exp = f'exp{k}'
            args = self._build_args(solution, exp)

            try:
                train(args)
            except Exception as e:
                print(f"DGOT train failed on {exp}: {e}")
                geometric_mean_scores.append(0.0)
                auc_scores.append(0.0)
                continue

            model_dir = f'./saved_log/DGOT/{self.dataset_name}/{exp}'
            test_dir = f'./datasets/{self.dataset_name}/TEST/{exp}'

            if not os.path.exists(os.path.join(model_dir, 'netG.pth')):
                print(f"No checkpoint saved for {exp} — skipping evaluation")
                geometric_mean_scores.append(0.0)
                auc_scores.append(0.0)
                continue

            try:
                clf = RandomForestClassifier(n_estimators=100, random_state=seed)
                device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                results = dgot_evaluate(filepath=model_dir, testpath=test_dir, classifiers=clf, oversample_rate=1.2,
                                        repetitions=5, devices=device_str)
                fold_gmean = results['gmean'].iloc[:-2].mean()
                fold_auc = results['auc'].iloc[:-2].mean()
            except Exception as e:
                print(f"DGOT evaluate failed on {exp}: {e}")
                fold_gmean = 0.0
                fold_auc = 0.0

            print(f"  fold {k + 1}/5  gmean={fold_gmean:.4f}  auc={fold_auc:.4f}")
            geometric_mean_scores.append(fold_gmean)
            auc_scores.append(fold_auc)

        return np.mean(geometric_mean_scores), geometric_mean_scores, auc_scores

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

    def run_experiment(self, data, file_name):
        filename = file_name
        self.data = data
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # prepare dataset folder structure once before GA starts
        prepare_dgot_data(data, self.dataset_name, base_dir=_DGOT_PATH)

        def callback_generation(ga_instance):
            print("Generation: {}".format(ga_instance.generations_completed))
            print("Fitness: {}".format(ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
            print("Solution: {}".format(ga_instance.best_solutions[-1]))
            ga_instance.save(filename=filename)

        def on_stop(ga_instance, last_population_fitness):
            print('------------------------------------------------')
            new_fitness, gmeans, aucs = self.eval_func(ga_instance, ga_instance.best_solutions[-1], None)
            result = {
                'fitness':        new_fitness,
                'solution':       ga_instance.best_solutions[-1].tolist(),
                'gmean_per_fold': gmeans,
                'auc_per_fold':   aucs,
            }
            with open(filename + '.txt', 'w') as f:
                f.write(str(result))
            print('evaluated fitness: {:.6f}  std: {:.6f}'.format(new_fitness, float(np.std(gmeans))))
            print('------------------------------------------------')

        if os.path.exists(filename + '.pkl'):
            ga_instance = pygad.load(filename)
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

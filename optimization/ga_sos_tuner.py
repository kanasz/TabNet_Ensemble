import os
import sys
import time

import ml_collections
import numpy as np
import torch
from imblearn.metrics import geometric_mean_score
from pygad import pygad
from sklearn.ensemble import RandomForestClassifier

# Import SOS components directly from the repository so the method is used
# exactly as published — no modifications.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOS_PATH     = os.path.join(_PROJECT_ROOT, 'ga_heso_sota_methods', 'SOS')
sys.path.insert(0, _SOS_PATH)

from run_lib       import train as sos_train, make_noise  # noqa: E402
from datasets      import get_data_inverse_scaler              # noqa: E402
from models        import utils as mutils                  # noqa: E402
from models.ema    import ExponentialMovingAverage          # noqa: E402
from models        import ncsnpp_tabular                   # noqa: E402  registers the model
import losses                                              # noqa: E402
import sampling    as sos_sampling                         # noqa: E402
import sde_lib                                             # noqa: E402
from utils         import restore_checkpoint               # noqa: E402
from prepare_data  import prepare_sos_data, get_dataset_noleak  # noqa: E402

sys.path.insert(0, _PROJECT_ROOT)
from constants import genes_sos                            # noqa: E402

# run_lib.py calls datasets.get_dataset(...) via the module object.
# Patching it here propagates into sos_train() so the score model trains with
# the same leak-free transformer used in _post_evaluate — no test statistics
# influence the normalisation of training data.
import datasets as _sos_datasets                          # noqa: E402
_sos_datasets.get_dataset = get_dataset_noleak

seed = 42
pygad.random.seed(42)

# Reduced from 1,300,001 — enough to get a meaningful checkpoint within GA time budget.
_GA_NUM_ITERS = 30000


def _build_config(solution, dataset_name, exp_k, image_size):
    lr, beta1_opt, beta_min, beta_max, num_scales, ema_rate = solution

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cfg = ml_collections.ConfigDict()

    cfg.training = ml_collections.ConfigDict()
    cfg.training.batch_size                   = 10000   # capped to len(train) in datasets.py
    cfg.training.n_iters                      = _GA_NUM_ITERS
    cfg.training.snapshot_freq                = _GA_NUM_ITERS + 1  # no intermediate snapshots
    cfg.training.log_freq                     = 500
    cfg.training.eval_freq                    = max(_GA_NUM_ITERS // 20, 500)
    cfg.training.snapshot_freq_for_preemption = max(_GA_NUM_ITERS // 10, 100)
    cfg.training.snapshot_sampling            = False
    cfg.training.likelihood_weighting         = False
    cfg.training.continuous                   = True
    cfg.training.reduce_mean                  = False
    cfg.training.sde                          = 'vpsde'

    cfg.optim = ml_collections.ConfigDict()
    cfg.optim.optimizer    = 'Adam'
    cfg.optim.lr           = float(lr)
    cfg.optim.beta1        = float(beta1_opt)
    cfg.optim.eps          = 1e-8
    cfg.optim.weight_decay = 0
    cfg.optim.warmup       = min(5000, _GA_NUM_ITERS // 4)
    cfg.optim.grad_clip    = 1.0
    cfg.optim.fine_tune_lr = 1e-5   # used only in fine_tune(), not train()

    cfg.model = ml_collections.ConfigDict()
    cfg.model.name           = 'ncsnpp_tabular'
    cfg.model.nf             = 128
    cfg.model.hidden_dims    = (256, 256)
    cfg.model.activation     = 'swish'
    cfg.model.layer_type     = 'concat'
    cfg.model.embedding_type = 'positional'
    cfg.model.conditional    = True
    cfg.model.scale_by_sigma = False
    cfg.model.fourier_scale  = 16.0
    cfg.model.ema_rate       = float(ema_rate)
    cfg.model.sigma_min      = 0.01
    cfg.model.sigma_max      = 10.0
    cfg.model.num_scales     = int(num_scales)
    cfg.model.beta_min       = float(beta_min)
    cfg.model.beta_max       = float(beta_max)
    cfg.model.dropout        = 0.1

    cfg.data = ml_collections.ConfigDict()
    cfg.data.dataset               = f'{dataset_name}_exp{exp_k}'
    cfg.data.image_size            = image_size
    cfg.data.num_channels          = 1
    cfg.data.centered              = False
    cfg.data.uniform_dequantization = False
    cfg.data.random_flip           = False

    cfg.sampling = ml_collections.ConfigDict()
    cfg.sampling.method            = 'pc'
    cfg.sampling.predictor         = 'euler_maruyama'
    cfg.sampling.corrector         = 'langevin'
    cfg.sampling.n_steps_each      = 1
    cfg.sampling.noise_removal     = True
    cfg.sampling.probability_flow  = False
    cfg.sampling.snr               = 0.1

    cfg.eval = ml_collections.ConfigDict()
    cfg.eval.batch_size   = 256
    cfg.eval.enable_loss  = True
    cfg.eval.enable_bpd   = False
    cfg.eval.bpd_dataset  = 'test'

    cfg.test = ml_collections.ConfigDict()
    cfg.test.n_iter = 2   # repetitions per downstream-classifier eval inside train()

    cfg.seed   = seed
    cfg.device = device

    return cfg


def _post_evaluate(cfg, workdir, dataset_name, exp_k):
    """
    After sos_train() saves checkpoints, load the minor-class model, generate
    synthetic minority samples, combine with training data, and compute G-mean
    with a RandomForest downstream classifier.
    Returns (gmean, auc) floats, or (0.0, 0.0) if no checkpoint was saved.
    """
    checkpoint_dir = os.path.join(workdir, 'checkpoints')

    # Rebuild the dataset for this fold (needed to get transformer, class splits)
    try:
        train_data_total, eval_ds, (transformer, meta), major_label, minor_label, num_classes = \
            get_dataset_noleak(cfg, uniform_dequantization=False)
    except Exception as e:
        print(f"SOS post-evaluate: dataset rebuild failed: {e}")
        return 0.0, [], []

    inverse_scaler = get_data_inverse_scaler(cfg)
    num_sampling = [int(np.max(list(num_classes))) - list(num_classes)[i] for i in minor_label]

    sde = sde_lib.VPSDE(
        beta_min=cfg.model.beta_min,
        beta_max=cfg.model.beta_max,
        N=cfg.model.num_scales,
    )
    sampling_eps = 1e-3

    synthesized = []
    for j, label in enumerate(minor_label):
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{label}.pth')
        if not os.path.exists(ckpt_path):
            print(f"SOS: no checkpoint for label {label} in {exp_k} — skipping")
            return 0.0, [], []

        score_model = mutils.create_model(cfg)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.model.ema_rate)
        optimizer = losses.get_optimizer(cfg, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        state = restore_checkpoint(ckpt_path, state, cfg.device)

        train_data = torch.tensor(train_data_total[label])
        batch_size  = min(cfg.training.batch_size, len(train_data))
        sampling_shape = (batch_size, train_data.shape[1])
        sampling_fn = sos_sampling.get_sampling_fn(cfg, sde, sampling_shape, inverse_scaler, sampling_eps)

        sample_size = num_sampling[j]
        if sample_size <= 0:
            continue

        # Use major-class data as the starting noise (Z_major approach from SOS paper)
        train_exclude = np.concatenate(train_data_total[np.arange(len(train_data_total)) != j + 1])
        perturbed = make_noise(cfg, sde, train_exclude,
                               shape=(sample_size, cfg.data.image_size), eps=1e-5, test=True)

        model = state['model']
        model.eval()
        with torch.no_grad():
            sample, _ = sampling_fn(model, z=perturbed)

        sample_orig = transformer.inverse_transform(sample.cpu().detach().numpy())
        synthesized.append(sample_orig)

    # Build augmented training set in original space
    train_orig = transformer.inverse_transform(np.concatenate(train_data_total))
    test_orig  = transformer.inverse_transform(eval_ds)

    if synthesized:
        augmented = np.concatenate([train_orig] + synthesized)
    else:
        augmented = train_orig

    X_train = augmented[:, :-1].astype(float)
    y_train  = augmented[:, -1].astype(int)
    X_test  = test_orig[:, :-1].astype(float)
    y_test   = test_orig[:, -1].astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    gmean = geometric_mean_score(y_test, preds)
    return float(gmean), y_test, preds


class GaSOSTuner:

    def __init__(self, num_generations, num_parents=10, population=20,
                 dataset_name='yeast3', image_size=10):
        self.num_generations = num_generations
        self.num_parents     = num_parents
        self.population      = population
        self.dataset_name    = dataset_name
        self.image_size      = image_size

    def eval_func(self, ga_instance, solution, solution_idx):
        gmeans          = []
        true_values_all = []
        pred_values_all = []

        for k in range(5):
            workdir = os.path.join(
                _SOS_PATH, 'saved_log', self.dataset_name, f'exp{k}'
            )
            os.makedirs(workdir, exist_ok=True)

            cfg = _build_config(solution, self.dataset_name, k, self.image_size)

            try:
                sos_train(cfg, workdir)
            except Exception as e:
                print(f"SOS train failed on exp{k}: {e}")
                gmeans.append(0.0)
                true_values_all.append([])
                pred_values_all.append([])
                continue

            gm, y_true, y_pred = _post_evaluate(cfg, workdir, self.dataset_name, k)
            print(f"  fold {k + 1}/5  gmean={gm:.4f}")
            gmeans.append(gm)
            true_values_all.append(y_true)
            pred_values_all.append(y_pred)

        return float(np.mean(gmeans)), true_values_all, pred_values_all

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        gm, _, _ = self.eval_func(ga_instance, solution, solution_idx)
        elapsed = time.time() - start_time
        print(
            "gmean: {:.6f}  lr={:.5f} beta1={:.3f} beta_min={:.3f} "
            "beta_max={:.1f} num_scales={} ema={:.4f}  ({:.1f}s)".format(
                gm, solution[0], solution[1], solution[2],
                solution[3], int(solution[4]), solution[5], elapsed,
            )
        )
        return gm

    def run_experiment(self, data, fname):
        filename = fname
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        prepare_sos_data(data, self.dataset_name)

        def callback_generation(ga_instance):
            print("Generation : {}".format(ga_instance.generations_completed))
            print("Fitness    : {}".format(
                ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            ))
            print("Solution   : {}".format(ga_instance.best_solutions[-1]))
            ga_instance.save(filename=filename)

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
            print('evaluated fitness: {:.6f}'.format(new_fitness))
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
                num_genes=len(genes_sos['types']),
                gene_type=genes_sos['types'],
                gene_space=genes_sos['spaces'],
                save_best_solutions=True,
                save_solutions=True,
                mutation_probability=0.1,
                mutation_percent_genes=0.1,
                on_generation=callback_generation,
                on_stop=on_stop,
            )

        ga_instance.run()
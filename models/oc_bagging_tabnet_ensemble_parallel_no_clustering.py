import os
import sys
import time
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
from joblib import Parallel, delayed
from pygad import pygad
from sklearn.model_selection import StratifiedKFold
from base_functions import custom_resample_minority_clusters, get_preprocessor
from models.oc_bagging_tabnet_ensemble import OCBaggingTabnetEnsemble
from optimization.ga_tabnet_functions import GMean

seed = 42
pygad.random.seed(42)


def _best_overall_solution(ga_instance):
    """The best solution over the whole run, not just the last generation.

    pygad's best_solutions list holds one entry per generation (it is filled
    because save_best_solutions=True), so best_solutions[-1] is whatever the
    final generation happened to produce. best_solutions_fitness holds their
    scores, so the global best is the argmax of that.
    """
    fitness_history = getattr(ga_instance, 'best_solutions_fitness', None)
    if fitness_history is not None and len(fitness_history) > 0:
        return ga_instance.best_solutions[int(np.argmax(fitness_history))]
    return ga_instance.best_solutions[-1]


class GaOCBaggingTabnetEnsembleTunerParallelNoClustering:
    """
    Ablation variant of GaOCBaggingTabnetEnsembleTunerParallel with clustering
    removed. The GA chromosome only selects which classifiers make the
    ensemble; minority oversampling is plain SMOTE/ADASYN with every
    synthetic sample used (no cluster-based subset selection). This isolates
    the marginal contribution of GA-HESO's cluster-level synthetic-sample
    selection from ensemble-member selection alone.
    """

    def __init__(self, tabnet_max_epochs,
                 num_generations, num_parents=10, population=20, config_files=[], device='cuda',
                 numerical_cols=None, categorical_cols=None,
                 save_partial_output=False,
                 sampling_algorithm=None, synthetic_minority_count=1500):
        self.tabnet_max_epochs = tabnet_max_epochs
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.population = population
        self.X_orig = []
        self.y_orig = []
        self.train_indices = []
        self.test_indices = []
        self.device = device
        self.config_files = config_files
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.save_partial_output = save_partial_output
        self.resampling_algorithm = sampling_algorithm
        self.synthetic_minority_count = synthetic_minority_count
        self.best_fitness_so_far = -np.inf

    def parallel_fit(self, index, train_index, test_index, X, y, solution,
                      tb_cls, tabnet_max_epochs):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        preprocessor = get_preprocessor(self.numerical_cols, self.categorical_cols)
        X_train_std = preprocessor.fit_transform(X_train)
        X_valid_std = preprocessor.transform(X_valid)

        cls_sum = np.sum(y_train)
        cls_num_list = [len(y_train) - cls_sum, cls_sum]

        # No clustering: clustering_algorithm=None makes
        # custom_resample_minority_clusters return the full SMOTE/ADASYN
        # output unfiltered — no GA-driven synthetic-sample selection.
        X_train_std, y_train = custom_resample_minority_clusters(
            X_train_std, y_train,
            syntetic_minority_count=self.synthetic_minority_count,
            resampling_algorithm=self.resampling_algorithm,
            clustering_algorithm=None,
        )

        tb_cls.fit(X_train_std, y_train,
                   solution=solution,
                   cls_num_list=cls_num_list,
                   eval_metric=[GMean],
                   max_epochs=tabnet_max_epochs,
                   patience=100,
                   batch_size=5000,
                   drop_last=False)

        y_pred = np.array(tb_cls.predict(X_valid_std))
        gmean = geometric_mean_score(y_valid, y_pred)

        return np.array(y_valid), y_pred, gmean

    def eval_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        X, y = self.X_orig.copy(), self.y_orig.copy()
        gmeans = []
        true_values = []
        predicted_values = []
        classifiers_mask = solution[0:len(self.config_files)]

        valid_classifiers = [item for item, include in zip(self.config_files, classifiers_mask) if include]
        tb_cls_instances = [OCBaggingTabnetEnsemble(valid_classifiers, solution[len(self.config_files):], self.device)
                            for _ in self.train_indices]
        results = Parallel(n_jobs=5)(
            delayed(self.parallel_fit)(
                index,
                self.train_indices[index],
                self.test_indices[index],
                X, y,
                solution,
                tb_cls_instances[index],
                self.tabnet_max_epochs
            ) for index in range(len(self.train_indices))
        )

        for y_valid, y_pred, gmean in results:
            true_values.append(y_valid)
            predicted_values.append(y_pred)
            gmeans.append(gmean)

        gm_mean = np.mean(gmeans)
        t = time.time() - start_time

        return gm_mean, true_values, predicted_values

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        gm_mean, true_values, predicted_values = self.eval_func(ga_instance, solution, solution_idx)
        if np.sum(solution[0:len(self.config_files)]) == 0:
            print("ERROR 0 clfs")
            return 0
        # Only persist a result file when this candidate beats the best fitness
        # seen so far in this run — writing on every single evaluation (every
        # individual x every generation) was flooding the results folder with
        # thousands of near-duplicate files.
        if self.save_partial_output and gm_mean > self.best_fitness_so_far:
            self.best_fitness_so_far = gm_mean
            result = {
                'fitness': gm_mean,
                'true_values': true_values,
                'predicted_values': predicted_values,
                'solution': np.array(solution)
            }
            f = os.path.join(self.log_dir, "{}_{}".format(gm_mean, self.basename))
            with open(f + '.txt', 'w') as data:
                data.write(str(result))

        t = time.time() - start_time
        print("gmean: {}, n_estimators: {}, {} seconds".format(gm_mean, np.sum(solution[0:len(self.config_files)]), t))
        return gm_mean

    def run_experiment(self, data, fname, max_classifier_count=None, log_dir=None):
        """Runs the GA and writes two kinds of output to two places.

        fname is the final result: '{fname}.txt' (metrics + predictions) and
        '{fname}.pkl' (the finished GA state holding the best solution) are the
        only files written there, so a results/{method}/ folder ends up with
        exactly one .txt and one .pkl per dataset.

        log_dir takes everything transient - the per-improvement partial dumps
        and the per-generation resume checkpoint. It defaults to fname's own
        folder, which keeps the old single-directory behaviour for callers that
        have not been migrated yet.
        """
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.filename = fname
        self.basename = os.path.basename(fname)
        self.log_dir = log_dir if log_dir is not None else os.path.dirname(fname)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        # Resume state lives with the transient output, not with the results.
        checkpoint = os.path.join(self.log_dir, 'checkpoint')

        sol_per_pop = self.population
        num_parents_mating = self.num_parents

        self.X_orig, self.y_orig = data

        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        num_classifiers = len(self.config_files)
        gene_type = [int] * num_classifiers
        params = [{'low': 0, 'high': 2}] * num_classifiers

        def custom_mutation(offspring, ga_instance):
            for chromosome in offspring:
                num_genes_to_mutate = max(1, int((ga_instance.mutation_percent_genes) * (len(chromosome))))
                mutation_indices = np.random.choice(range(0, len(chromosome)), num_genes_to_mutate, replace=False)

                for i in mutation_indices:
                    rnd = np.random.rand()
                    if rnd <= ga_instance.mutation_probability:
                        if gene_type[i] == int:
                            chromosome[i] = np.random.randint(ga_instance.gene_space[i]["low"], ga_instance.gene_space[i]["high"])

                nonzero_indices = np.where(chromosome != 0)[0]
                # Repair to a fixed ensemble size: if too many classifiers are
                # active, randomly deactivate the excess; if too few, randomly
                # activate more from the inactive set. O(K) in num_classifiers.
                if len(nonzero_indices) > max_classifier_count:
                    excess_indices = np.random.choice(nonzero_indices, len(nonzero_indices) - max_classifier_count,
                                                       replace=False)
                    chromosome[excess_indices] = 0
                if len(nonzero_indices) < max_classifier_count:
                    zero_indices = np.where(chromosome == 0)[0]
                    new_nonzero_indices = np.random.choice(zero_indices, max_classifier_count - len(nonzero_indices),
                                                           replace=False)
                    chromosome[new_nonzero_indices] = 1

            return offspring

        def custom_initial_population(num_parents_mating, max_classifier_count):
            population = []
            for _ in range(num_parents_mating):
                chromosome = np.zeros(num_classifiers, dtype=int)
                nonzero_indices = np.random.choice(num_classifiers, max_classifier_count, replace=False)
                chromosome[nonzero_indices] = 1
                population.append(chromosome)
            return np.array(population)

        def callback_generation(ga_instance):
            print("Generation : {gen}, Fitness: {fitness}".format(
                gen=ga_instance.generations_completed,
                fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
            # Per-generation resume checkpoint - transient, so it goes to the
            # log dir and is overwritten in place.
            ga_instance.save(filename=checkpoint)
            sys.stdout.flush()

        def on_stop(ga_instance, last_population_fitness):
            print('------------------------------------------------')
            sys.stdout.flush()
            print('last population fitness: {}'.format(last_population_fitness[0]))
            sys.stdout.flush()
            # best_solutions[-1] is the best of the LAST generation, not the
            # best found over the whole run — pick the global best instead.
            best_solution = _best_overall_solution(ga_instance)
            new_fitness, true_values, predicted_values = self.eval_func(ga_instance,
                                                                        best_solution, None)
            result = {
                'fitness': new_fitness,
                'true_values': true_values,
                'predicted_values': predicted_values
            }
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename + '.txt', 'w') as data:
                data.write(str(result))
            # The final model: one .pkl next to the one .txt in results/.
            ga_instance.save(filename=self.filename)
            print('evaluated fitness: {}'.format(new_fitness))
            sys.stdout.flush()
            gm = [geometric_mean_score(true_values[i], predicted_values[i]) for i in range(len(true_values))]
            print('evaluated gmean: {}'.format(np.mean(gm)))
            sys.stdout.flush()
            print('------------------------------------------------')
            sys.stdout.flush()

        exists = os.path.exists(checkpoint + '.pkl')

        if exists:
            ga_instance = pygad.load(checkpoint)

            # pygad's run() executes num_generations MORE generations on a
            # loaded instance instead of treating it as a total budget, so a
            # job that stopped at generation 31 would otherwise finish at 81.
            # Clamp to what is left of the configured budget.
            done = ga_instance.generations_completed
            remaining = self.num_generations - done
            print("Resuming {}.pkl at generation {}/{} ({} to go)".format(
                checkpoint, done, self.num_generations, max(remaining, 0)))
            sys.stdout.flush()

            # save() pickles the callbacks together with the tuner they were
            # bound to, so a resumed run would otherwise keep using the
            # previous run's paths and settings. Re-bind them to this instance.
            ga_instance.fitness_func = self.fitness_func
            ga_instance.on_generation = callback_generation
            ga_instance.on_stop = on_stop

            if remaining <= 0:
                print("Configured generations already completed - writing the "
                      "final result without running further generations.")
                sys.stdout.flush()
                on_stop(ga_instance, getattr(ga_instance, 'last_generation_fitness', [0.0]))
                return

            ga_instance.num_generations = remaining
        else:
            mutation_type = "random"
            initial_population = None
            if max_classifier_count is not None:
                mutation_type = custom_mutation
                initial_population = custom_initial_population(num_parents_mating=sol_per_pop,
                                                               max_classifier_count=max_classifier_count)
            ga_instance = pygad.GA(num_generations=self.num_generations,
                                   random_seed=42,
                                   num_parents_mating=num_parents_mating,
                                   initial_population=initial_population,
                                   mutation_type=mutation_type,
                                   crossover_type="single_point",
                                   parent_selection_type="sss",
                                   fitness_func=self.fitness_func,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_classifiers,
                                   gene_type=gene_type,
                                   save_best_solutions=True,
                                   mutation_probability=0.1,
                                   save_solutions=True,
                                   mutation_percent_genes=0.1,
                                   gene_space=params,
                                   on_stop=on_stop,
                                   on_generation=callback_generation)

        ga_instance.run()
        return

    def evaluate_experiment(self, data, solution):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        self.X_orig, self.y_orig = data
        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        result = self.eval_func(None, solution, None)

        return result

    def evaluate_experiment_from_pkl(self, data, filename):
        ga_instance = pygad.load(filename)
        solution = ga_instance.best_solutions[-1]
        new_fitness, true_values, predicted_values = self.evaluate_experiment(data, solution)

        result = {
            'fitness': new_fitness,
            'true_values': true_values,
            'predicted_values': predicted_values,
            'solution': solution
        }
        with open(filename + '.txt', 'w') as data:
            data.write(str(result))
        print(new_fitness)
        sys.stdout.flush()
        return
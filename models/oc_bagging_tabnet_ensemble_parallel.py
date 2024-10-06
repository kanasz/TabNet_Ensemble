import os
import sys
import time
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
from pygad import pygad
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from base_functions import resample_minority_samples, custom_resample_minority_samples
from constants import CLUSTER_COUNT, SYNTHETIC_MINORITY_COUNT
from models.oc_bagging_tabnet_ensemble import OCBaggingTabnetEnsemble
from optimization.ga_tabnet_functions import GMean

seed = 42
pygad.random.seed(42)


class GaOCBaggingTabnetEnsembleTunerParallel:

    def __init__(self, tabnet_max_epochs,
                 num_generations, num_parents=10, population=20, config_files = [], device='cuda',
                 numerical_cols=None, categorical_cols=None,
                 save_partial_output=False,
                 sampling_algorithm=None, clustering_algorithm=None):
        self.tabnet_max_epochs = tabnet_max_epochs
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.population = population
        self.X_orig = []
        self.y_orig = []
        self.loss_function = None
        self.train_indices = []
        self.test_indices = []
        self.device = device
        self.config_files = config_files
        self.categorical_cols = categorical_cols
        self.numerical_cols  = numerical_cols
        self.save_partial_output = save_partial_output
        self.resampling_algorithm = sampling_algorithm
        self.clustering_algorithm = clustering_algorithm

    def parallel_fit(self, index, train_index, test_index, X, y, valid_classifiers, solution, CLUSTER_COUNT,
                     SYNTHETIC_MINORITY_COUNT, tb_cls, numerical_cols, categorical_cols, tabnet_max_epochs, device):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        preprocessor = ColumnTransformer(transformers=[])

        if self.numerical_cols is not None:
            preprocessor.transformers.append(('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]),
                 self.numerical_cols))
        if self.categorical_cols is not None:
            preprocessor.transformers.append(('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]),
                 self.categorical_cols))

        X_train_std = preprocessor.fit_transform(X_train)
        X_valid_std = preprocessor.transform(X_valid)

        selected = solution[index * CLUSTER_COUNT:(index + 1) * CLUSTER_COUNT]

        cls_sum = np.sum(y_train)
        cls_num_list = [len(y_train) - cls_sum, cls_sum]

        '''
        X_train_std, y_train = resample_minority_samples(X_train_std, y_train, selected, cluster_count=CLUSTER_COUNT,
                                                         syntetic_minority_count=SYNTHETIC_MINORITY_COUNT)
        '''
        X_train_std, y_train = custom_resample_minority_samples(X_train_std, y_train, selected,
                                                                cluster_count=CLUSTER_COUNT,
                                                                syntetic_minority_count=SYNTHETIC_MINORITY_COUNT,
                                                                resampling_algorithm=self.resampling_algorithm,
                                                                clustering_algorithm=self.clustering_algorithm)
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

        # Return values for aggregation
        return np.array(y_valid), y_pred, gmean

    def eval_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()

        X, y = self.X_orig.copy(), self.y_orig.copy()
        #X = X.values
        #y = y.to_numpy()



        gmeans = []
        true_values = []
        predicted_values = []
        fold = 0

        classifiers_mask = solution[0:len(self.config_files)]

        valid_classifiers = [item for item, include in zip(self.config_files, classifiers_mask) if include]
        tb_cls_instances = [OCBaggingTabnetEnsemble(valid_classifiers, solution[len(self.config_files):], self.device) for _ in
                            self.train_indices]
        results = Parallel(n_jobs=5)(
            delayed(self.parallel_fit)(
                index,
                self.train_indices[index],
                self.test_indices[index],
                X, y,
                valid_classifiers, solution,
                CLUSTER_COUNT, SYNTHETIC_MINORITY_COUNT,
                tb_cls_instances[index],
                self.numerical_cols, self.categorical_cols,
                self.tabnet_max_epochs,
                self.device
            ) for index in range(len(self.train_indices))
        )

        for y_valid, y_pred, gmean in results:
            true_values.append(y_valid)
            predicted_values.append(y_pred)
            gmeans.append(gmean)

        gm_mean = np.mean(gmeans)
        #return true_values, predicted_values, gmeans, gm_mean
        t = time.time() - start_time

        return gm_mean, true_values, predicted_values

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        #try:
        gm_mean, true_values, predicted_values = self.eval_func(ga_instance, solution, solution_idx)

        result = {
            'fitness': gm_mean,
            'true_values': true_values,
            'predicted_values': predicted_values,
            'solution': np.array(solution)
        }
        arr = self.filename.split("/")
        arr[-1] = "{}_{}".format(gm_mean,arr[-1])
        f = "/".join(arr)
        if self.save_partial_output:
            with open(f + '.txt', 'w') as data:
                data.write(str(result))

        t = time.time() - start_time
        print("gmean: {}, n_estimators: {}, {} seconds".format(gm_mean, np.sum(solution[0:len(self.config_files)]), t))

        return gm_mean

    def run_experiment(self, data, fname, loss_function):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.loss_function = loss_function
        self.filename = fname
        sol_per_pop = self.population
        num_parents_mating = self.num_parents

        self.X_orig, self.y_orig = data

        params = [{'low': 0, 'high': 2}] * (CLUSTER_COUNT * 5 + len(self.config_files))
        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        def callback_generation(ga_instance):
            print("Generation : {gen}".format(gen=ga_instance.generations_completed))
            print("Fitness    : {fitness}".format(
                fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
            print("Solution   : {solution}".format(solution=ga_instance.best_solutions[-1]))
            ga_instance.save(filename=self.filename)
            sys.stdout.flush()

        def on_stop(ga_instance, last_population_fitness):
            print('------------------------------------------------')
            sys.stdout.flush()
            print('last population fitness: {}'.format(last_population_fitness[0]))
            sys.stdout.flush()
            new_fitness, true_values, predicted_values = self.eval_func(ga_instance,
                                                                        ga_instance.best_solutions[-1], None)
            result = {
                'fitness': new_fitness,
                'true_values': true_values,
                'predicted_values': predicted_values
            }
            with open(self.filename + '.txt', 'w') as data:
                data.write(str(result))
            print('evaluated fitness: {}'.format(new_fitness))
            sys.stdout.flush()
            gm = []
            gm.append(geometric_mean_score(true_values[0], predicted_values[0]))
            gm.append(geometric_mean_score(true_values[1], predicted_values[1]))
            gm.append(geometric_mean_score(true_values[2], predicted_values[2]))
            gm.append(geometric_mean_score(true_values[3], predicted_values[3]))
            gm.append(geometric_mean_score(true_values[4], predicted_values[4]))
            print('evaluated gmean: {}'.format(np.mean(gm)))
            sys.stdout.flush()
            print('------------------------------------------------')
            sys.stdout.flush()

        exists = os.path.exists(self.filename + '.pkl')
        if exists:
            ga_instance = pygad.load(self.filename)
        else:
            ga_instance = pygad.GA(num_generations=self.num_generations,

                                   random_seed=42,
                                   mutation_type="random",
                                   #parallel_processing=['process', 15],
                                   num_parents_mating=num_parents_mating,

                                   crossover_type="single_point",
                                   parent_selection_type="sss",
                                   fitness_func=self.fitness_func,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=CLUSTER_COUNT * 5 + len(self.config_files),
                                   gene_type=[int] * CLUSTER_COUNT * 5 + [int] * len(self.config_files),
                                   save_best_solutions=True,
                                   mutation_probability=0.1,
                                   save_solutions=True,
                                   mutation_percent_genes=0.1,
                                   gene_space=params,
                                   on_stop=on_stop,
                                   # on_fitness=callback_fitness,
                                   on_generation=callback_generation)

        ga_instance.run()
        return

    def evaluate_experiment(self, data, loss_function, solution):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.loss_function = loss_function

        self.X_orig, self.y_orig = data
        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        # self.fitness_func(None, solution, None)
        result = self.eval_func(None, solution, None)

        return result

    def evaluate_experiment_from_pkl(self, data, loss_function, filename):
        ga_instance = pygad.load(filename)
        solution = ga_instance.best_solutions[-1]
        #solution[-1] = 500
        new_fitness, true_values, predicted_values = self.evaluate_experiment(data, loss_function, solution)

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

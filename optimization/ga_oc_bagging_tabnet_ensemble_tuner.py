import os
import time
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from pygad import pygad
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from base_functions import resample_minority_samples
from constants import CLUSTER_COUNT, SYNTHETIC_MINORITY_COUNT
from models.boosting_tabnet import BoostingTabNet
from models.oc_bagging_tabnet import OCBaggingTabNet
from models.oc_bagging_tabnet_ensemble import OCBaggingTabnetEnsemble
from optimization.ga_tabnet_functions import GMean, get_loss, get_boosting_gene_type_and_space, \
    get_oc_bagging_gene_type_and_space

seed = 42
pygad.random.seed(42)


class GaOCBaggingTabnetEnsembleTuner:

    def __init__(self, tabnet_max_epochs, num_generations, num_parents=10, population=20, config_files = [], device='cuda'):
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

    def eval_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()

        X, y = self.X_orig.copy(), self.y_orig.copy()
        X = X.values
        y = y.to_numpy()
        gmeans = []

        true_values = []
        predicted_values = []
        fold = 0

        classifiers_mask = solution[0:len(self.config_files)]
        #valid_classifiers = self.config_files[classifiers_mask==1]

        valid_classifiers = [item for item, include in zip(self.config_files, classifiers_mask) if include]
        for index, train_index in enumerate(self.train_indices):
            test_index = self.test_indices[index]
            tb_cls = OCBaggingTabnetEnsemble(valid_classifiers, solution[len(self.config_files):], self.device)
            X_train, X_valid = X[train_index], X[test_index]

            y_train, y_valid = y[train_index], y[test_index]
            imputer = SimpleImputer()
            X_train_imp = imputer.fit_transform(X_train)
            X_valid_imp = imputer.transform(X_valid)

            std_scaler = StandardScaler()
            X_train_std = std_scaler.fit_transform(X_train_imp)
            X_valid_std = std_scaler.transform(X_valid_imp)
            selected = solution[index * CLUSTER_COUNT:(index+1) * CLUSTER_COUNT]

            cls_sum = np.sum(y_train)
            cls_num_list = [len(y_train) - cls_sum, cls_sum]

            X_train_std, y_train = resample_minority_samples(X_train_std, y_train, selected, cluster_count=CLUSTER_COUNT, syntetic_minority_count=SYNTHETIC_MINORITY_COUNT)

            #cls_sum = np.sum(y_train)
            #cls_num_list = [len(y_train) - cls_sum, cls_sum]

            # loss_fn = get_loss(self.loss_function, solution[10:], cls_num_list, self.device)


            tb_cls.fit(X_train_std, y_train,
                       solution=solution,
                       cls_num_list = cls_num_list,
                       eval_metric=[GMean],
                       max_epochs=self.tabnet_max_epochs,
                       patience=100,
                       batch_size=5000,

                       drop_last=False)
            fold = fold + 1
            y_pred = tb_cls.predict(X_valid_std)
            true_values.append(y_valid)
            predicted_values.append(y_pred)
            gmean = geometric_mean_score(y_valid, y_pred)
            gmeans.append(gmean)
        gm_mean = np.mean(gmeans)
        t = time.time() - start_time
        return gm_mean, true_values, predicted_values

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        #try:
        gm_mean, true_values, predicted_values = self.eval_func(ga_instance, solution, solution_idx)

        '''
        except Exception as e:
            print(e)
            gm_mean = 0
            t = time.time() - start_time
            print("gmean: {}, n_estimators: {}, {} seconds - ERROR".format(gm_mean, solution[9], t))
            return 0
        '''
        t = time.time() - start_time
        print("gmean: {}, n_estimators: {}, {} seconds".format(gm_mean, np.sum(solution[0:len(self.config_files)]), t))

        return gm_mean

    def run_experiment(self, data, fname, loss_function):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.loss_function = loss_function
        filename = fname
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
            ga_instance.save(filename=filename)

        def on_stop(ga_instance, last_population_fitness):
            print('------------------------------------------------')
            print('last population fitness: {}'.format(last_population_fitness[0]))
            new_fitness, true_values, predicted_values = self.eval_func(ga_instance,
                                                                        ga_instance.best_solutions[-1], None)
            result = {
                'fitness': new_fitness,
                'true_values': true_values,
                'predicted_values': predicted_values
            }
            with open(filename + '.txt', 'w') as data:
                data.write(str(result))
            print('evaluated fitness: {}'.format(new_fitness))
            gm = []
            gm.append(geometric_mean_score(true_values[0], predicted_values[0]))
            gm.append(geometric_mean_score(true_values[1], predicted_values[1]))
            gm.append(geometric_mean_score(true_values[2], predicted_values[2]))
            gm.append(geometric_mean_score(true_values[3], predicted_values[3]))
            gm.append(geometric_mean_score(true_values[4], predicted_values[4]))
            print('evaluated gmean: {}'.format(np.mean(gm)))
            print('------------------------------------------------')

        exists = os.path.exists(filename + '.pkl')
        if exists:
            ga_instance = pygad.load(filename)
        else:
            ga_instance = pygad.GA(num_generations=self.num_generations,

                                   random_seed=42,
                                   mutation_type="random",
                                   #parallel_processing=['process', 1],
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
        return

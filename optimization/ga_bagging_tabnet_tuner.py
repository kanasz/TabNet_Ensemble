import os
import time
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from pygad import pygad
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.bagging_tabnet import BaggingTabNet
from models.boosting_tabnet import BoostingTabNet
from optimization.ga_tabnet_functions import GMean, get_loss, get_boosting_gene_type_and_space, \
    get_bagging_gene_type_and_space

seed = 42
pygad.random.seed(42)


class GaBaggingTabnetTuner:

    def __init__(self, tabnet_max_epochs, num_generations, num_parents=10, population=20, device='cuda',
                 use_smote=False, use_new_model=False):
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
        self.use_smote = use_smote
        self.use_new_model = use_new_model

    def eval_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        n_d = int(solution[0])
        n_a = int(solution[1])
        n_steps = int(solution[2])
        gamma = np.float64(solution[3])
        lambda_sparse = np.float64(solution[4])
        momentum = np.float64(solution[5])
        n_shared = int(solution[6])
        n_independent = int(solution[7])
        learning_rate = (solution[8])
        n_enstimators = int(solution[9])

        X, y = self.X_orig.copy(), self.y_orig.copy()
        X = X.values
        y = y.to_numpy()
        gmeans = []

        true_values = []
        predicted_values = []
        fold = 0
        for index, train_index in enumerate(self.train_indices):
            test_index = self.test_indices[index]
            tb_cls = BaggingTabNet(seed=42, n_d=n_d, n_a=n_a,
                                    device=self.device,
                                    n_steps=n_steps, gamma=gamma, lambda_sparse=lambda_sparse, momentum=momentum,
                                    n_shared=n_shared, n_independent=n_independent, n_estimators=n_enstimators,
                                    learning_rate=learning_rate)
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            imputer = SimpleImputer()
            X_train_imp = imputer.fit_transform(X_train)
            X_valid_imp = imputer.transform(X_valid)

            std_scaler = StandardScaler()
            X_train_std = std_scaler.fit_transform(X_train_imp)
            X_valid_std = std_scaler.transform(X_valid_imp)
            cls_sum = np.sum(y_train)
            cls_num_list = [len(y_train) - cls_sum, cls_sum]
            loss_fn = get_loss(self.loss_function, solution[10:], cls_num_list, self.device)

            if self.use_smote:
                smote = SMOTE(random_state=11, k_neighbors=2)
                X_train_std, y_train = smote.fit_resample(X_train_std, y_train)
            tb_cls.fit(X_train_std, y_train,
                       eval_metric=[GMean],
                       loss_fn=loss_fn,
                       max_epochs=self.tabnet_max_epochs,
                       patience=100,
                       batch_size=3000,
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
        try:
            gm_mean, true_values, predicted_values = self.eval_func(ga_instance, solution, solution_idx)
        except:
            gm_mean = 0
            t = time.time() - start_time
            print("gmean: {}, n_estimators: {}, {} seconds - ERROR".format(gm_mean, solution[9], t))
            return 0
        t = time.time() - start_time
        print("gmean: {}, n_estimators: {}, {} seconds".format(gm_mean, solution[9], t))

        return gm_mean

    def run_experiment(self, data, fname, loss_function):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.loss_function = loss_function
        filename = fname
        sol_per_pop = self.population
        num_parents_mating = self.num_parents

        self.X_orig, self.y_orig = data

        params = get_bagging_gene_type_and_space(loss_function)
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
                                   parallel_processing=['thread', 1],
                                   num_parents_mating=num_parents_mating,

                                   crossover_type="single_point",
                                   parent_selection_type="sss",
                                   fitness_func=self.fitness_func,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=len(params['types']),
                                   gene_type=params['types'],
                                   save_best_solutions=True,
                                   mutation_probability=0.1,
                                   save_solutions=True,
                                   mutation_percent_genes=0.1,
                                   gene_space=params['space'],
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

        self.fitness_func(None, solution, None)
        return

    def evaluate_experiment_from_pkl(self, data, loss_function, filename):
        ga_instance = pygad.load(filename)
        solution = ga_instance.best_solutions[-1]
        self.evaluate_experiment(data, loss_function, solution)
        return


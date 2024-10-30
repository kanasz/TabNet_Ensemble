import os
import sys
import time
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
from joblib import Parallel, delayed
from pygad import pygad
from sklearn.cluster import MeanShift
from sklearn.model_selection import StratifiedKFold
from base_functions import custom_resample_minority_clusters, get_preprocessor
from models.oc_bagging_tabnet_ensemble import OCBaggingTabnetEnsemble
from optimization.ga_tabnet_functions import GMean

seed = 42
pygad.random.seed(42)


class GaOCBaggingTabnetEnsembleTunerParallel:

    def __init__(self, tabnet_max_epochs,
                 num_generations, num_parents=10, population=20, config_files = [], device='cuda',
                 numerical_cols=None, categorical_cols=None,
                 save_partial_output=False,
                 sampling_algorithm=None, clustering_algorithm=None, synthetic_minority_count = 1500, cluster_count = 300,
                 clustering_params = None):
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
        self.synthetic_minority_count = synthetic_minority_count
        self.cluster_count = cluster_count
        self.clustering_params = clustering_params

    def parallel_fit(self, index, train_index, test_index, X, y, solution,
                      tb_cls,  tabnet_max_epochs):
        print("PARALLEL_FIT {}".format(index))
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        preprocessor = get_preprocessor(self.numerical_cols, self.categorical_cols)
        X_train_std = preprocessor.fit_transform(X_train)
        X_valid_std = preprocessor.transform(X_valid)

        if self.clustering_params is None:
            selected = solution[index * self.cluster_count:(index + 1) * self.cluster_count]
            clustering_algorithm = self.clustering_algorithm
        else:
            type = self.clustering_params["type"]
            if type=="MS":
                clustering_algorithm = MeanShift(bandwidth=self.clustering_params["bandwidths"][index])
                if index==0:
                    start_index = 0
                    end_index = self.clustering_params['clusters'][0]
                elif index==4:
                    start_index = np.sum(self.clustering_params['clusters'][0:4])
                    end_index = np.sum(self.clustering_params['clusters'])
                else:
                    start_index = np.sum(self.clustering_params['clusters'][0:index])
                    end_index = np.sum(self.clustering_params['clusters'][0:index+1])

                selected = solution[start_index:end_index]
        cls_sum = np.sum(y_train)
        cls_num_list = [len(y_train) - cls_sum, cls_sum]


        X_train_std, y_train = custom_resample_minority_clusters(X_train_std, y_train, selected,
                                                                    cluster_count=self.cluster_count,
                                                                    syntetic_minority_count=self.synthetic_minority_count,
                                                                    resampling_algorithm=self.resampling_algorithm,
                                                                    clustering_algorithm=clustering_algorithm)

        '''
        X_train_std, y_train = custom_resample_minority_samples(X_train_std, y_train, selected,
                                                                #cluster_count=CLUSTER_COUNT,
                                                                syntetic_minority_count=self.synthetic_minority_count,
                                                                resampling_algorithm=self.resampling_algorithm,
        
                                                                clustering_algorithm=self.clustering_algorithm)
        '''
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
        gmeans = []
        true_values = []
        predicted_values = []
        classifiers_mask = solution[0:len(self.config_files)]

        #if np.sum(classifiers_mask)>10:
        #    return 0, None, None

        valid_classifiers = [item for item, include in zip(self.config_files, classifiers_mask) if include]
        tb_cls_instances = [OCBaggingTabnetEnsemble(valid_classifiers, solution[len(self.config_files):], self.device) for _ in
                            self.train_indices]
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
        #return true_values, predicted_values, gmeans, gm_mean
        t = time.time() - start_time

        return gm_mean, true_values, predicted_values

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        gm_mean = 0
        try:
            gm_mean, true_values, predicted_values = self.eval_func(ga_instance, solution, solution_idx)
            if np.sum(solution[0:len(self.config_files)]) == 0:
                print("ERROR 0 clfs")
                return 0
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
        except:
            print("error")
        return gm_mean

    def run_experiment(self, data, fname, max_classifier_count = None):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.filename = fname
        sol_per_pop = self.population
        num_parents_mating = self.num_parents

        self.X_orig, self.y_orig = data


        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        def custom_mutation(offspring, ga_instance):
            print(offspring.shape)
            for chromosome in offspring:
                nonzero_indices = np.where(chromosome[:35] != 0)[0]

                # If there are fewer than n nonzero genes, add more
                #if len(nonzero_indices) < n:
                #    zero_indices = np.where(chromosome == 0)[0]
                #    new_nonzero_indices = np.random.choice(zero_indices, n - len(nonzero_indices), replace=False)
                #    chromosome[new_nonzero_indices] = np.random.uniform(-1.0, 1.0, len(new_nonzero_indices))

                # If there are more than n nonzero genes, set excess to zero
                if len(nonzero_indices) > max_classifier_count:
                    excess_indices = np.random.choice(nonzero_indices, len(nonzero_indices) - max_classifier_count, replace=False)
                    chromosome[excess_indices] = 0

                if np.random.rand() <= ga_instance.mutation_probability:
                    num_genes_to_mutate = max(1, int((ga_instance.mutation_percent_genes / 100) * (len(chromosome) - 35)))
                    mutation_indices = np.random.choice(range(35, len(chromosome)), num_genes_to_mutate, replace=False)

                    # Apply mutation to selected genes as integers
                    for i in mutation_indices:
                        if gene_type[i - 35] == int:
                            chromosome[i] = np.random.randint(ga_instance.gene_space[i - 35]["low"], ga_instance.gene_space[i - 35]["high"] + 1)


            print(offspring.shape)
            return offspring

        def custom_initial_population(gene_space, num_parents_mating, max_classifier_count):
            print('CUSTOM POPULATION')
            population = []
            m = len(gene_space)
            for _ in range(num_parents_mating):
                chromosome = np.zeros(m, dtype=int)

                # Initialize the first 35 genes with exactly n nonzero integer values
                nonzero_indices = np.random.choice(35, max_classifier_count, replace=False)
                chromosome[nonzero_indices] = np.ones((1,max_classifier_count)) #np.random.randint(low=0, high=2, size=max_classifier_count)

                # Initialize genes from 35 to m according to gene_space and gene_type
                for i in range(35, m):
                    if gene_type[i - 35] == int:
                        chromosome[i] = np.random.randint(gene_space[i - 35]["low"], gene_space[i - 35]["high"] )
                population.append(chromosome)
            return np.array(population)

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
        num_genes = self.cluster_count * 5 + len(self.config_files)
        gene_type = [int] * self.cluster_count * 5 + [int] * len(self.config_files)
        params = [{'low': 0, 'high': 2}] * (self.cluster_count * 5 + len(self.config_files))
        if self.clustering_params is not None:
            if self.clustering_params["type"]=="MS":
                gene_type = [int] * np.sum(self.clustering_params["clusters"]) + [int] * len(self.config_files)
                num_genes = int(np.sum(self.clustering_params["clusters"]) + len(self.config_files))
                params = [{'low': 0, 'high': 2}] * (np.sum(self.clustering_params["clusters"]) + len(self.config_files))
        if exists:
            ga_instance = pygad.load(self.filename)
        else:
            mutation_type= "random"
            initial_population  = None
            if max_classifier_count is not None:
                mutation_type = custom_mutation
                initial_population = custom_initial_population(params, num_parents_mating=sol_per_pop,
                                                               max_classifier_count=max_classifier_count)
            ga_instance = pygad.GA(num_generations=self.num_generations,

                                   random_seed=42,
                                   #parallel_processing=['process', 15],
                                   num_parents_mating=num_parents_mating,
                                   initial_population=initial_population,
                                    mutation_type  =mutation_type,
                                   crossover_type="single_point",
                                   parent_selection_type="sss",
                                   fitness_func=self.fitness_func,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   gene_type=gene_type,
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

    def evaluate_experiment(self, data, solution):
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        self.X_orig, self.y_orig = data
        self.train_indices = []
        self.test_indices = []
        for train_index, test_index in kf.split(self.X_orig, self.y_orig):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        # self.fitness_func(None, solution, None)
        result = self.eval_func(None, solution, None)

        return result

    def evaluate_experiment_from_pkl(self, data, filename):
        ga_instance = pygad.load(filename)
        solution = ga_instance.best_solutions[-1]
        #solution[-1] = 500
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

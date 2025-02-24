import os
import time
import numpy as np
from imbalanced_ensemble.metrics import geometric_mean_score
#from imbalanced_ensemble.sampler.over_sampling import ADASYN
from imblearn.over_sampling import  ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from pygad import pygad
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from base_functions import get_classifier
from constants import Classifier, genes_weighted_svc, genes_balanced_cascade, genes_svc, genes_adacost, \
    genes_self_paced, tabnet_gene_types, tabnet_gene_space

seed = 42
pygad.random.seed(42)


class GaTuner:

    def __init__(self, num_generations, num_parents=10, population=20, use_smote=True,  clf_type = None,
                 numerical_cols = None, categorical_cols = None, k_neighbors = 2, use_adasyn=False, save_partial_output=False):
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.population = population
        self.X_orig = []
        self.y_orig = []
        self.loss_function = None
        self.train_indices = []
        self.test_indices = []
        self.use_smote = use_smote
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.clf_type = clf_type
        self.k_neighbors = k_neighbors
        self.use_adasyn = use_adasyn,
        self.save_partial_output = save_partial_output

    def eval_func(self, ga_instance, solution, solution_idx):
        X, y = self.X_orig.copy(), self.y_orig.copy()
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

        gmeans = []

        true_values = []
        predicted_values = []

        for index, train_index in enumerate(self.train_indices):

            clf = get_classifier(self.clf_type, solution)

            test_index = self.test_indices[index]
            if self.use_smote:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    #('scaler', StandardScaler()),
                    ('smote', SMOTE(random_state=42, k_neighbors=self.k_neighbors)),
                    ('clf', clf)
                ])

            elif self.use_adasyn:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    # ('scaler', StandardScaler()),
                    ('adasyn', ADASYN(random_state=42, n_neighbors=self.k_neighbors, sampling_strategy='all')),
                    ('clf', clf)
                ])

            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    #('scaler', StandardScaler()),
                    ('clf', clf)])
            # Split data
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            try:
                X_preprocessed = preprocessor.fit_transform(X_train)
                pipeline.fit(X_train, y_train)
            except Exception as e:
                print(e)
                return 0, None, None
            y_pred = pipeline.predict(X_valid)
            true_values.append(y_valid)
            predicted_values.append(y_pred)
            gmean = geometric_mean_score(y_valid, y_pred)
            gmeans.append(gmean)
        gm_mean = np.mean(gmeans)
        return gm_mean, true_values, predicted_values

    def fitness_func(self, ga_instance, solution, solution_idx):
        start_time = time.time()
        gm_mean, true_values, predicted_values = self.eval_func(ga_instance, solution, solution_idx)
        t = time.time() - start_time
        print("gmean: {0:.10f}, {1:.2f} seconds".format(gm_mean, t))

        result = {
            'fitness': gm_mean,
            'true_values': np.array(true_values),
            'predicted_values': np.array(predicted_values),
            'solution': np.array(solution)
        }

        arr = self.filename.split("/")
        arr[-1] = "{}_{}".format(gm_mean, arr[-1])
        f = "/".join(arr)
        if self.save_partial_output:
            with open(f + '.txt', 'w') as data:
                data.write(str(result))
        return gm_mean

    def run_experiment(self, data, fname):
        self.filename = fname
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


        if self.clf_type==Classifier.BalancedCascade:
            gene_types = genes_balanced_cascade['types']
            spaces = genes_balanced_cascade['spaces']
        if self.clf_type==Classifier.SVC:
            gene_types = genes_svc['types']
            spaces = genes_svc['spaces']
        if self.clf_type==Classifier.WeightedSVC:
            gene_types = genes_weighted_svc['types']
            spaces = genes_weighted_svc['spaces']
        if self.clf_type ==Classifier.AdaCost:
            gene_types = genes_adacost['types']
            spaces = genes_adacost['spaces']
        if self.clf_type ==Classifier.SelfPaced:
            gene_types = genes_self_paced['types']
            spaces = genes_self_paced['spaces']
        if self.clf_type ==Classifier.TabNet:
            gene_types = tabnet_gene_types
            spaces = tabnet_gene_space


        filename = fname
        sol_per_pop = self.population
        num_parents_mating = self.num_parents

        self.X_orig, self.y_orig = data

        #params = get_gene_type_and_space(loss_function)
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
                'true_values': np.array(true_values),
                'predicted_values': np.array(predicted_values)
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
                                   num_genes=len(gene_types),
                                   gene_type=gene_types,
                                   save_best_solutions=True,
                                   mutation_probability=0.1,
                                   save_solutions=True,
                                   mutation_percent_genes=0.1,
                                   gene_space=spaces,
                                   on_stop=on_stop,
                                   # on_fitness=callback_fitness,
                                   on_generation=callback_generation)

        ga_instance.run()
        return

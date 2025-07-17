from enum import Enum

RANDOM_STATE = 42
CLUSTER_COUNT = 800 #60
WEAK_CLASSIFIERS_COUNT = 35 #55
#WEAK_CLASSIFIERS_COUNT = 55
SYNTHETIC_MINORITY_COUNT = 1500 #100
SMOTE_K_NEIGHBORS = 3 #6 - vo vsetkych okrem syntetic 03 je 6
class LossFunction(Enum):
    BINARYVSLOSS = "BinaryVsLoss"
    FOCALLOSS = "FocalLoss"
    IBFOCALLOSS = "IBFocalLoss"
    IBLOSS = "IBLOSS"
    LDAMLOSS = "LDAMLoss"
    VSLOSS = "VSLoss"
    VSLOSSMDR = "VSLossMDR"
    LDAMLOSSMDR = "LDAMLossMDR"
    IBLOSSMDR = "IBLossMDR"
    BINARYVSLOSSMDR = "BinaryVsLossMDR"
    CROSSENTROPYLOSS = "CROSSENTROPYLoss"


class Classifier:
    SVC = "SVC"
    WeightedSVC = "WeightedSVC"
    BalancedCascade = "BCascade"
    AdaCost = "AdaCost"
    SelfPaced = "SelfPaced"
    FTTransformer = "FTTransformer"
    TabNet = "TabNet"
    EasyEnsemble = "EasyEnsemble"


tabnet_gene_types = [int, int, int, float, float, float, int, int]
tabnet_gene_space = [
    {'low': 8, 'high': 64},  # n_d
    {'low': 8, 'high': 64},  # n_a
    {'low': 1, 'high': 1},  # n_steps
    {'low': 1, 'high': 5},  # gamma
    {'low': 1e-4, 'high': 1e-2},  # lambda
    {'low': 0.01, 'high': 0.8},  # momentum
    {'low': 1, 'high': 5},  # n_shared
    {'low': 1, 'high': 5},  # n_independent
]

boosting_tabnet_gene_types = tabnet_gene_types + [int]
boosting_tabnet_gene_space = tabnet_gene_space + [
    {'low': 3, 'high': 6},  # n_estimators
]

genes_binary_vs_loss = {
    "types": tabnet_gene_types + [float, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': -5, 'high': 5},  # iota_neg
        {'low': -5, 'high': 5},  # iota_pos
        {'low': -5, 'high': 5},  # delta_neg
        {'low': -5, 'high': 5},  # delta_pos
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
    ]
}

genes_binary_vs_loss_mdr = {
    "types": tabnet_gene_types + [float, float, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': -5, 'high': 5},  # iota_neg
        {'low': -5, 'high': 5},  # iota_pos
        {'low': -5, 'high': 5},  # delta_neg
        {'low': -5, 'high': 5},  # delta_pos
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 0, 'high': 5},  #lambda
    ]
}

genes_focal_loss = {
    "types": tabnet_gene_types + [float, float, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 0, 'high': 5}  # gamma
    ]
}

genes_ib_focal_loss = {
    "types": tabnet_gene_types + [float, float, int, float, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 5},  # epsilon
        {'low': 0, 'high': 5}  # gamma
    ]
}

genes_ib_loss = {
    "types": tabnet_gene_types + [float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 1},  # epsilon 0.001

    ]
}

genes_ib_loss_mdr = {
    "types": tabnet_gene_types + [float, float, int, float, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 1},  # epsilon 0.001
        {'low': 0, 'high': 100}  # lambda
    ]
}

genes_ldam_loss = {
    "types": tabnet_gene_types + [float, float, float, int],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 20},  # max_m
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 10, 'high': 100},  # s
    ]
}

genes_ldam_loss_mdr = {
    "types": tabnet_gene_types + [float, float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 20},  # max_m
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 10, 'high': 100},  # s
        {'low': 0, 'high': 50},  # lambda
    ]
}

genes_vs_loss = {
    "types": tabnet_gene_types + [float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # gamma
        {'low': 0, 'high': 5},  # tau
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
    ]
}

genes_vs_loss_mdr = {
    "types": tabnet_gene_types + [float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # gamma
        {'low': 0, 'high': 5},  # tau
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1,
        {'low': 0, 'high': 5},  # lambda
    ]
}

genes_crossentropy_loss = {
    "types": tabnet_gene_types + [float, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500}  # weight1
    ]
}

genes_boosting_binary_vs_loss = {
    "types": tabnet_gene_types + [int, float, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': -5, 'high': 5},  # iota_neg
        {'low': -5, 'high': 5},  # iota_pos
        {'low': -5, 'high': 5},  # delta_neg
        {'low': -5, 'high': 5},  # delta_pos
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
    ]
}

genes_boosting_binary_vs_loss_mdr = {
    "types": tabnet_gene_types + [int, float, float, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': -5, 'high': 5},  # iota_neg
        {'low': -5, 'high': 5},  # iota_pos
        {'low': -5, 'high': 5},  # delta_neg
        {'low': -5, 'high': 5},  # delta_pos
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 0, 'high': 5},  #lambda
    ]
}

genes_boosting_focal_loss = {
    "types": tabnet_gene_types + [int, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 0, 'high': 5}  # gamma
    ]
}

genes_boosting_ib_focal_loss = {
    "types": tabnet_gene_types + [int, float, float, int, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 5},  # epsilon
        {'low': 0, 'high': 5}  # gamma
    ]
}

genes_boosting_ib_loss = {
    "types": tabnet_gene_types + [int, float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 1},  # epsilon 0.001

    ]
}

genes_boosting_ib_loss_mdr = {
    "types": tabnet_gene_types + [int, float, float, int, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},        # n_estimators
        {'low': 0, 'high': 5},         # weight0
        {'low': 0, 'high': 500},       # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 1},         # epsilon 0.001
        {'low': 0, 'high': 100}        # lambda
    ]
}

genes_boosting_ldam_loss = {
    "types": tabnet_gene_types + [int, float, float, float, int],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},    # n_estimators
        {'low': 0, 'high': 20},    # max_m
        {'low': 0, 'high': 5},     # weight0
        {'low': 0, 'high': 500},   # weight1
        {'low': 10, 'high': 100},  # s
    ]
}

genes_boosting_ldam_loss_mdr = {
    "types": tabnet_gene_types + [int, float, float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},    # n_estimators
        {'low': 0, 'high': 20},    # max_m
        {'low': 0, 'high': 5},     # weight0
        {'low': 0, 'high': 500},   # weight1
        {'low': 10, 'high': 100},  # s
        {'low': 0, 'high': 50},    # lambda
    ]
}

genes_boosting_vs_loss = {
    "types": tabnet_gene_types + [int, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},   # n_estimators
        {'low': 0, 'high': 5},    # gamma
        {'low': 0, 'high': 5},    # tau
        {'low': 0, 'high': 5},    # weight0
        {'low': 0, 'high': 500},  # weight1
    ]
}

genes_boosting_vs_loss_mdr = {
    "types": tabnet_gene_types + [int, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},   # n_estimators
        {'low': 0, 'high': 5},    # gamma
        {'low': 0, 'high': 5},    # tau
        {'low': 0, 'high': 5},    # weight0
        {'low': 0, 'high': 500},  # weight1,
        {'low': 0, 'high': 5},    # lambda
    ]
}

genes_boosting_crossentropy_loss = {
    "types": tabnet_gene_types + [int, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},   # n_estimators
        {'low': 0, 'high': 1},    # p
        {'low': 0, 'high': 1},    # alpha
        {'low': 0, 'high': 1},    # beta
        {'low': 0, 'high': 5},    # weight0
        {'low': 0, 'high': 500}   # weight1
    ]
}

genes_oc_bagging_crossentropy_loss = {
    "types": tabnet_gene_types + [int, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 20},  # n_estimators
        {'low': 0, 'high': 5},   # weight0
        {'low': 0, 'high': 500}  # weight1
    ]
}

genes_bagging_crossentropy_loss = {
    "types": tabnet_gene_types + [int, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 20},  # n_estimators
        {'low': 0, 'high': 5},   # weight0
        {'low': 0, 'high': 500}  # weight1
    ]
}

genes_svc = {
    "types": [float, float],
    "spaces": [
        {'low': 0.001, 'high': 100},   # gamma
        {'low': 0.01, 'high': 100}     # C
    ]
}

genes_weighted_svc = {
    "types": [float, float, int, int],
    "spaces": [
        {'low': 0.001, 'high': 100},   # gamma
        {'low': 0.01, 'high': 100},    # C
        {'low': 1, 'high': 100},       # class 0 weight
        {'low': 1, 'high': 100}        # class 1 weight
    ]
}

genes_balanced_cascade = {
    "types": [int, int, int, float],
    "spaces": [
        {'low': 10, 'high': 500},
        {'low': 0, 'high': 2},
        {'low': 0, 'high': 2},
        {'low': 0.1, 'high': 0.9}
    ]
}

genes_adacost = {
    "types": [int, float, int],
    "spaces": [
        {'low': 10, 'high': 500},
        {'low': 0.001, 'high': 1},
        {'low': 0, 'high': 2}
    ]
}

genes_self_paced = {
    "types":[int, int, int, float],
    "spaces": [
        {'low': 10, 'high': 500},
        {'low': 0, 'high': 2},
        {'low': 0, 'high': 2},
        {'low': 0.1, 'high': 0.9}

    ]
}

genes_fttransformer = {
    "types":[float, int, int, int, float, float],
    "spaces": [
        {'low': 0.001, 'high': 0.005},
        {'low': 10, 'high': 32},    # dim
        {'low': 1, 'high': 5},      # depth
        {'low': 2, 'high': 16},     # heads
        {'low': 0.1, 'high': 0.3},  # attn_dropout
        {'low': 0.1, 'high': 0.3},  # ff_dropout
    ]
}

genes_easy_ensemble = {
    "types": [int, int, float],
    "spaces": [
        {'low': 10, 'high': 500},     # n_estimators
        {'low': 10, 'high': 50},     # base classifier (AdaBoost): n_estimators
        {'low': 0.01, 'high': 2.0}    # base classifier (AdaBoost): learning_rate
    ]
}

'''

params_imbalanced_ensemble_self_paced = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__base_estimator__criterion": ['gini', 'entropy'],
    "model__base_estimator__splitter": ['best', 'random'],
    "model__base_estimator__ccp_alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
}

params_imbalanced_ensemble_adacost = {
    "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "model__learning_rate": [0.001, 0.01, 0.1, 1],
    "model__algorithm": ['SAMME', 'SAMME.R']
}


genes_crossentropy_loss = {
    "types": tabnet_gene_types + [float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 5}  # weight1
    ]
}
'''
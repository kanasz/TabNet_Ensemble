from enum import Enum

RANDOM_STATE = 42


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


tabnet_gene_types = [int, int, int, float, float, float, int, int, float]
tabnet_gene_space = [
    {'low': 8, 'high': 64},  # n_d
    {'low': 8, 'high': 64},  # n_a
    {'low': 1, 'high': 1},  # n_steps
    {'low': 1, 'high': 5},  # gamma
    {'low': 1e-4, 'high': 1e-2},  # lambda
    {'low': 0.01, 'high': 0.8},  # momentum
    {'low': 1, 'high': 5},  # n_shared
    {'low': 1, 'high': 5},  # n_independent
    {'low': 0.001, 'high': 1}  # learning_rate
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
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 1000, 'high': 10000},  # alpha
        {'low': 0, 'high': 1},  # epsilon 0.001
        {'low': 0, 'high': 100}  # lambda
    ]
}

genes_boosting_ldam_loss = {
    "types": tabnet_gene_types + [int, float, float, float, int],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 20},  # max_m
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 10, 'high': 100},  # s
    ]
}

genes_boosting_ldam_loss_mdr = {
    "types": tabnet_gene_types + [int, float, float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 20},  # max_m
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
        {'low': 10, 'high': 100},  # s
        {'low': 0, 'high': 50},  # lambda
    ]
}

genes_boosting_vs_loss = {
    "types": tabnet_gene_types + [int, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # gamma
        {'low': 0, 'high': 5},  # tau
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1
    ]
}

genes_boosting_vs_loss_mdr = {
    "types": tabnet_gene_types + [int, float, float, float, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # gamma
        {'low': 0, 'high': 5},  # tau
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500},  # weight1,
        {'low': 0, 'high': 5},  # lambda
    ]
}

genes_boosting_crossentropy_loss = {
    "types": tabnet_gene_types + [int, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 10},  # n_estimators
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500}  # weight1
    ]
}

genes_bagging_crossentropy_loss = {
    "types": tabnet_gene_types + [int, float, float],
    "space": tabnet_gene_space + [
        {'low': 1, 'high': 20},  # n_estimators
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 500}  # weight1
    ]
}
'''
genes_crossentropy_loss = {
    "types": tabnet_gene_types + [float, float, int, float],
    "space": tabnet_gene_space + [
        {'low': 0, 'high': 5},  # weight0
        {'low': 0, 'high': 5}  # weight1
    ]
}
'''

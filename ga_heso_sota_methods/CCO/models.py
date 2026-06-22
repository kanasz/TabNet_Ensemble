from utils import *


def model_train(net, optimizer, criterion, train_loader, test_loader, epochs):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    bacc = 0
    mcc_best = 0
    f1_score_best = 0
    gmean_best = 0
    auc_best = 0
    epoch_best = 0

    for ep in range(epochs):
        model, balanced_train = train(net, optimizer, criterion, train_loader, device)
        balanced_test, mcc, f1_scores, gmean, auc = test(ep, model, test_loader, device)
        if balanced_test > bacc:
            bacc = balanced_test
            mcc_best = mcc
            f1_score_best = f1_scores
            gmean_best = gmean
            auc_best = auc
            epoch_best = ep

    print("Balanced Test", "MCC", "F1_Score", "GMEAN", "AUC", "epoch")
    print(bacc, mcc_best, f1_score_best, gmean_best, auc_best, epoch_best)
    return bacc, mcc_best, f1_score_best, gmean_best, auc_best, model, epoch_best
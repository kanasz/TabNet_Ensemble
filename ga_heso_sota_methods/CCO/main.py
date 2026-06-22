from utils import *
from models import *
import argparse


def main(PATH, state, k, D, t, beta, split_no, batch_size, num_workers, epochs):
    data = load_data(PATH, state)

    ct1 = "split"

    f = open(PATH + "TEST.txt", "a")
    counter = 0
    f.write("Split," + str(counter) + "," + "," + ", " + "\n")
    f.write("Model,bacc,mcc,f1,gmean,auc\n")

    all_bacc, all_mcc, all_f1, all_gmean, all_auc = [], [], [], [], []

    for i in range(split_no):
        temp = ct1 + str(i)
        X_train, Y_train, X_test, Y_test = data[temp]

        X_train, X_test = scaling(X_train, X_test)

        device = 'cpu'
        X_train = torch.tensor(X_train).to(device)
        X_test = torch.tensor(X_test).to(device)

        # Clustering based on the Cluster Core
        CC = Cluster(X_train, k, D, t, beta)

        # Generation of Synthetic Samples
        X, Y = synthetic_generation(CC, X_train, Y_train, t)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        X = X.to(device)
        Y = Y.to(device)

        input_dim = X_train.shape[1]
        num_classes = len(torch.unique(Y_train))
        assert num_classes == 2, (
            f"Expected binary classification (2 classes), got {num_classes}. "
            "Labels must be 0 and 1."
        )
        net = Net(input_dim, num_classes)
        net = net.to(device)

        num_samples_per_class = []
        ct = Counter(Y_train.cpu().numpy())
        for c in range(len(ct)):
            num_samples_per_class.append(ct[c])

        per_cls_weights = torch.Tensor([1 / n for n in num_samples_per_class]).to(device)
        criterion = FocalLoss(weight=per_cls_weights, gamma=1, reduction='none')
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        train_data = CustomDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        test_data = CustomDataset(X_test, Y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        balanced_test, mcc, f1_scores, gmean, auc, model, epoch_best = model_train(
            net, optimizer, criterion, train_loader, test_loader, epochs
        )
        all_bacc.append(balanced_test)
        all_mcc.append(mcc)
        all_f1.append(f1_scores)
        all_gmean.append(gmean)
        all_auc.append(auc)

        temp = PATH + "model" + str(counter) + ".pt"
        torch.save(model, temp)
        f.write(
            "ours," + str(balanced_test) + "," + str(mcc) + "," +
            str(f1_scores) + "," + str(gmean) + "," + str(auc) + "," + str(epoch_best) + "\n"
        )
        counter += 1
        f.write("Split," + str(counter) + "," + "," + ", " + "\n")
        f.write("Model,bacc,mcc,f1,gmean,auc\n")
        counter += 1

    f.write("Summary,mean+std\n")
    f.write("Model,bacc,mcc,f1,gmean,auc\n")
    means = [np.mean(all_bacc), np.mean(all_mcc), np.mean(all_f1), np.mean(all_gmean), np.mean(all_auc)]
    stds = [np.std(all_bacc), np.std(all_mcc), np.std(all_f1), np.std(all_gmean), np.std(all_auc)]
    f.write("mean," + ",".join([str(m) for m in means]) + "\n")
    f.write("std," + ",".join([str(s) for s in stds]) + "\n")
    print("Summary — mean ± std")
    print("bacc:  {:.4f} ± {:.4f}".format(means[0], stds[0]))
    print("mcc:   {:.4f} ± {:.4f}".format(means[1], stds[1]))
    print("f1:    {:.4f} ± {:.4f}".format(means[2], stds[2]))
    print("gmean: {:.4f} ± {:.4f}".format(means[3], stds[3]))
    print("auc:   {:.4f} ± {:.4f}".format(means[4], stds[4]))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main.py', description='oversampling')
    parser.add_argument('-k', '--k', type=float)
    parser.add_argument('-D', '--D', type=float)
    parser.add_argument('-beta', '--beta', type=float)
    parser.add_argument('-t', '--t', type=float)
    parser.add_argument('-num_workers', '--num_workers', type=float)
    parser.add_argument('-epochs', '--epochs', type=int)
    parser.add_argument('-batch_size', '--batch_size', type=int)
    parser.add_argument('-state', '--state', type=int)
    parser.add_argument('-split_no', '--split_no', type=int)
    parser.add_argument('-PATH', '--PATH', type=str)

    args = parser.parse_args()
    k = args.k
    D = args.D
    t = args.t
    beta = args.beta
    state = args.state
    split_no = args.split_no
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    PATH = args.PATH
    set_seeds(state, True)
    main(PATH, state, k, D, t, beta, split_no, batch_size, num_workers, epochs)
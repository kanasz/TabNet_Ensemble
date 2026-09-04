from utils import *

def model_train(net,optimizer,criterion,train_loader, test_loader, epochs):

  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'

  temp = 0
  temp_mcc = 0
  temp_f1_score = 0
  temp_gmean = 0


  
  balanced_train = 0
  epoch_train = 0

  balanced_test = 0
  epoch_test = 0
  temp = 0
  bal_acc_train = []
  bal_acc_test = []





  mcc_adam_loss = []

  f1_adam_loss = []
  gmean_adam_loss = []
  bacc = 0
  mcc_best = 0
  f1_score_best = 0
  gmean_best = 0
  epoch_best = 0

  #f = open(PATH + "TEST.txt", "a")
  for ep in range(epochs):
    #print("Checkpoint 4")

    count = 0
    prediction_labels_train = torch.empty(0).to(device)
    groundtruth_labels_train = torch.empty(0).to(device)

    prediction_labels_test = torch.empty(0).to(device)
    groundtruth_labels_test = torch.empty(0).to(device)

    model,balanced_train = train(net, optimizer, criterion, train_loader, device)
    balanced_test, mcc, f1_scores, gmean = test(ep, model, test_loader,device)
    if balanced_test>bacc:
      bacc = balanced_test
      mcc_best = mcc
      f1_score_best = f1_scores
      gmean_best = gmean
      epoch_best = ep


  print("Balanced Test", "MCC", "F1_Score","GMEAN", "epoch")
  print(bacc, mcc_best, f1_score_best, gmean_best, epoch_best)
  return bacc, mcc_best, f1_score_best, gmean_best, model, epoch_best
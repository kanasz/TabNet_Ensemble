from utils import *
from models import *
import argparse



def main(PATH,state,k,D,t,beta,split_no,batch_size,num_workers,epochs): 
  data = load_data(PATH, state)

  ct1 = "split"

  f = open(PATH + "TEST.txt", "a")
  counter = 0
  f.write("Split,"+str(counter)+","+","+", "+"\n")
  f.write("Model,"+"bacc"+",mcc"+",f1"+",gmean"+"\n")
  
  for i in range(split_no):

    temp = ct1 + str(i)

    X_train,Y_train,X_test,Y_test = data[temp]
    

    X_train, X_test = scaling(X_train, X_test)

    
    device = 'cpu'

    X_train = torch.tensor(X_train).to(device)
    X_test = torch.tensor(X_test).to(device)

    #Clustering based on the Cluster Core
    CC = Cluster(X_train, k, D, t, beta)

    #Generation of Synthetic Samples
    X,Y = synthetic_generation(CC,X_train,Y_train,t)

    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'


    X_test = X_test.to(device)
    Y_test = Y_test.to(device)


    X = X.to(device)
    Y = Y.to(device)

    net = Net()
    net = net.to(device)  
    num_samples_per_class = []
    ct = Counter(Y_train.cpu().numpy())

    for c in range(len(ct)):
      num_samples_per_class.append(ct[c])


    per_cls_weights = torch.Tensor([1/n for n in num_samples_per_class]).to(device)
    criterion = FocalLoss(weight=per_cls_weights, gamma=1, reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #print("Checkpoint 3")

    train_data = CustomDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)#,num_workers=num_workers)

    test_data = CustomDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)#,num_workers=num_workers)

    
    balanced_test, mcc, f1_scores, gmean, model, epoch_best = model_train(net,optimizer,criterion,train_loader, test_loader, epochs)
    temp = PATH +"model"+str(counter)+".pt"
    torch.save(model,temp)
    f.write("ours,"+str(balanced_test)+","+str(mcc)+","+str(f1_scores)+","+str(gmean)+","+ str(epoch_best) +"\n")
    counter += 1
    f.write("Split,"+str(counter)+","+","+", "+"\n")
    f.write("Model,"+"bacc"+",mcc"+",f1"+",gmean"+"\n")
    counter += 1
  f.close()
    


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
                    prog='main.py',
                    description='oversampling')   # positional argument
          
  parser.add_argument('-k', '--k', type=float)      # option that takes a value k
  parser.add_argument('-D', '--D', type = float)      # option that takes a value D
  parser.add_argument('-beta', '--beta', type = float)      # option that takes a value beta
  parser.add_argument('-t', '--t', type = float)      # option that takes a value temperature t
  parser.add_argument('-num_workers', '--num_workers', type = float)      # option that takes a value number of workers for computation
  parser.add_argument('-epochs', '--epochs', type = int)      # option that takes a value as the number of epochs
  parser.add_argument('-batch_size', '--batch_size', type = int)    #option that takes value the batch size
  parser.add_argument('-state', '--state', type = int)      #option that takes value the random seed value as input
  parser.add_argument('-split_no', '--split_no', type = int)    #option that takes value the number of splits for performing the stratified-k-fold cross-validation
  parser.add_argument('-PATH', '--PATH', type = str)    #option that takes value a path for storing the output of the experiments


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
  main(PATH,state,k,D,t,beta,split_no,batch_size,num_workers,epochs)
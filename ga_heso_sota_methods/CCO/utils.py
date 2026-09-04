import torchvision
from torchvision import transforms, datasets
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.optim as optim
import numpy as np
import torch




import torchvision
from torchvision import transforms, datasets
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from sklearn.cluster import KMeans


#from __future__ import division
import numpy as np
import sklearn
import imblearn
import pandas as pd
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
import numpy as np
import math
import random
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean, stdev
import imblearn.datasets as dt
import torch
import math
from sklearn.preprocessing import MinMaxScaler
import os




def set_seeds(seed_value, use_cuda):
  np.random.seed(seed_value)  # cpu vars
  torch.manual_seed(seed_value)  # cpu  vars
  random.seed(seed_value)  # Python
  os.environ['PYTHONHASHSEED'] = str(seed_value) 
  if use_cuda:
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)  # gpu vars
      torch.backends.cudnn.deterministic = True  # needed
      torch.backends.cudnn.benchmark = False

def load_data(PATH, state):
  
  device = 'cpu'
  PATHX = PATH + "X"
  PATHY = PATH + "Y"
  X = torch.load(PATHX)
  Y = torch.load(PATHY)
  # data = pd.read_csv(PATH1,header = None)
  # X, Y = data.values[:,:-1], data.values[:,-1]
  # Y = np.array(Y)
  # Y = np.array(Y)
  # Y = Y.astype(int)
  # X = X.astype(float)

  scaler = MinMaxScaler()
  scaler = scaler.fit(X.detach().numpy())
  X = scaler.transform(X)
  X = torch.tensor(X)



  # print(len(X))
  # print(type(Y[0:440]))
  # print(Counter(Y[0:440]))

  data_X = torch.tensor(X).float()
  data_Y = torch.tensor(Y).float()


  # data_X = torch.load(PATH + "X").to(device).float()
  # data_Y = torch.load(PATH + "Y").to(device).float()

  # Create StratifiedKFold object.
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=state)
  ct1 = "split"
  counter = 0
  data = {}

  for train_index, test_index in skf.split(data_X.cpu().detach().numpy(), data_Y.cpu().detach().numpy()):
    X_train, X_test = data_X[train_index], data_X[test_index]
    Y_train, Y_test = data_Y[train_index], data_Y[test_index]

    X_train, X_test = X_train.to(device), X_test.to(device)
    Y_train, Y_test = Y_train.to(device), Y_test.to(device)
    temp = ct1 + str(counter)

    data[temp] = (X_train,Y_train, X_test, Y_test)

    counter += 1

  return data

def focal_loss(input_values, gamma):
  """Computes the focal loss"""
  p = torch.exp(-input_values)
  loss = (1 - p) ** gamma * input_values
  return loss.mean()

class FocalLoss(nn.Module):
  """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
  def __init__(self, weight=None, gamma=0., reduction='mean'):
      super(FocalLoss, self).__init__()
      assert gamma >= 0
      self.gamma = gamma
      self.weight = weight
      self.reduction = reduction

  def forward(self, input, target):
      return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)


def gen_points(train, gen_no,t):

    flag = 0

    size = len(train)
    X_gen = 0
    for i in range(gen_no):
        weights = nn.Parameter(torch.rand(1,size))
        # weights_normalized = nn.functional.softmax(weights, dim=-1)
        weights_normalized1 = nn.functional.normalize(weights, dim=-1)
        #t = 0.93
        #t = 0.5499999999999999
        #t = 0.499
        
        weights_normalized = nn.functional.softmax(weights_normalized1/t, dim=-1)     
        result = weights_normalized@train
        if flag == 0:
            X_gen = result
            flag = 1
        else:
            X_gen = torch.cat((X_gen,result), axis = 0)
    X_gen = torch.cat((X_gen,train), axis = 0)
    print(X_gen.shape)

    return X_gen

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(34, 20)
    self.fc2 = nn.Linear(20, 12)
    self.fc3 = nn.Linear(12, 6)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x



  
class CustomDataset(Dataset):
  def __init__(self, pred, y):
      self.data = []
      predictor = pred
      response = y
      for i in range(len(predictor)): 
        self.data.append([predictor[i],response[i]])
  def __len__(self):
      return len(self.data)
  def __getitem__(self, idx):
      data_instance, class_name = self.data[idx]
      return data_instance, class_name
  
def min_max(count):
  min = 0
  max = 0
  for i in count:
    if count[min]>count[i]:
      min = i
    if count[max]<count[i]:
      max = i
  return max,min

def scaling(X_train,X_test):
  scaler = MinMaxScaler()
  # scaler = scaler.fit(X_train.cpu().detach().numpy())
  # X_train = scaler.transform(X_train.cpu().detach().numpy())
  scaler = scaler.fit(X_train.cpu().detach().numpy())
  X_train = scaler.transform(X_train.cpu().detach().numpy())
  X_test = scaler.transform(X_test.cpu().detach().numpy())
  return X_train, X_test



def Cluster_Kmeans(X_train):

  kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_train)
  CC = kmeans.labels_

  return CC


def Cluster(X_train,k,D,t,beta):
  device = 'cpu'
  distance = torch.cdist(X_train,X_train)

  # print(distance)

  distance_mask = (distance > k).to(device)
  Radius = torch.mul(distance_mask,distance).to(device)
  Radius[Radius == 0] = float("Inf")
  Radius = Radius.min(axis=1)
  Radius_values = Radius.values
  print("RADIUS",Radius.values)


  Radius_position = Radius.indices
  print("RADIUS INDEX",Radius_position)


  v_d = 4*math.pi/3
  absolute1 = math.pow(torch.pi,(D/2))
  stranger = torch.tensor([D/2])
  absolute2 = torch.lgamma(stranger + 1)
  #print("Hey 1")
  #print(absolute2[0])
  absolute2 = torch.exp(absolute2[0])
  #print("Hey 2:", k)
  absolute = absolute1/absolute2
  #print(absolute)
  v_d = absolute
  d = D
  n = len(X_train)
  density = k/(n*v_d*pow(Radius_values,d)).to(device)


  CC = [-1 for i in range(n)]
  CC = torch.tensor(CC).to(device)


  #
  count = 0
  while True:
    index = torch.where(CC == -1)[0].to(device)
    #print(index)
    if len(index) < 100 or count > 100:
      break
    
    f_x = torch.max(density[index]).to(device)
    
    a = density[index] > beta*f_x
    b = (a*index).to(device)
    
    CC[b] = count
    count += 1
  cluster_no = torch.unique(CC).to(device)
  CC1 = CC
  unassigned = torch.where(CC == -1)[0].to(device)
  a = []
  print(Counter(CC.cpu().numpy()))
  print(torch.unique(CC).to(device))
  print(cluster_no)
  for i in range(len(cluster_no)-1):
    pos = torch.where(CC == i)[0]
    #print(len(pos))
    mean = torch.mean(X_train[pos],0).to(device)
    a.append(mean)
    if i == 0:
      c = mean
      c = torch.reshape(c,(1,len(c)))
    else:
      mean = torch.reshape(mean,(1,len(mean)))
      #print(c.shape)
      c = torch.cat((c,mean), axis = 0)
  if len(cluster_no) > 1:
    X_unassigned = X_train[unassigned]
    X_classmean = c.clone().detach()

    dist = torch.cdist(X_unassigned,X_classmean,p=2)
    closest_dist = torch.min(dist, 1)
    index = closest_dist.indices
    CC[unassigned] = index
    No = torch.unique(CC)
    #Directed graph G with vertices 1,2,............,n
    A = torch.zeros(n, n)
    for i in No:
      idx = torch.where(CC == i)[0]
      flag = 0
      for i1 in range(len(idx)):
        d = density[i1]
        position = Radius_position[i1]
        den = density[position]
        if den > d:
          A[i1][position] = 1

    cluster_no = torch.unique(CC)
    for i in cluster_no:
      index = torch.where(CC == i)[0]

    b1 = torch.sum(A, dim=0)#sum of each column
    b2 = torch.sum(A, dim=1)#sum of each row
    ct = 0
    unassigned = []
    for i in range(n):
      if b1[i] == b2[i] and b1[i] == 0:
        ct += 1
        unassigned.append(i)

    unassigned = torch.tensor(unassigned)

    CC2 = CC.clone().detach().to(device)
    CC2[unassigned] = -1

    cluster_no = torch.unique(CC2)
    for i in cluster_no:
      index = torch.where(CC2 == i)[0]
      print("Cluster No",i,"contains:",len(index),"elements")

    #mean calculations for each clusters
    a = []
    for i in range(len(No)):
      pos = torch.where(CC2 == i)[0]
      #print(len(pos))
      mean = torch.mean(X_train[pos],0)
      a.append(mean)
      if i == 0:
        c = mean
        c = torch.reshape(c,(1,len(c))).to(device)
      else:
        mean = torch.reshape(mean,(1,len(mean)))
        c = torch.cat((c,mean), axis = 0)

    X_unassigned = X_train[unassigned]
    X_classmean = c.clone().detach()

    dist = torch.cdist(X_unassigned,X_classmean,p=2)
    closest_dist = torch.min(dist, 1)
    index = closest_dist.indices#.to(device)


    CC2[unassigned] = index#.to(device)


  else:

    CC2 = CC.clone().detach()
  
  return CC2



def synthetic_generation(CC,X_train,Y_train,t):

  No = torch.unique(CC)
  device = 'cpu'

  #After assignment
  a = Counter(CC.cpu().numpy())
  epsilon = 1
  X = torch.zeros(1,74).to(device)
  Y = torch.zeros(1).to(device)
  flag = 0

  for c in No:
    index_c = torch.where(CC == c)[0]
    x_c = X_train[index_c]
    y_c = Y_train[index_c]
    count_y_c = Counter(y_c.cpu().numpy())
    majority_class, minority_class = min_max(count_y_c)

    for cls in count_y_c:
      ratio_i = count_y_c[cls]/ count_y_c[majority_class]

      if ratio_i >= epsilon:
        if flag == 0:
          X = x_c[torch.where(y_c == cls)[0]].to(device)
          Y = y_c[torch.where(y_c == cls)[0]].to(device)
          flag = 1
        else:
          X = torch.cat((X,x_c[torch.where(y_c == cls)[0]].to(device)), axis=0)
          Y = torch.cat((Y,y_c[torch.where(y_c == cls)[0]].to(device)), axis=0)
      
      if ratio_i < epsilon:
        x1_temp = x_c[torch.where(y_c == cls)[0]]
        y1_temp = y_c[torch.where(y_c == cls)[0]]
        x2_temp = x_c[torch.where(y_c == majority_class)[0]]
        y2_temp = y_c[torch.where(y_c == majority_class)[0]]
        x = torch.cat((x1_temp,x2_temp), axis = 0)
        y = torch.cat((y1_temp,y2_temp), axis = 0)
        if len(y1_temp) < 5:
          X1 = torch.clone(x1_temp)
          Y1 = torch.clone(y1_temp)
        else:
          gen_no = len(y2_temp) - len(y1_temp)
          X1 = gen_points(x1_temp,gen_no,t)
          perm = torch.randperm(len(X1))
          Y1 = torch.ones(len(X1))
          Y1 = Y1.float()
          c = torch.tensor(cls)
          Y1[:] = c
          X1 = X1[perm]
          Y1 = Y1[perm]
        if flag == 0:
          X = X1.to(device)
          Y = Y1.to(device)
          flag = 1
        else:
          X = torch.cat((X,X1.to(device)), axis = 0)
          Y = torch.cat((Y,Y1.to(device)), axis = 0)
  return  X,Y

def train(model, optimizer,criterion, train_loader, device=None):
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  count = 0
  prediction_labels_train = torch.empty(0).to(device)
  groundtruth_labels_train = torch.empty(0).to(device)
  balanced_train = 0
  for i, data in enumerate(train_loader, 0):
    
    inputs, train_labels = data
    inputs, train_labels = inputs.to(device), train_labels.to(device)
    optimizer.zero_grad()
    train_outputs = model(inputs)
    train_labels = train_labels.type(torch.int64)
    loss = criterion(train_outputs, train_labels)
    loss.backward(retain_graph=True)
    optimizer.step()
    prediction_labels_train = torch.cat((prediction_labels_train, torch.tensor([x.argmax() for x in train_outputs.detach()]).to(device)), -1)
    groundtruth_labels_train = torch.cat((groundtruth_labels_train, train_labels), 0)
    pred = torch.tensor([x.argmax() for x in train_outputs.detach()])
  temp = balanced_accuracy_score(groundtruth_labels_train.tolist(),prediction_labels_train.tolist())
  if balanced_train<temp:
    balanced_train = temp
  
  return model, balanced_train


def test(ep,model, test_loader, device=None):
  correct =0
  total =0
  prediction_labels_test = torch.empty(0).to(device)
  groundtruth_labels_test = torch.empty(0).to(device)
  mcc = 0
  f1_scores = 0
  gmean = 0
  balanced_test = 0
  for j, data in enumerate(test_loader, 0):
    images,test_labels = data
    images,test_labels = images.to(device),test_labels.to(device)
    test_outputs=model(images.float())
    _,predicted=torch.max(test_outputs.data,1)
    prediction_labels_test = torch.cat((prediction_labels_test, torch.tensor([x.argmax() for x in test_outputs.detach()]).to(device)), -1)
    groundtruth_labels_test = torch.cat((groundtruth_labels_test, test_labels), 0)
    total+=test_labels.size(0)
    correct += (predicted==test_labels).sum()
  accuracy = 100 *correct/total
  print("Epochs: {}, Test Accuracy:{}%".format(ep ,accuracy))

  balanced_test = balanced_accuracy_score(groundtruth_labels_test.tolist(),prediction_labels_test.tolist())
  gt = groundtruth_labels_test.cpu().numpy().astype(int)
  p = prediction_labels_test.cpu().numpy().astype(int)
  mcc = sklearn.metrics.matthews_corrcoef(gt, p)
  f1_scores = sklearn.metrics.f1_score(gt,p, average='weighted')
  gmean = imblearn.metrics.geometric_mean_score(gt.tolist(),p.tolist())
  
  return balanced_test, mcc, f1_scores, gmean

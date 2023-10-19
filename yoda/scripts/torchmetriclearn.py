from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(32)
### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
# from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv1 = nn.Conv2d(17, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(602176, 9216)
        self.fc2 = nn.Linear(9216, 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # NEW
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #print(f"{x.shape}")
        x = torch.flatten(x, 1)
        #print(f"{x.shape}")
        x = self.fc1(x) # NEW
        x = self.fc2(x)
        x = self.fc3(x)
        return x


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
import time
timestart = time.time()
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()

        if batch_idx % 4 == 0:
            print( "Ep {} Iter {} Loss {}, triplets {}, time {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets,time.time()-timestart ))


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
from sklearn.cluster import KMeans
def test(train_set, test_set, model, accuracy_calculator):
    # train_embeddings, train_labels = get_all_embeddings(train_set, model)
    with torch.no_grad(): # not sure if pml does this
        test_embeddings, test_labels = get_all_embeddings(test_set, model)
    # train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    silhou = silhouette_score(test_embeddings.numpy(),  test_labels)
    print(f"{silhou= }")
    ari = adjusted_rand_score( KMeans(n_clusters=len(np.unique(test_labels))).fit_predict(test_embeddings.numpy()), test_labels)
    print(f"{ ari=}")
    return silhou, ari

    # print("Computing accuracy")
    # accuracies = accuracy_calculator.get_accuracy(
    #     test_embeddings, test_labels, train_embeddings, train_labels, False
    # )
    # print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


device = torch.device("cpu")

# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )

batch_size = 96

# dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
# dataset2 = datasets.MNIST(".", train=False, transform=transform)
# train_loader = torch.utils.data.DataLoader(
#     dataset1, batch_size=batch_size, shuffle=True
# )
# test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

from yoda.ml.nnembedder import torchloader
from yoda import alignments

def getdata():
    a,l = alignments.load_rfam(add_cov='')
    a,l = alignments.size_filter(a,l,400)
    alis,labels = alignments.manifest_sequences(a,l,instances=10, mp=True)
    graphs = [a.graph for a in alis]
    del a
    return graphs, labels

import structout as so
def printres(res):
    s,a = Transpose(res)
    print("sil and ari so far...")
    so.lprint(s)
    so.lprint(a)
import ubergauss.tools as ut


if __name__ == '__main__':
    graphs, labels = ut.cache('graphscache.delme',getdata)
    train_loader, dataset1, dataset2 = torchloader(batch_size, graphs,labels)
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100

    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###

    res = []
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        r = test(dataset1, dataset2, model, accuracy_calculator)
        res.append(r)
        printres(res)

    breakpoint()

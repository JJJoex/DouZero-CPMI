from math import gamma
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np
import torch
import csv

class ClassifierModel(nn.Module):
    def __init__(self, input_size, num_classes, tau):
        super().__init__()
        self.h1 = nn.Linear(input_size, 512)
        self.h2 = nn.Linear(512, 128)
        self.h3 = nn.Linear(128, num_classes)
        self.tau = tau
    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.softmax(x, dim=-1)
        hardT = nn.Hardtanh(self.tau, 1-self.tau)
        x = hardT(x)
        return x 


def sample_batch(data, batch_size, sample_mode, neighbor_size):
    if sample_mode == 'joint':
        index = np.random.choice(len(data), batch_size, replace=False)
        # history, last action, current action
        batch = []
        for i in index:
            b = data[i][0] + [data[i][1]] + [data[i][2]]
            batch.extend(b)
        batch = np.array(batch).reshape(-1, 12*54)
    elif sample_mode == 'prod_knn':
        m = len(data) // neighbor_size
        index_yz = np.random.choice(len(data), m, replace=False).tolist()
        Z1 = np.array([np.concatenate(data[i][0]) for i in index_yz])
        neigh = NearestNeighbors(n_neighbors=neighbor_size, metric='euclidean')
        Z2 = np.array([np.concatenate(data[i][0]) for i in range(len(data)) if i not in index_yz])
        neigh.fit(Z2)
        neigh_index = neigh.kneighbors(Z1, return_distance=False).tolist()
        batch = []
        for i in range(len(index_yz)):
            for j in neigh_index[i]:
                b = data[index_yz[i]][0] + [data[index_yz[i]][1]] + [data[j][2]]
                batch.extend(b)
        batch = np.array(batch).reshape(-1, 12*54)
        batch = batch[:batch_size]
    print(f'Sample mode: {sample_mode}, Batch size: {len(batch)}, Neighbors size: {neighbor_size}')
    return batch, len(batch)


def construct_batch(data, set_size, neighbor_size=7):
    n = len(data)
    assert n > set_size
    joint_set, joint_set_size = sample_batch(data, batch_size=set_size, sample_mode='joint', neighbor_size=neighbor_size)
    t1 = int(0.8 * joint_set_size)
    joint_train = joint_set[: t1]
    joint_test = joint_set[t1 :]
    prod_set, prod_set_size = sample_batch(data, batch_size=set_size, sample_mode='prod_knn', neighbor_size=neighbor_size)
    t2 = int(0.8 * prod_set_size)
    prod_train = prod_set[: t2]
    prod_test = prod_set[t2 :]
    batch_train = torch.tensor(np.concatenate((joint_train, prod_train))).float()
    batch_test = torch.tensor(np.concatenate((joint_test, prod_test))).float()
    
    joint_target_set = np.repeat([[1, 0]], joint_set_size, axis=0)
    joint_target = joint_target_set[: t1]
    joint_test = joint_target_set[t1 :]
    prod_target_set = np.repeat([[0, 1]], prod_set_size, axis=0)
    prod_target = prod_target_set[: t2]
    prod_test = prod_target_set[t2 :]
    target_train = np.concatenate((joint_target, prod_target),axis=0)
    target_train = torch.tensor(target_train).float()
    target_test = np.concatenate((joint_test, prod_test),axis=0)
    target_test = torch.tensor(target_test).float()
    print(f'Train size: {len(target_train)}, Test size: {len(target_test)}')
    return batch_train, target_train, batch_test, target_test


def estimate_cpmi(model, data):
    joint_prob = model(data)
    joint_prob, _ = torch.max(joint_prob, dim=1, keepdim=True)
    return torch.log(joint_prob / (1-joint_prob))


def train_classifer(batch_train, target_train, batch_test, target_test, epoch=10000, learning_rate=1e-3, seed=1237, epsilon=1e-7):
    loss_e = []
    last_loss = 1000
    acc = 0
    acc_list = []

    input_size = 12 * 54
    # for state feature as state
    # input_size = 2 * 54 + 128
    num_classes = 2
    tau = 1e-4
    
    torch.manual_seed(seed)
    model = ClassifierModel(input_size, num_classes, tau)    
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    target_test = torch.argmax(target_test, dim=1)

    for i in range(epoch + 1):
        out = model(batch_train)      

        loss = F.binary_cross_entropy(out, target_train) 
        print(f'epoch={i}, loss={loss.detach().numpy()}')
        loss_e.append(loss.detach().numpy())
        
        if i % 100 == 0:
            out_test = model(batch_test)
            out_test = torch.argmax(out_test, dim=1)
            acc = torch.sum(out_test == target_test).item() / len(batch_test)
            print(f'epoch={i}, acc={acc}')
            acc_list.append(acc)

        if abs(loss-last_loss) < epsilon:
            out_test = model(batch_test)
            out_test = torch.argmax(out_test, dim=1)
            acc = torch.sum(out_test == target_test).item() / len(batch_test)
            if acc > 0.80:
                print(print(f'epoch={i}, acc={acc}'))
                break
        
        last_loss = loss
        model.zero_grad()
        loss.backward()
        optimizer.step()
    with open('log/loss_10_7.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'loss'])
            for idx, loss_value in enumerate(loss_e):
                writer.writerow([idx, loss_value])
    with open('log/accuracy_10_7.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'accuracy'])
        for idx, acc in enumerate(acc_list):
            writer.writerow([idx * 100, acc])    
    return model

import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

class Meter(object):
    """Computes and stores the sum, average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_sum(self):
        return self.sum
    
    def value(self):
        return self.sum

class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val

def make_meter_matching():
    meters_dict = {
        'loss': Meter(),
        'acc_la': Meter(),
        'acc_max': Meter(),
        'batch_time': Meter(),
        'data_time': Meter(),
        'epoch_time': Meter(),
    }
    return meters_dict

def accuracy_linear_assigment(weights,labels=None):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    bs = weights.shape[0]
    n = weights.shape[1]
    if labels is None:
        labels = np.stack([np.arange(n) for _ in range(bs)])
    acc = 0
    for i in range(bs):
        cost = -weights[i,:,:]
        _ , preds = linear_sum_assignment(cost)
        acc += np.sum(preds == labels[i,:])
    return acc, n, bs

def accuracy_max(weights,labels=None):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    bs = weights.shape[0]
    n = weights.shape[1]
    if labels is None:
        labels = np.stack([np.arange(n) for _ in range(bs)])
    acc = 0
    for i in range(bs):
        preds = np.argmax(weights[i,:,:], 0)
        #print(preds)
        acc += np.sum(preds == labels[i,:])
    return acc, n, bs

def transform_cycles(batch,num_neighbors=2):
    bs, n_nodes, _ = batch.shape
    adjacency = torch.zeros((bs,n_nodes,n_nodes))
    for i in range(bs):
        W_val = batch[i,:,:].cpu().detach().numpy()
        knns = np.argpartition(W_val, kth=2, axis=-1)[:, num_neighbors::-1]
        for j in range(n_nodes):
            [*u,w] = knns[j,:]
            for v in u:
                adjacency[i,v,w] = 1
                adjacency[i,w,v] = 1
            #adjacency[i,v,w] = 1
            #adjacency[i,w,v] = 1
    return adjacency

def f1_score(preds,labels):
    """
    take 2 adjacency matrices and compute precision, recall, f1_score for a tour
    """
    bs, n_nodes ,_  = labels.shape
    #n = len(preds[0])
    true_pos = 0
    false_pos = 0
    for i in range(bs):
        true_pos += torch.sum(preds[i,:,:]*labels[i,:,:].cpu()).item()
        false_pos += torch.sum(preds[i,:,:]*(1-labels[i,:,:].cpu())).item()
        #pos += np.sum(preds[i][0,:] == labels[i][0,:])
        #pos += np.sum(preds[i][1,:] == labels[i][1,:])
    #prec = pos/2*n
    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/(2*n_nodes*bs)
    return prec, rec, 2*prec*rec/(prec+rec)#, n, bs


def gap_tsp(weights,labels,distances):
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    bs = weights.shape[0]
    n = weights.shape[1]
    diff = 0
    for i in range(bs):
        preds = np.argmax(weights[i,:,:], 0)
        opt_tour = np.sum([distances[i,j,labels[i,j]] for j in range(n)])
        pred_tour = np.sum([distances[i,j,preds[j]] for j in range(n)])
        print(opt_tour,pred_tour)
        diff += pred_tour-opt_tour
    return diff, n, bs
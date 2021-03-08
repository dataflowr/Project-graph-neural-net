#Script to 
# 1) get node embeddings from a graph
# 2) get a (t-SNE) projection from that

from commander import load_model
from loaders.data_generator import Generator
from random import randint
from argparse import ArgumentParser
from models import get_model
import json
import os
import torch
from sklearn.manifold import TSNE
import numpy as np

data_path = "dataset/QAP_steps_ErdosRenyi_100_25_1.0_0.1_0.2/test.pkl"
model_path = "runs/Reg-ER-100/QAP_ErdosRenyi_ErdosRenyi_25_1.0_0.05_0.2"

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings(model, g1, g2):
    """Take g1, g2 1-batches of graph
    returns embeddings of shape (1,n, embed_dim)"""
    embeddings = []
    handle = model.node_embedder.register_forward_hook(lambda module, inp, outp : embeddings.append(outp))
    model(g1,g2)
    handle.remove()
    return embeddings

def get_graphs(data, i, device):
    print(f"Using data point nb {args.i}")
    g1, g2 = data[args.i] #graphs !
    g1.unsqueeze_(0) #batches of 1
    g2.unsqueeze_(0)
    print(g1.shape)
    g1 = g1.to(device)
    g2 = g2.to(device)

    return g1,g2


def embed(g1,g2, model):
    e1, e2 = get_embeddings(model, g1, g2)
    e = torch.cat((e1,e2), axis = 1)
    tsne = TSNE()
    v = tsne.fit_transform(e.cpu().detach().squeeze().numpy())
    _,n,_ = e1.shape
    v1 = v[:n, :]
    v2 = v[n:, :]
    return v1, v2

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", default=None, help="Id of the data point to use. by default, use random")
    parser.add_argument("-m", help="Path to model to load", default = model_path)
    parser.add_argument("-d", help="Path to data to load", default = data_path)

    args = parser.parse_args()
    print("Using data from " + args.d)
    data = list (torch.load(args.d))

    if args.i is None :
        args.i = randint(0, len(data)-1)

    with open(os.path.join(model_path,"config.json")) as reader :
        cfg = json.load(reader)

    model = get_model(cfg["arch"])
    model.eval()
    model.to(device)
    model = load_model(model, device, os.path.join(model_path,"model_best.pth.tar"))
    g1,g2 = get_graphs(data,args.i, device)

    e1, e2 = embed(g1, g2, model)
    #embeddings
    np.save("embeds/g1/embeds", e1)
    np.save("embeds/g2/embeds", e2)
    #original graphs
    np.save("embeds/g1/graph", g1.detach().cpu().numpy())
    np.save("embeds/g2/graph", g2.detach().cpu().numpy())


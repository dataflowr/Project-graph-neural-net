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

data = list (torch.load("dataset/QAP_ErdosRenyi_ErdosRenyi_1000_25_1.0_0.05_0.2/val.pkl"))

model_path = "runs/Reg-ER-100/QAP_ErdosRenyi_ErdosRenyi_25_1.0_0.05_0.2"

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings(model, g1, g2):
    """Take g1, g2 1-batches of graph"""
    embeddings = []
    handle = model.node_embedder.register_forward_hook(lambda module, inp, outp : embeddings.append(outp))
    model(g1,g2)
    handle.remove()
    return embeddings

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", default=randint(0, len(data)-1), help="Id of the data point to use. by default, use random")
    parser.add_argument("-m", help="Path to model to load", default = model_path)
    args = parser.parse_args()
    print(f"Using data point nb {args.i}")
    g1, g2 = data[args.i] #graphs !
    g1.unsqueeze_(0) #batches of 1
    g2.unsqueeze_(0)
    print(g1.shape)
    g1 = g1.to(device)
    g2 = g2.to(device)

    with open(os.path.join(model_path,"config.json")) as reader :
        cfg = json.load(reader)

    model = get_model(cfg["arch"])
    model.eval()
    model.to(device)
    model = load_model(model, device, os.path.join(model_path,"model_best.pth.tar"))
    
    emb, _ = get_embeddings(model, g1, g2)
    print(emb.shape)


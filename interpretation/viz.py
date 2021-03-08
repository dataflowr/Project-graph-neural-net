import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np 
from argparse import ArgumentParser
import os

#graph has the shape (1, n,n, 2) where layer 1 is adjacency
plt.ion()

def create_graph(graph):
    _, n,m, _ = graph.shape
    G = nx.Graph()
    G.add_nodes_from(range(0,n))
    for i in range(n):
        for j in range(m):
            if graph[0,i,j,1]==1:
                G.add_edge(i,j)
    return G

#idea 1 : plot graphs with positions as in the embedding
def draw_pos_embedding(G, embeds):
    plt.figure()
    nx.draw(G, embeds, with_labels=True)
    plt.draw()

def viz_simil(G, M):
    """Draws the graph as given by similarity matrix M"""
    n=len(M)
    color = ["red"]*n//2 + ["blue"]*n//2
    G.set_edge_attributes({(i,j) : {"weight":M[i,j]} for i in range(n) for j in range(n)})
    pos = nx.spring_layout(G, weight="weight" )
    nx.draw(G, pos,with_labels=True, node_color=color )
    plt.draw()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", help="Paths to np graph files", required=True, nargs="+")
    parser.add_argument("-pos", choices=["embeds", "simil"], required=True)

    args = parser.parse_args()

    graphs = []
    embeds = []
    for p in args.f:

        graph = np.load(os.path.join(p, "graph.npy"))
        embeds = np.load(os.path.join(p, "embeds.npy"))
        G = create_graph(graph)

        if args.pos == "embeds":
            draw_pos_embedding(G, embeds)
        if args.pos == "simil":
            M = np.load(os.path.join(p, "simil.npy"))
            viz_simil(G,M)
    input("end")


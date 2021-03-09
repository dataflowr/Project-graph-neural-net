import os
import random
import itertools
import networkx
import torch
import torch.utils
from toolbox import utils

GENERATOR_FUNCTIONS = {}

def generates(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS[name] = func
        return func
    return decorator

@generates("ErdosRenyi")
def generate_erdos_renyi_netx(p, N):
    """ Generate random Erdos Renyi graph """
    g = networkx.erdos_renyi_graph(N, p)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("BarabasiAlbert")
def generate_barabasi_albert_netx(p, N):
    """ Generate random Barabasi Albert graph """
    m = int(p*(N -1)/2)
    g = networkx.barabasi_albert_graph(N, m)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("Regular")
def generate_regular_graph_netx(p, N):
    """ Generate random regular graph """
    d = p * N
    d = int(d)
    # Make sure N * d is even
    if N * d % 2 == 1:
        d += 1
    g = networkx.random_regular_graph(d, N)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("steps")
def generates_steps_network(p,N):
    """Generates a graph with 5 groups of nodes of increasing degree"""
    g = networkx.Graph()
    g.add_nodes_from(range(N))
    a=0
    step = N//5
    for i in range(5):
        for j in range(a, a+step):
            for k in range(0, a+step):
                if random.random() < p:
                    g.add_edge(j,k)
        a+=step
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)

@generates("Symmetric")
def generates_symmetric_netx(p,N):
    "Generates a graph composed of two identical erdos-reyni graphs, so that there are no differences between the clusters"
    assert (N%2==0)
    g1 = networkx.erdos_renyi_graph(N, p)
    g2 = networkx.erdos_renyi_graph(N, p)
    g = networkx.disjoint_union(g1,g2)
    for i in range(N):
        for j in range(N):
            if random.random() < p/3 :
                g.add_edges_from([(i+N,j),(j+N,i)])
    W =  networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float)


NOISE_FUNCTIONS = {}

def noise(name):
    """ Register a noise function """
    def decorator(func):
        NOISE_FUNCTIONS[name] = func
        return func
    return decorator

@noise("ErdosRenyi")
def noise_erdos_renyi(g, W, noise, edge_density):
    n_vertices = len(W)
    pe1 = noise
    pe2 = (edge_density*noise)/(1-edge_density)
    _,noise1 = generate_erdos_renyi_netx(pe1, n_vertices)
    _,noise2 = generate_erdos_renyi_netx(pe2, n_vertices)
    W_noise = W*(1-noise1) + (1-W)*noise2
    return W_noise

def is_swappable(g, u, v, s, t):
    """
    Check whether we can swap
    the edges u,v and s,t
    to get u,t and s,v
    """
    actual_edges = g.has_edge(u, v) and g.has_edge(s, t)
    no_self_loop = (u != t) and (s != v)
    no_parallel_edge = not (g.has_edge(u, t) or g.has_edge(s, v))
    return actual_edges and no_self_loop and no_parallel_edge

def do_swap(g, u, v, s, t):
    g.remove_edge(u, v)
    g.remove_edge(s, t)
    g.add_edge(u, t)
    g.add_edge(s, v)

@noise("EdgeSwap")
def noise_edge_swap(g, W, noise, edge_density):
    g_noise = g.copy()
    edges_iter = list(itertools.chain(iter(g.edges), ((v, u) for (u, v) in g.edges)))
    for u,v in edges_iter:
        if random.random() < noise:
            for s, t in edges_iter:
                if random.random() < noise and is_swappable(g_noise, u, v, s, t):
                    do_swap(g_noise, u, v, s, t)
    W_noise = networkx.adjacency_matrix(g_noise).todense()
    return torch.as_tensor(W_noise, dtype=torch.float)

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples

    def load_dataset(self):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading dataset at {}'.format(path))
            data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset.')
            self.data = []
            self.create_dataset()
            print('Saving datatset at {}'.format(path))
            torch.save(self.data, path)
    
    def create_dataset(self):
        for _ in range(self.num_examples):
            example = self.compute_example()
            self.data.append(example)

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)


class Generator(Base_Generator):
    """
    Build a numpy dataset of pairs of (Graph, noisy Graph)
    """
    def __init__(self, name, args):
        self.generative_model = args['generative_model']
        self.noise_model = args['noise_model']
        self.edge_density = args['edge_density']
        self.noise = args['noise']
        num_examples = args['num_examples_' + name]
        n_vertices = args['n_vertices']
        vertex_proba = args['vertex_proba']
        subfolder_name = 'QAP_{}_{}_{}_{}_{}_{}_{}'.format(self.generative_model,
                                                     self.noise_model,
                                                     num_examples,
                                                     n_vertices, vertex_proba,
                                                     self.noise, self.edge_density)
        path_dataset = os.path.join(args['path_dataset'],
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = (vertex_proba == 1.)
        self.n_vertices_sampler = torch.distributions.Binomial(n_vertices, vertex_proba)
        
        
        utils.check_dir(self.path_dataset)

    def compute_example(self):
        """
        Compute pairs (Adjacency, noisy Adjacency)
        """
        n_vertices = int(self.n_vertices_sampler.sample().item())
        try:
            g, W = GENERATOR_FUNCTIONS[self.generative_model](self.edge_density, n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        try:
            W_noise = NOISE_FUNCTIONS[self.noise_model](g, W, self.noise, self.edge_density)
        except KeyError:
            raise ValueError('Noise model {} not supported'
                             .format(self.noise_model))
        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)
        return (B, B_noise)

    

    

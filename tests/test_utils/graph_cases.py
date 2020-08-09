from collections import defaultdict
import backend as F
import dgl
import numpy as np
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import backend as F

case_registry = defaultdict(list)

def register_case(labels):
    def wrapper(fn):
        for lbl in labels:
            case_registry[lbl].append(fn)
        fn.__labels__ = labels
        return fn
    return wrapper

def get_cases(labels=None, exclude=[]):
    """Get all graph instances of the given labels."""
    cases = set()
    if labels is None:
        # get all the cases
        labels = case_registry.keys()
    for lbl in labels:
        for case in case_registry[lbl]:
            if not any([l in exclude for l in case.__labels__]):
                cases.add(case)
    return [fn() for fn in cases]

@register_case(['bipartite', 'zero-degree'])
def bipartite1():
    return dgl.bipartite([(0, 0), (0, 1), (0, 4), (2, 1), (2, 4), (3, 3)])

@register_case(['bipartite'])
def bipartite_full():
    return dgl.bipartite([(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)])

@register_case(['homo'])
def graph0():
    return dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                      [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]))

@register_case(['homo', 'zero-degree', 'homo-zero-degree'])
def bipartite1():
    return dgl.graph([(0, 0), (0, 1), (0, 4), (2, 1), (2, 4), (3, 3)])

@register_case(['homo', 'has_feature'])
def graph1():
    g = dgl.graph(([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 6, 7, 8, 9],
                   [4, 5, 1, 2, 4, 7, 9, 8 ,6, 4, 1, 0, 1, 0, 2, 3, 5]))
    g.ndata['h'] = F.copy_to(F.randn((g.number_of_nodes(), 2)), F.cpu())
    g.edata['w'] = F.copy_to(F.randn((g.number_of_edges(), 3)), F.cpu())
    return g

@register_case(['hetero', 'has_feature'])
def heterograph0():
    g = dgl.heterograph({
        ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])})
    g.nodes['user'].data['h'] = F.copy_to(F.randn((g.number_of_nodes('user'), 3)), F.cpu())
    g.nodes['game'].data['h'] = F.copy_to(F.randn((g.number_of_nodes('game'), 2)), F.cpu())
    g.nodes['developer'].data['h'] = F.copy_to(F.randn((g.number_of_nodes('developer'), 3)), F.cpu())
    g.edges['plays'].data['h'] = F.copy_to(F.randn((g.number_of_edges('plays'), 1)), F.cpu())
    g.edges['develops'].data['h'] = F.copy_to(F.randn((g.number_of_edges('develops'), 5)), F.cpu())
    return g


@register_case(['batched', 'homo'])
def batched_graph0():
    g1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
    g2 = dgl.graph(([1, 1], [2, 0]))
    g3 = dgl.graph(([0], [1]))
    return dgl.batch([g1, g2, g3])

@register_case(['block', 'bipartite', 'block-biparitite'])
def block_graph0():
    g = dgl.graph(([2, 3, 4], [5, 6, 7]), num_nodes=100)
    return dgl.to_block(g)

@register_case(['block'])
def block_graph1():
    g = dgl.heterograph({
            ('user', 'plays', 'game') : ([0, 1, 2], [1, 1, 0]),
            ('user', 'likes', 'game') : ([1, 2, 3], [0, 0, 2]),
            ('store', 'sells', 'game') : ([0, 1, 1], [0, 1, 2]),
        })
    return dgl.to_block(g)

@register_case(['clique'])
def clique():
    g = dgl.graph(([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]))
    return g

def random_dglgraph(size):
    return dgl.DGLGraph(nx.erdos_renyi_graph(size, 0.3))

def random_graph(size):
    return dgl.graph(nx.erdos_renyi_graph(size, 0.3))

def random_bipartite(size_src, size_dst):
    return dgl.bipartite(ssp.random(size_src, size_dst, 0.1))

def random_block(size):
    g = dgl.graph(nx.erdos_renyi_graph(size, 0.1))
    return dgl.to_block(g, np.unique(F.zerocopy_to_numpy(g.edges()[1])))

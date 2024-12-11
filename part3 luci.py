import networkx as nx

#from networkx.algorithms.community import girvan_newman
#from community import community_louvain



def num_common_nodes(*arg):
    if len(arg) < 2:
        raise ValueError("At least two graphs are required to find common nodes.")
    node_sets = [set(graph.nodes) for graph in arg]
    common_nodes = set.intersection(*node_sets)
    return len(common_nodes)

    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """


def get_degree_distribution(g: nx.Graph) -> dict:
    degree_count = {}
    for _, degree in g.degree():
        if degree not in degree_count:
            degree_count[degree] = 0  # Inicializar si el grado no está en el diccionario
        degree_count[degree] += 1   # Incrementar el contador para este grado
    return degree_count

    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    if metric == 'degree':
        centrality = nx.degree_centrality(g)
    elif metric == 'betweenness':
        centrality = nx.betweenness_centrality(g)
    elif metric == 'closeness':
        centrality = nx.closeness_centrality(g)
    elif metric == 'eigenvector':
        centrality = nx.eigenvector_centrality(g)
    else:
        raise ValueError("Unsupported metric. Use 'degree', 'betweenness', 'closeness', or 'eigenvector'.")
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:num_nodes]]

    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes.
    """
 

def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    all_cliques = list(nx.find_cliques(g))
    filtered_cliques = [clique for clique in all_cliques if len(clique) >= min_size_clique]
    nodes_in_cliques = set(node for clique in filtered_cliques for node in clique)
    return filtered_cliques, list(nodes_in_cliques)
    
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """


def detect_communities(g: nx.Graph, method: str) -> tuple:
    if method == 'girvan-newman':
        communities_generator = girvan_newman(g)
        communities = next(communities_generator)
        communities = [list(c) for c in communities]
    elif method == 'louvain':
        partition = community_louvain.best_partition(g)
        communities = {}
        for node, comm in partition.items():
            communities.setdefault(comm, []).append(node)
        communities = list(communities.values())
    else:
        raise ValueError("Unsupported method. Use 'girvan-newman' or 'louvain'.")
    modularity = nx.algorithms.community.quality.modularity(g, communities)
    return communities, modularity

    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
  


if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #


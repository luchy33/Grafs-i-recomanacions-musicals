import networkx as nx

#from networkx.algorithms.community import girvan_newman
#from community import community_louvain

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    if len(arg) < 2: #si no hi ha grafs a la tupla arg o només 1 no es poden trobar nodes comuns
        raise ValueError("Error: Com a mínim es necessiten dos grafs per poder trobar nodes comuns.")
    node_sets = [set(graph.nodes) for graph in arg] #recorrem cada graf de la tupla, seleccionem els seus nodes i els guardem en un conjunt. Com a resultat obtenim una llista de conjunts de nodes
    common_nodes = set.intersection(*node_sets) #fem la intersecció de tots els conjunts de nodes per seleccionar els comuns, * permet obtenir tots els conjunts de la llista
    return len(common_nodes) #retornem la longitud del conjunt de nodes comuns per saber quants n'hi ha de comuns


def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    degree_count = {} #creem un diccionari per poder guardar la quantitat de nodes amb x degree
    for node, degree in g.degree(): #iterem sobre el graf i obtenim un node i el grau d'aquest
        if degree not in degree_count: #comprovem que el grau no estigui al diccionari
            degree_count[degree] = 0  #l'inicialitzem si no està en el diccionari a 0
        degree_count[degree] += 1   #incrementem el comptador (tant si s'acaba d'afegir el grau o ja existeix com a clau)
    return degree_count #retornem el diccionari


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes.
    """
    centrality_functions = { 
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality,
    } #diccionari amb les mètriques disponibles a la funció
    
    if metric not in centrality_functions: #comprovem si la mètrica està en el diccionari, per tant, si és vàlida
        raise ValueError("Error: Mètric invàlida, utilitza 'degree', 'betweenness', 'closeness', o 'eigenvector'.")

    centrality = centrality_functions[metric](g) #calculem la centralitat del graf, seleccionant la mètrica del diccionari i ho guardem en el diccionari centrality (claus: nodes, valors: valor centralitat)

    return sorted(centrality, key=centrality.get, reverse=True)[:num_nodes] #ordenem els nodes segons els valors de centralitat més alts (reverse=True) i els guardem en una llista, després seleccionem els primers num_nodes i els retornem 


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


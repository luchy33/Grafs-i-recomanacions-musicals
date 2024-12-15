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
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    all_cliques = list(nx.find_cliques(g)) #trobem totes les cliques del graf, cada clique emmagatzemada en una llista (de nodes que formen la clique) i totes les llistes es troben en una mateixa llista anomenada "all_cliques" 
    filtered_cliques = [clique for clique in all_cliques if len(clique) >= min_size_clique] #filtrem la llista de llistes/cliques per quedar-nos només amb les que tinguin igual o major nombre de nodes que "min_size_cliques"
    nodes_in_cliques = set(node for clique in filtered_cliques for node in clique) #recorrem cada node de cada clique i l'afegim al conjunt "nodes_in_cliques" per quedar-nos sense cap repetit
    return filtered_cliques, list(nodes_in_cliques) #retornem la llista "filtered_cliques" i el conjunt "nodes_in_cliques" el retornem com una llista


def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
   if method.lower() == 'girvan-newman': #si el mètode seleccionat es Girvan-Newman utilitzem el seu algoritme
        communities_generator = girvan_newman(g) #inicialitzem un generador que produirà particionsd successivament
        communities = next(communities_generator) #amb next obtenim la primera partició de comunitats del generador (al ser la primera conté poques comunitats grans)
        #la partició retornada ("comunities") és una tupla de conjunts, on cada conjunt representa una comunitat de nodes -> ({comunitat 1}, {comunitat 2})
        communities = [list(c) for c in communities] #iterem sobre "comunities" i ho convertim en una llista per guardar a "comunities" una llista de llistes -> [[comunitat 1], [comunitat 2]]
    
    elif method.lower() == 'louvain': #si el mètode seleccionat és Louvain
        partition = community_louvain.best_partition(g) #detectem comunitats, "partition" és un diccionari on key=nodes i value=número per indicar a quina comunitat pertany el node
        communities = {} #creem un diccionari per reorganitzar els nodes
        for node, comm in partition.items(): #iterem sobre el diccionari partition per tenir cada clau i valor
            if comm not in communities:  # omprovem si la comunitat no està ja al diccionari
                communities[comm] = []  #si no existeix, la inicialitzem amb una llista buida
            communities[comm].append(node)  #afegim el node a la llista de la comunitat corresponent
        communities = list(communities.values())  #convertim el diccionari en una llista de llistes
    
    else: #si el mètode que s'ha passat com a paràmetre no és girvan-newman/louvain llancem excepció
        raise ValueError("Error: Mètode invàlida, utilitza 'girvan-newman' o 'louvain'.")
    modularity = nx.algorithms.community.quality.modularity(g, communities) #calculem la modularitat utilitzant netowrkx de la partició obtinguda
    return communities, modularity #retornem les comunitats i la modularitat de la partició
  


if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #


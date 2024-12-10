import networkx as nx
import pandas as pd
import numpy as np


def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    graf = nx.Graph() #creem un graf NO dirigit
    for u, v in g.edges(): #recorrem totes les arestes del graf g (que és dirigit i és passat com a paràmetre)
        if g.has_edge(v, u):  #si les arestes són bidireccionals (de v a u i de u a v)
            graf.add_edge(u, v) #afegim una aresta (no dirigida) entre u i v en el graf NO dirigit creat
    nx.write_graphml(graf, out_filename) #guardem el graf NO dirigit creat amb el nom passat com a paràmetre (out_filename) en format GraphML
    return graf #retorna el graf NO dirigit creat

    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """



def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    graf = g.copy() #fem una còpia del graf passat com a paràmetre (g)
    nodes_eliminar = []  #creem llista buida pels nodes a eliminar
    for node, degree in graf.degree(): #iterem cada node i el seu grau en el graf
        if degree < min_degree:  #si el grau del node és més petit que el mínim
            nodes_eliminar.append(node)  #l'afegim el node a la lista
    graf.remove_nodes_from(nodes_eliminar) #eliminem els nodes que estiguin a la llista creada
    grau_zero = [] #creem llista buida pels nodes que s'hagin quedat amb grau 0
    for node, degree in graf.degree(): #iterem cada node i el seu grau en el graf
        if degree == 0: #si el grau és 0
            grau_zero.append(node) #l'afegim a la llista
    graf.remove_nodes_from(grau_zero) #eliminem els nodes que estiguin a la llista creada
    #nx.write_graphml(graf, out_filename)  ##guardem el graf creat amb el nom passat com a paràmetre (out_filename) en format GraphML
    return graf #retorna el graf "modificat" on s'ha eliminat els nodes de grau més petit al mínim passat

    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    # Validar los parámetros de entrada
    if (min_weight is None and min_percentile is None) or (min_weight is not None and min_percentile is not None):
        raise ValueError("Specify exactly one of 'min_weight' or 'min_percentile', not both or neither.")

    # Determinar el umbral basado en el percentil, si corresponde
    if min_percentile is not None:
        if not (0 <= min_percentile <= 100):
            raise ValueError("'min_percentile' must be between 0 and 100.")
        
        # Obtener los pesos de las aristas
        edge_weights = [data['weight'] for _, _, data in g.edges(data=True)]
        min_weight = np.percentile(edge_weights, min_percentile)

    # Crear un nuevo grafo
    pruned_graph = g.copy()

    # Eliminar las aristas con peso menor al umbral
    edges_to_remove = [(u, v) for u, v, data in pruned_graph.edges(data=True) if data['weight'] < min_weight]
    pruned_graph.remove_edges_from(edges_to_remove)

    # Eliminar nodos aislados (grado 0)
    zero_degree_nodes = [node for node, degree in pruned_graph.degree() if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)

    # Guardar el grafo en formato graphml si se especifica
    if out_filename:
        nx.write_graphml(pruned_graph, out_filename)

    return pruned_graph
    # ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    required_columns = {'artist_id', 'artist_name'}
    if not required_columns.issubset(tracks_df.columns):
        raise ValueError(f"The DataFrame must contain at least the following columns: {required_columns}")
    
    # Identificar las columnas que contienen características de audio
    audio_feature_columns = tracks_df.columns.difference(['artist_id', 'artist_name'])
    
    # Agrupar por artista y calcular el promedio de las características de audio
    artist_features = tracks_df.groupby(['artist_id', 'artist_name'])[audio_feature_columns].mean().reset_index()
    
    return artist_features
    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> \
        nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    # Extract artist names and audio features
    artist_names = artist_audio_features_df.index.tolist()  # Assuming artists are indexed
    features = artist_audio_features_df.values  # Get the feature matrix (numeric data)

    # Compute the similarity or distance matrix
    if similarity.lower() == "cosine":
        similarity_matrix = cosine_similarity(features)
    elif similarity.lower() == "euclidean":
        similarity_matrix = -euclidean_distances(features)  # Invert distances to represent similarity
    else:
        raise ValueError("Unsupported similarity metric. Use 'cosine' or 'euclidean'.")

    # Create a NetworkX graph
    similarity_graph = nx.Graph()

    # Add nodes with artist metadata
    for artist in artist_names:
        similarity_graph.add_node(artist)

    # Add edges with similarity weights
    for i, artist_a in enumerate(artist_names):
        for j, artist_b in enumerate(artist_names):
            if i < j:  # Avoid duplicate edges (undirected graph)
                weight = similarity_matrix[i, j]
                similarity_graph.add_edge(artist_a, artist_b, weight=weight)

    # Save the graph to a file if an output filename is provided
    if out_filename:
        nx.write_graphml(similarity_graph, out_filename)

    return similarity_graph
    # ----------------- END OF FUNCTION --------------------- #




if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #Pas a)
    print("------Processant BFS i DFS------")
    gb = nx.read_graphml("BrunoMars_100_BFS.graphml")
    undirected_graph_bfs = retrieve_bidirectional_edges(gb, "BrunoMars_100_BFS_undirected.graphml")
    gd = nx.read_graphml("BrunoMars_100_DFS.graphml")
    undirected_graph_dfs = retrieve_bidirectional_edges(gd, "BrunoMars_100_DFS_undirected.graphml")
    print("------Grafs no dirigits guardats------")
    
    # Pas (b): Crear un graf de similitud basat en les característiques mitjanes
    print("\n------ Processant característiques d'àudio ------")
    # Llegir el DataFrame amb les característiques d'àudio (pots canviar l'arxiu si cal)
    tracks_df = pd.read_csv("BrunoMars_tracks.csv")  # Arxiu amb les característiques de les cançons
    
    # Calcular les característiques mitjanes per artista
    mean_audio_features_df = compute_mean_audio_features(tracks_df)
    print("Característiques mitjanes calculades per a cada artista.")
    
    # Crear un graf de similitud (g_w) basat en les característiques mitjanes
    similarity_graph = create_similarity_graph(
        mean_audio_features_df.set_index("artist_name").iloc[:, 2:],  # Usar artist_name com a índex i excloure ID
        similarity="cosine",  # Mètrica de similitud (també pots usar "euclidean")
        out_filename="BrunoMars_similarity_graph.graphml"
    )
    print("Graf de similitud creat i desat a 'BrunoMars_similarity_graph.graphml'.")

    
    # ------------------- END OF MAIN ------------------------ #


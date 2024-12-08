import networkx as nx
import pandas as pd

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    undirected_graph = nx.Graph()
    
    # Iterar sobre totes les arestes
    for u, v in g.edges():
        if g.has_edge(v, u):  # Verificar si les arestes són bidireccionals
            undirected_graph.add_edge(u, v)
    
    # Guardar el graf resultant en format graphml
    nx.write_graphml(undirected_graph, out_filename)
    
    return undirected_graph
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pruned_graph = g.copy()

    # Identificar els nodes amb grau menor a min_degree
    nodes_to_remove = [node for node, degree in pruned_graph.degree() if degree < min_degree]

    # Eliminar-los
    pruned_graph.remove_nodes_from(nodes_to_remove)

    # Eliminar nodes de grau 0 que hagin quedat
    zero_degree_nodes = [node for node, degree in pruned_graph.degree() if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)

    #nx.write_graphml(pruned_graph, out_filename) 
    #MIRAR SI CAL GUARDAR

    return pruned_graph
    # ----------------- END OF FUNCTION --------------------- #


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
    pass
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    undirected_graph = retrieve_bidirectional_edges(g, out_filename)
    
    # ------------------- END OF MAIN ------------------------ #

if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #


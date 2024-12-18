import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def generate_report(graph: nx.Graph):
    most_similar_artists = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:2]
    least_similar_artists = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])[:2]

    print("\nArtistes més similars:")
    for artist_a, artist_b, data in most_similar_artists:
        print(f"{graph.nodes[artist_a]['name']} i {graph.nodes[artist_b]['name']} amb pes {data['weight']}")

    print("\nArtistes menys similars:")
    for artist_a, artist_b, data in least_similar_artists:
        print(f"{graph.nodes[artist_a]['name']} i {graph.nodes[artist_b]['name']} amb pes {data['weight']}")

    weighted_degrees = {node: sum(data['weight'] for _, _, data in graph.edges(node, data=True)) for node in graph.nodes()}
    most_similar_to_all = max(weighted_degrees, key=weighted_degrees.get)
    least_similar_to_all = min(weighted_degrees, key=weighted_degrees.get)

    print(f"\nL'artista més similar a la resta: {graph.nodes[most_similar_to_all]['name']} amb grau ponderat {weighted_degrees[most_similar_to_all]}")
    print(f"L'artista menys similar a la resta: {graph.nodes[least_similar_to_all]['name']} amb grau ponderat {weighted_degrees[least_similar_to_all]}")


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    graf = nx.Graph() #creem un graf NO dirigit
    
    for u, v in g.edges(): #recorrem totes les arestes del graf g (que és dirigit i és passat com a paràmetre)
        if g.has_edge(v, u):  #si les arestes són bidireccionals (de v a u i de u a v)
            graf.add_edge(u, v) #afegim una aresta (no dirigida) entre u i v en el graf NO dirigit creat
    for node, data in g.nodes(data=True): #iterem sobre tots els nodes del nou graf
        graf.add_node(node, **data) #afegim tota la informació del node que havia al grafs dirigit
    nx.write_graphml(graf, out_filename) #guardem el graf NO dirigit creat amb el nom passat com a paràmetre (out_filename) en format GraphML
    return graf #retorna el graf NO dirigit creat
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
    if (min_weight is None and min_percentile is None) or (min_weight is not None and min_percentile is not None): #comprova que es passi per paràmetre o bé min_weight o bé min_percentile
        raise ValueError("Has d'especificar o min_weight o min_percentile, no els dos o cap.") #sino, salta error
    if min_percentile is not None: #si el paràmetre és el min_percentile
        if not (0 <= min_percentile <= 100): #ha d'estar entre 0 i 100
            raise ValueError("El 'min_percentile' ha d'estar entre 0 i 100.") #sino, salta error 
        pesos_arestes = [] #creem una llista buida per emmagatzemar els pesos de les arestes
        for u, v, data in g.edges(data=True): #iterem sobre cada aresta del graf i les dades asociades a aquestes
            pesos_arestes.append(data['weight']) #de les dades extreiem el pes i ho afegim a la lista creada 
        min_weight = np.percentile(pesos_arestes, min_percentile) #el percentil dels pesos de les arestes s'emmagatzema a "min_wight" (per a utilitzar posterioirment en el codi)
    graf = g.copy() #fem una còpia de g (per modificar-lo)
    arestes_eliminar = [] #creem llista on emmagatzemarem les arestes que eliminarem
    for u, v, data in graf.edges(data=True): #iterem les arestes i les seves dades
        if data['weight'] < min_weight: #si el pes de l'aresta és més petit que el 'min_weight´
            arestes_eliminar.append((u, v)) #afegim l'aresta a la llista        
    graf.remove_edges_from(arestes_eliminar) #eliminem les arestes del graf que estàn a la llista
    grau_zero = [] #creem llista buida pels nodes que s'hagin quedat amb grau 0
    for node, degree in graf.degree(): #iterem cada node i el seu grau en el graf
        if degree == 0: #si el grau és 0
            grau_zero.append(node) #l'afegim a la llista
    graf.remove_nodes_from(grau_zero) #eliminem els nodes que estiguin a la llista creada
    nx.write_graphml(graf, out_filename)  ##guardem el graf creat amb el nom passat com a paràmetre (out_filename) en format GraphML
    return graf #retorna el graf "modificat" on s'ha eliminat els node
    # ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    required_columns = {'artist_id', 'artist_name'} #indiquem les columnes relacionades amb les característiques d'àudio
    audio_feature_columns = {
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo"
    } #característiques d'àudio utilitzades per calcular les mitjanes
    
    #validem que el DataFrame conté les columnes necessàries per processar la informació
    if not required_columns.issubset(tracks_df.columns):
        raise ValueError(f"El DataFrame ha de contenir com a mínim: {required_columns}")
    
    #seleccionem només les columnes rellevants (identificadors i característiques d'àudio)
    selected_columns = list(required_columns | audio_feature_columns) #combinem les dues categories
    tracks_df = tracks_df[selected_columns] #les extraiem del DataFrame
    
    #agrupem les pistes per 'artist_id' i 'artist_name' i calculem la mitjana de les característiques d'àudio
    artist_features = (
        tracks_df
        .groupby(['artist_id', 'artist_name'])[list(audio_feature_columns)]
        .mean()  #calculem la mitjana per cada columna "audio_features_columns"
        .reset_index()  #eestablim l'índex per obtenir un DataFrame perquè artits_id i artist_name no facin "d'índex"
    )
    artist_features.to_csv("Bruno_Mars_features.csv", index=False)
    return artist_features # Retornem el nou DataFrame amb les mitjanes calculades
    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    artist_ids = artist_audio_features_df["artist_id"].tolist()
    artist_names = artist_audio_features_df["artist_name"].tolist()
    features = artist_audio_features_df.iloc[:, 2:].values  # Agafem només les característiques d'àudio

    if similarity.lower() == "cosine":
        similarity_matrix = cosine_similarity(features)
    elif similarity.lower() == "euclidean":
        similarity_matrix = 1 / (1 + euclidean_distances(features))
    else:
        raise ValueError("Invalid similarity metric. Use 'cosine' or 'euclidean'.")

    similarity_graph = nx.Graph()
    
    # Afegir els nodes amb noms dels artistes
    for artist_id, artist_name in zip(artist_ids, artist_names):
        similarity_graph.add_node(artist_id, name=artist_name)

    # Afegir arestes amb el pes de la similitud
    for i, artist_a in enumerate(artist_ids):
        for j, artist_b in enumerate(artist_ids):
            if i < j:  # Evitem duplicar arestes
                weight = similarity_matrix[i, j]
                similarity_graph.add_edge(artist_a, artist_b, weight=weight)

    if out_filename:
        nx.write_graphml(similarity_graph, out_filename)

    return similarity_graph

    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    # Pas a)
    print("------ Processant BFS i DFS ------")
    try:
        gb = nx.read_graphml("BrunoMars_100_BFS.graphml")
        undirected_graph_bfs = retrieve_bidirectional_edges(gb, "BrunoMars_100_BFS_undirected.graphml")
        gd = nx.read_graphml("BrunoMars_100_DFS.graphml")
        undirected_graph_dfs = retrieve_bidirectional_edges(gd, "BrunoMars_100_DFS_undirected.graphml")
        print("------ Grafs no dirigits guardats ------")
    except FileNotFoundError as e:
        print(f"No s'ha trobat algun fitxer: {e}")
    except Exception as e:
        print(f"Error processant BFS o DFS: {e}")

    # Pas (b): Crear un graf de similitud basat en les característiques mitjanes
    print("\n------ Processant característiques d'àudio ------")
    try:
        tracks_df = pd.read_csv("BrunoMars_100_tracks.csv")
        required_columns = {'artist_id', 'artist_name', 'danceability', 'energy', 'loudness',
                            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'}
        if not required_columns.issubset(tracks_df.columns):
            print(f"El DataFrame no conté totes les columnes necessàries: {required_columns - set(tracks_df.columns)}")
            exit()
        mean_audio_features_df = compute_mean_audio_features(tracks_df)
        print("Característiques mitjanes calculades per a cada artista.")
    except Exception as e:
        print(f"Error processant el CSV o les característiques d'àudio: {e}")
        exit()

    try:
        similaritat = "cosine"
        similarity_graph = create_similarity_graph(
    mean_audio_features_df,
    similarity= similaritat,
    out_filename="BrunoMars_similarity_graph.graphml"
)
        print(f"Graf de similitud {similaritat} creat i desat a 'BrunoMars_similarity_graph.graphml'\n")
        print("------ Generant report del graf ------")
        generate_report(similarity_graph)
    except Exception as e:
        print(f"Error creant el graf de similitud: {e}")

    # ------------------- END OF MAIN ------------------------ #



    # ------------------- END OF MAIN ------------------------ #


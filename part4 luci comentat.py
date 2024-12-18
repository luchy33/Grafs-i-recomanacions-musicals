import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def get_artist_id_by_name(g, artist_name):
    """Obté l'ID de l'artista a partir del seu nom."""
    for node_id, data in g.nodes(data=True):  #iterem sobre els nodes del graf i les seves dades
        if data.get('name') == artist_name:  #comprovem si el nom de l'artista coincideix
            return node_id  #retornem l'ID del node si trobem una coincidència
    return None  #retornem None si no trobem l'artista

def get_degree_distribution(g: nx.Graph) -> dict:
    degree_count = {} #inicialitzem un diccionari per comptar graus.
    for node, degree in g.degree(): #iterem sobre els nodes i els seus graus.
        if degree not in degree_count: #si el grau no està al diccionari
            degree_count[degree] = 0 #l'inicalitzem
        degree_count[degree] += 1 #incrementem el comptador del grau.
    return degree_count #retorna la distribució de graus.

def find_most_similar_artist(similarity_graph: nx.Graph, bruno_mars_id: str) -> str:
    """
    Troba l'artista més similar a Bruno Mars dins del grafo de similitud.
    """
    max_similarity = 0 #inicialitzem la màxima similitud com a zero.
    most_similar_artist = None #inicialitzem l'artista més similar com a None.
   
    for u, v, data in similarity_graph.edges(data=True): #iterem sobre les arestes i les seves dades.
        if bruno_mars_id in (u, v): #comprovem si Bruno Mars és un dels extrems de l'aresta.
            other_artist = v if u == bruno_mars_id else u #determina l'artista oposat
            if data['weight'] > max_similarity: #comprova si el pes és més gran que la màxima similaritat
                max_similarity = data['weight']
                most_similar_artist = other_artist #actualitza l'artista més similar
   
    return most_similar_artist #retorna l'artista més similar

def find_least_similar_artist(similarity_graph: nx.Graph, bruno_mars_id: str) -> str:
    """Troba l'artista menys similar a Bruno Mars."""
    min_similarity = float('inf')  #inicialitza la mínima similitud amb infinit
    least_similar_artist = None  #inicialitza l'artista menys similar com a None
    
    for u, v, data in similarity_graph.edges(data=True):  #itera sobre les arestes i les seves dades
        if bruno_mars_id in (u, v):  #comprova si Bruno Mars és un dels extrems de l'aresta
            other_artist = v if u == bruno_mars_id else u  # determina l'artista oposat
            if data['weight'] < min_similarity:  #comprova si el pes és menor que la minima similaritat
                min_similarity = data['weight']
                least_similar_artist = other_artist  #actualitza l'artista menys similar
    
    return least_similar_artist  #retorna l'artista menys similar.


def get_artist_names(g: nx.Graph, node_ids: list) -> list:
    """Converteix una llista d'ID de nodes als noms corresponents."""
    artist_names = []  #inicialitzem la llista de noms
    for node_id in node_ids:  #recorre els IDs dels nodes.
        name = g.nodes[node_id].get('name', "Desconegut")  #obté el nom o retorna "Desconegut"
        artist_names.append(name)  #afegeix el nom a la llista.
    return artist_names  #retorna la llista de noms.


def plot_degree_distribution(degree_dict: dict, graph_name: str = "Graph", normalized: bool = False, loglog: bool = False) -> None:
    """Genera un gràfic de la distribució de graus."""
    graus = list(degree_dict.keys())  #extreu els graus
    occurences = list(degree_dict.values())
    if normalized:  #normalitza
        total = sum(occurences)
        occurences = [o/total for o in occurences]

    plt.figure(figsize=(8, 6)) #crea la figura
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7) #afegeix la graella
    if loglog:  #configura el graf com a log-log
        plt.loglog(graus, occurences, marker="o", linestyle="", label="Distribució de graus")
        plt.xlabel("Grau (escala log)")
        plt.ylabel("Freqüència (escala log)" if not normalized else "Probabilitat (escala log)")
    else:
        plt.plot(graus, occurences, marker="o", linestyle="", label="Distribució de graus")
        plt.xlabel("Grau")
        plt.ylabel("Freqüència" if not normalized else "Probabilitat")
    plt.title(f"Distribució de graus de {graph_name}")
    plt.legend()
    plt.show()


def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """Plot a (single) figure with a plot of mean audio features of two different artists."""
    if artist1_id not in artists_audio_feat.index or artist2_id not in artists_audio_feat.index:
        raise ValueError("Algun artist IDs no s'ha trobat al DataFrame index.")
   
    artist1_features = artists_audio_feat.loc[artist1_id].drop('artist_name', errors='ignore')
    artist2_features = artists_audio_feat.loc[artist2_id].drop('artist_name', errors='ignore')
    audio_features = artist1_features.index.tolist()

    x = np.arange(len(audio_features))
    width = 0.35
   
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, artist1_features, width, label=f"Artist 1: {artist1_id}")
    plt.bar(x + width / 2, artist2_features, width, label=f"Artist 2: {artist2_id}")
    plt.xticks(x, audio_features, rotation=45, ha="right")
    plt.xlabel("Audio Features")
    plt.ylabel("Mean Value")
    plt.title("Comparison of Mean Audio Features Between Two Artists")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """Plot a heatmap of the similarity between artists."""
    if similarity.lower() not in {"cosine", "euclidean"}:
        raise ValueError("ERROR: utilitza 'cosine' o 'euclidean'.")
    features = artist_audio_features_df.values
    if similarity.lower() == "cosine": #si la similaritat passada és cosine
        similarity_matrix = cosine_similarity(features)
    elif similarity.lower() == "euclidean": #si la similaritat passada és euclidian
        similarity_matrix = -euclidean_distances(features)
    artist_names = artist_audio_features_df.index.tolist()
    similarity_df = pd.DataFrame(similarity_matrix, index=artist_names, columns=artist_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True,
                cbar_kws={"label": f"{similarity.capitalize()} Similarity"})
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(f"Heatmap of {similarity.capitalize()} Similarity Between Artists")
    plt.xlabel("Artists")
    plt.ylabel("Artists")
    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename)
        print(f"Heatmap saved to {out_filename}")
    plt.show()

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


if __name__ == "__main__":
    gw = nx.read_graphml("BrunoMars_similarity_graph.graphml")
    artists_audio_feat = pd.read_csv("Bruno_Mars_features.csv", index_col="artist_name")
    artists_audio_feat = artists_audio_feat.drop(columns=['artist_id'], errors='ignore')

    # Trobar l'artista més similar a Bruno Mars
    bruno_mars_id = get_artist_id_by_name(gw, "Bruno Mars")
    if bruno_mars_id is None:
        raise ValueError("No s'ha trobat Bruno Mars al graf gw.")

    most_similar_artist_id = find_most_similar_artist(gw, bruno_mars_id)
    most_similar_artist_name = get_artist_names(gw, [most_similar_artist_id])[0]
    print(f"L'artista més similar a Bruno Mars és: {most_similar_artist_name}")

    # Trobar l'artista menys similar a Bruno Mars
    least_similar_artist_id = find_least_similar_artist(gw, bruno_mars_id)
    least_similar_artist_name = get_artist_names(gw, [least_similar_artist_id])[0]
    print(f"L'artista menys similar a Bruno Mars és: {least_similar_artist_name}")

    # Comparar l'artista més similar amb Bruno Mars
    if most_similar_artist_name not in artists_audio_feat.index:
        raise ValueError(f"L'artista més similar ({most_similar_artist_name}) no està al DataFrame.")

    if least_similar_artist_name not in artists_audio_feat.index:
        raise ValueError(f"L'artista menys similar ({least_similar_artist_name}) no està al DataFrame.")

    plot_audio_features(artists_audio_feat, artist1_id="Bruno Mars", artist2_id=most_similar_artist_name)

    # Comparar l'artista menys similar amb Bruno Mars
    plot_audio_features(artists_audio_feat, artist1_id="Bruno Mars", artist2_id=least_similar_artist_name)

    # Generar el heatmap de similitud
    plot_similarity_heatmap(artists_audio_feat, similarity="cosine")
    # Comparar Bruno Mars amb Lady Gaga
    plot_audio_features(artists_audio_feat, artist1_id="Bruno Mars", artist2_id="Lady Gaga")
   
    # Comparar Bruno Mars amb T-Pain
    plot_audio_features(artists_audio_feat, artist1_id="Bruno Mars", artist2_id="T-Pain")

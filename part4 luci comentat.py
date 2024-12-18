import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    graus = list(degree_dict.keys()) #emmagatzema les claus del diccionari (que representen els graus) en una llista
    occurences = list(degree_dict.values()) #emmagatzema els valors del diccionari (que representa la freqüència de cada grau) en una llista
    if normalized: #si es TRUE (passat com a paràmetre)
        total = sum(occurences) #es calcula el total
        occurences = [o/total for o in occurences] #normalitza les ocurrències dividint per la suma total
    
    plt.figure(figsize=(8, 6)) #crea una nova figura de gràfic
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7) #afegeix una quadrícula amb línies discontínues
    if loglog: #si loglog és TRUE
        plt.loglog(graus, occurences, marker="o", linestyle="", label="Degree distribution") #grafica en escala log-log
        plt.xlabel("Degree (log scale)") #etiqueta per a l'eix X
        plt.ylabel("Frequency (log scale)" if not normalized else "Probability (log scale)") #etiqueta per a l'eix Y
    else:
        plt.plot(graus, occurences, marker="o", linestyle="", label="Degree distribution") #grafica en escala lineal
        plt.xlabel("Degree") #etiqueta per a l'eix X
        plt.ylabel("Frequency" if not normalized else "Probability") #etiqueta per a l'eix Y
    plt.title("Degree Distribution") #títol del gràfic
    plt.legend() #afegir llegenda
    plt.show() #mostrar el gràfic
    # ----------------- END OF FUNCTION --------------------- #

def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    
    if artist1_id not in artists_audio_feat.index or artist2_id not in artists_audio_feat.index: 
        raise ValueError("Algun artist IDs no s'ha trobat al DataFrame index.") #si no es troben els IDs dels artistes, llança error
    artist1_features = artists_audio_feat.loc[artist1_id] #obté les característiques de l'artista 1
    artist2_features = artists_audio_feat.loc[artist2_id] #obté les característiques de l'artista 2
    audio_features = artists_audio_feat.columns #les característiques d'àudio
    x = np.arange(len(audio_features))  #genera un array amb l'índex de les característiques d'àudio
    width = 0.35  #ample de les barres per la gràfica
    plt.figure(figsize=(10, 6)) #crea una nova figura de gràfic
    plt.bar(x - width/2, artist1_features, width, label=f"Artist 1: {artist1_id}", color="blue") #gràfic de barres per l'artista 1
    plt.bar(x + width/2, artist2_features, width, label=f"Artist 2: {artist2_id}", color="orange") #gràfic de barres per l'artista 2
    plt.xticks(x, audio_features, rotation=45, ha="right") #afegir etiquetes a l'eix X amb rotació
    plt.xlabel("Audio Features") #etiqueta per a l'eix X
    plt.ylabel("Mean Value") #etiqueta per a l'eix Y
    plt.title("Comparison of Mean Audio Features Between Two Artists") #títol del gràfic
    plt.legend() #afegir llegenda
    plt.tight_layout()  # Ajustar márgenes per evitar superposició
    plt.grid(axis="y", linestyle="--", alpha=0.7) #afegir quadrícula a l'eix Y
    plt.show() #mostrar el gràfic

def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    if similarity.lower() not in {"cosine", "euclidean"}: #si "similarity" (passat com a paràmetre) no és "cosine" o "euclidean"
        raise ValueError("ERROR: utilitza 'cosine' o 'euclidean'.")  #salta error si no es troba un valor correcte
    features = artist_audio_features_df.values  #agafa els valors numèrics del dataframe i ho emmagatzema a "features"
    if similarity.lower() == "cosine": #si és cosine
        similarity_matrix = cosine_similarity(features) #calcula la similitud cosinus
    elif similarity.lower() == "euclidean":
        similarity_matrix = -euclidean_distances(features)  #calcula les distàncies euclidianes i les nega
    artist_names = artist_audio_features_df.index.tolist() #obtén els noms dels artistes
    similarity_df = pd.DataFrame(similarity_matrix, index=artist_names, columns=artist_names) #crea un dataframe amb la matriu de similitud
    plt.figure(figsize=(10, 8)) #crea una nova figura de gràfic
    sns.heatmap(similarity_df, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True, 
                cbar_kws={"label": f"{similarity.capitalize()} Similarity"}) #crea el heatmap
    plt.title(f"Heatmap of {similarity.capitalize()} Similarity Between Artists") #títol del gràfic
    plt.xlabel("Artists") #etiqueta per a l'eix X
    plt.ylabel("Artists") #etiqueta per a l'eix Y
    plt.tight_layout() #ajusta els marges per evitar sobreposició
    if out_filename:
        plt.savefig(out_filename) #guarda la imatge si s'ha passat un nom de fitxer
        print(f"Heatmap saved to {out_filename}") #imprimeix que s'ha desat
    plt.show() #mostrar el gràfic


if __name__ == "__main__":
    degree_dict = {1: 10, 2: 20, 3: 15, 4: 5}
    print("Plotting Degree Distribution...")
    plot_degree_distribution(degree_dict, normalized=False, loglog=False)
    plot_degree_distribution(degree_dict, normalized=True, loglog=True)
    audio_features_data = {
        "danceability": [0.8, 0.6, 0.7],
        "energy": [0.9, 0.5, 0.6],
        "valence": [0.7, 0.4, 0.5],
        "tempo": [120, 140, 135]
    }
    artists_audio_feat = pd.DataFrame(audio_features_data, index=["Artist_A", "Artist_B", "Artist_C"])

    print("Plotting Audio Features Comparison...")
    plot_audio_features(artists_audio_feat, artist1_id="Artist_A", artist2_id="Artist_B")
    print("Plotting Similarity Heatmap...")
    plot_similarity_heatmap(artists_audio_feat, similarity="cosine", out_filename="cosine_similarity_heatmap.png")
    plot_similarity_heatmap(artists_audio_feat, similarity="euclidean")



if __name__ == "__main__":
    
    print("# --- Exercici 4 --- #")
    gw = nx.read_graphml("BrunoMars_similarity_graph.graphml")
    gb_un = nx.read_graphml("BrunoMars_100_BFS_undirected.graphml")
    gd_un = nx.read_graphml("BrunoMars_100_DFS_undirected.graphml")
    """
    #Part a)
    degree_dict_gw = get_degree_distribution(gw)
    degree_dict_gb_un = get_degree_distribution(gb_un)
    degree_dict_gd_un = get_degree_distribution(gd_un)
    print("Plotting Degree Distribution for gw...")
    plot_degree_distribution(degree_dict_gw, "Gw", normalized=True, loglog=True)
    print("Plotting Degree Distribution for gb_un...")
    plot_degree_distribution(degree_dict_gb_un, "G'b",normalized=True, loglog=True)
    print("Plotting Degree Distribution for gd_un...")
    plot_degree_distribution(degree_dict_gd_un, "G'd", normalized=True, loglog=True)
    """
    #Part b)
    bruno_mars_id = get_artist_id_by_name(gw, "Bruno Mars")
    
    # Trobar l'artista més similar a Bruno Mars
    most_similar_artist = find_most_similar_artist(gw, bruno_mars_id)
    print(f"L'artista més similar a Bruno Mars és: {most_similar_artist}")
    
    # Carregar el DataFrame amb les característiques d'àudio dels artistes
    artists_audio_feat = pd.read_csv("Bruno_Mars_features.csv", index_col="artist_id")
    
    # Generar la comparació de les característiques d'àudio entre Bruno Mars i l'artista més similar
    plot_audio_features(artists_audio_feat, artist1_id=bruno_mars_id, artist2_id=most_similar_artist, gw=gw)
    """
    node_attributes = list(gw.nodes(data=True))[:5]
    # Mostrar els atributs del node d'un artista concret, per exemple "BrunoMars"
    print(node_attributes)
    """


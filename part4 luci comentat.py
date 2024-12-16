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
    graus = list(degree_dict.keys()) #emmagatzema les claus del diccionari (que representen els graus) en una llista
    occurences = list(degree_dict.values()) #emmagatzema els valors del diccionari (que representa la freqüència de cada grau) en una llista
    if normalized: #si es TRUE (passat com a paràmetre)
        total = sum(occurences) #es calcula el total
        occurences = [o/total for o in occurences] #FALTA POSARU SENSE LIST COMPR LO QUE NO SE FERHO
    plt.figure(figsize=(8, 6))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    if loglog:
        plt.loglog(graus, occurences, marker="o", linestyle="", label="Degree distribution")
        plt.xlabel("Degree (log scale)")
        plt.ylabel("Frequency (log scale)" if not normalized else "Probability (log scale)")
    else:
        plt.plot(graus, occurences, marker="o", linestyle="", label="Degree distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency" if not normalized else "Probability")
    plt.title("Degree Distribution")
    plt.legend()
    plt.show()



def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    
    if artist1_id not in artists_audio_feat.index or artist2_id not in artists_audio_feat.index:
        raise ValueError("Algun artist IDs no s'ha trobat al DataFrame index.")
    artist1_features = artists_audio_feat.loc[artist1_id]
    artist2_features = artists_audio_feat.loc[artist2_id]
    audio_features = artists_audio_feat.columns
    x = np.arange(len(audio_features))  
    width = 0.35  
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, artist1_features, width, label=f"Artist 1: {artist1_id}", color="blue")
    plt.bar(x + width/2, artist2_features, width, label=f"Artist 2: {artist2_id}", color="orange")
    plt.xticks(x, audio_features, rotation=45, ha="right")
    plt.xlabel("Audio Features")
    plt.ylabel("Mean Value")
    plt.title("Comparison of Mean Audio Features Between Two Artists")
    plt.legend()
    plt.tight_layout()  # Ajustar márgenes para evitar superposición
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()



def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """

    if similarity.lower() not in {"cosine", "euclidean"}: #si "similarity" (passat com a paràmetre) no és "cosine" o "euclidean"
        raise ValueError("ERROR: utilitza 'cosine' o 'euclidean'.")  #salta error
    features = artist_audio_features_df.values  #agafa els valors numèrics del dataframe i ho emmagatzema a "features"
    if similarity.lower() == "cosine": #si és cosine
        similarity_matrix = cosine_similarity(features) #
    elif similarity.lower() == "euclidean":
        similarity_matrix = -euclidean_distances(features)  
    artist_names = artist_audio_features_df.index.tolist()
    similarity_df = pd.DataFrame(similarity_matrix, index=artist_names, columns=artist_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True, 
                cbar_kws={"label": f"{similarity.capitalize()} Similarity"})
    plt.title(f"Heatmap of {similarity.capitalize()} Similarity Between Artists")
    plt.xlabel("Artists")
    plt.ylabel("Artists")
    plt.tight_layout()
    if out_filename:
        plt.savefig(out_filename)
        print(f"Heatmap saved to {out_filename}")
    plt.show()



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


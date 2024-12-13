import networkx as nx
import pandas as pd
import spotipy

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

from spotipy.oauth2 import SpotifyClientCredentials



# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    try:
        result = sp.search(q=f'artist:{artist_name}', type='artist', limit=1) #busquem l'artista a Spotify amb el nom especificat
        if result['artists']['items']: #comprovem si hi ha resultats
            artist_id = result['artists']['items'][0]['id'] #recuperem l'ID del primer artista trobat
            return artist_id #retornem l'ID de l'artista
        else:
            return f"Artista '{artist_name}' no trobat" #si no hi ha resultats, informem que no s'ha trobat l'artista
    except spotipy.exceptions.SpotifyException as e:
        return f"SpotifyException: {e}" #gestiona errors específics de Spotify
    except Exception as e:
        return f"Error: {e}" #gestiona errors generals
    # ----------------- END OF FUNCTION --------------------- #


def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    graph = nx.DiGraph() #creem un graf dirigit buit
    visited = set()  #conjunt d'artistes ja visitats
    to_visit = [seed]  #llista d'artistes pendents de visitar
    
    # Explora els artistes fins arribar al límit màxim o fins que no quedi cap node pendent
    while to_visit and len(visited) < max_nodes_to_crawl:
        current_artist = to_visit.pop(0 if strategy == "BFS" else -1) #agafem el següent artista a visitar segons l'estratègia
       
        if current_artist in visited: 
            continue #si ja hem visitat l'artista, saltem aquesta iteració
       
        try:
            visited.add(current_artist) #marca l'artista com a visitat afegint-lo al conjunt
       
            #recupera tota la informació de l'artista: nom, seguidors, popularitat, gèneres
            artist_data = sp.artist(current_artist)
            graph.add_node(
                current_artist,
                name=artist_data["name"],
                followers=artist_data["followers"]["total"],
                popularity=artist_data["popularity"],
                genres=artist_data["genres"]
            )
       
            related_artists = sp.artist_related_artists(current_artist)["artists"] #recupera els artistes relacionats
            for related_artist in related_artists: #itera sobre cada artista relacionat a l'artista actual
                related_id = related_artist["id"] #recupera l'ID dels artistes relacionats
                if related_id not in visited and related_id not in to_visit: #comprova que no hagin estat visitats
                    to_visit.append(related_id) #afegeix els artistes relacionats a la llista a visitar
                graph.add_edge(current_artist, related_id) #afegeix una aresta del node actual als artistes relacionats
       
        except spotipy.exceptions.SpotifyException as e:
            print(f"S'ha produït una SpotifyException per a l'artista {current_artist}: {e}") #gestiona errors específics de Spotify
        except Exception as e:
            print(f"S'ha produït un error per a l'artista {current_artist}: {e}") #gestiona errors generals
       
    nx.write_graphml(graph, out_filename) #guardem el graf a l'arxiu que hi ha a la variable "out_filename"
    return graph #retornem el graf
    # ----------------- END OF FUNCTION --------------------- #

       
 
def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get top songs in Spain for the artists that appear in each and all graphs in the list.

    :param sp: spotipy client object
    :param graphs: a list of networkx graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with the top song data.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    common_artists = set.intersection(*(set(graph.nodes) for graph in graphs)) #troba els artistes comuns entre tots els grafs
    track_data = [] #llista buida per guardar informació de les cançons

    for artist_id in common_artists:  #iterem sobre cada artista comú entre tots els grafs
        try:
            #cerquem el nom de l'artista dins dels nodes dels grafs
            artist_name = next( # Utilitzem `next` per obtenir el primer resultat vàlid d'entre els grafs que contenen aquest artista
                graph.nodes[artist_id]["name"]  #recuperem el nom de l'artista des dels atributs del node
                for graph in graphs  #iterem sobre la llista de grafs
                if artist_id in graph.nodes  #només considerem els grafs que contenen aquest artista
            )
            
            top_tracks = sp.artist_top_tracks(artist_id, country="ES")["tracks"] #recupera les cançons més escoltades de l'artista
            
            for track in top_tracks:
                audio_features = sp.audio_features([track["id"]])[0] #recupera les característiques d'àudio de cada cançó
                
                track_data.append({ #afegeix la informació de cada cançó a la llista
                    "track_id": track["id"],
                    "track_name": track["name"],
                    "track_duration_ms": track["duration_ms"],
                    "track_popularity": track["popularity"],
                    "album_id": track["album"]["id"],
                    "album_name": track["album"]["name"],
                    "album_release_date": track["album"]["release_date"],
                    "artist_id": artist_id,
                    "artist_name": artist_name,
                    "danceability": audio_features["danceability"] if audio_features else None,
                    "energy": audio_features["energy"] if audio_features else None,
                    "loudness": audio_features["loudness"] if audio_features else None,
                    "speechiness": audio_features["speechiness"] if audio_features else None,
                    "acousticness": audio_features["acousticness"] if audio_features else None,
                    "instrumentalness": audio_features["instrumentalness"] if audio_features else None,
                    "liveness": audio_features["liveness"] if audio_features else None,
                    "valence": audio_features["valence"] if audio_features else None,
                    "tempo": audio_features["tempo"] if audio_features else None,
                })
        
        except spotipy.exceptions.SpotifyException as e:
            print(f"SpotifyException occurred for artist {artist_id}: {e}") #gestiona errors específics de Spotify
        except Exception as e:
            print(f"An error occurred for artist {artist_id}: {e}") #gestiona errors generals

    df = pd.DataFrame(track_data) #crea un DataFrame amb la informació recollida

    df.to_csv(out_filename, index=False) #guarda el DataFrame en un fitxer CSV
    
    return df #retorna el DataFrame
    # ----------------- END OF FUNCTION --------------------- #



if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    CLIENT_ID = "0a03746de0e54710b6aab33d9e73c5c5"
    CLIENT_SECRET = "19a652f5d6a14596a078cba2b5edef3a"
    auth_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    playlists = sp.user_playlists('claudiabf27')
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            print(f"{i + 1 + playlists['offset']:4d} {playlist['uri']} {playlist['name']}")
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None
    print(playlists)
    artist_id = search_artist(sp, "Bruno Mars") #obtenim la id de l'artista
    ll_grafs = list() #fem una llista per emmagatzemar els grafs BFS i DFS
    ll_grafs.append(crawler(sp, str(artist_id), 100, "BFS", "BM_BFS.graphml")) #cridem la funció amb Burno Mars per obtenir el graf BFS
    ll_grafs.append(crawler(sp, str(artist_id), 100, "DFS", "BM_DFS.graphml")) #cridem la funció amb Bruno Mars per obtenir el graf DFS
    data_frame = get_track_data(sp, ll_grafs, "BrunoMars_100.csv") #obtenim el data frame
    # ------------------- END OF MAIN ------------------------ #

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
        result = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
        if result['artists']['items']:
            artist_id = result['artists']['items'][0]['id']
            return artist_id
        else:
            return f"Artist '{artist_name}' not found."
    except spotipy.exceptions.SpotifyException as e:
        return f"SpotifyException occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
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
    graph = nx.DiGraph()
    visited = set()  
    to_visit = [seed] 
    
    # Crawl artists up to the maximum limit or until no nodes remain
    while to_visit and len(visited) < max_nodes_to_crawl:
        current_artist = to_visit.pop(0 if strategy == "BFS" else -1)
        
        if current_artist in visited:
            continue
    
        try:
            # Mark artist as visited
            visited.add(current_artist)
    
            # Retrieve artist information
            artist_data = sp.artist(current_artist)
            graph.add_node(
                current_artist,
                name=artist_data["name"],
                followers=artist_data["followers"]["total"],
                popularity=artist_data["popularity"],
                genres=artist_data["genres"]
            )
    
            # Retrieve related artists
            related_artists = sp.artist_related_artists(current_artist)["artists"]
            for related_artist in related_artists:
                related_id = related_artist["id"]
                if related_id not in visited and related_id not in to_visit:
                    to_visit.append(related_id)
                graph.add_edge(current_artist, related_id)
    
        except spotipy.exceptions.SpotifyException as e:
            print(f"SpotifyException occurred for artist {current_artist}: {e}")
        except Exception as e:
            print(f"An error occurred for artist {current_artist}: {e}")
    
    # Save the graph in graphml format
    nx.write_graphml(graph, out_filename)
    return graph
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
    # Troba els artistes comuns a tots els grafs
    common_artists = set.intersection(*(set(graph.nodes) for graph in graphs))

    track_data = []

    for artist_id in common_artists:
        try:
            # Obté la informació de l'artista d'un dels grafs
            artist_name = next(
                graph.nodes[artist_id]["name"]
                for graph in graphs
                if artist_id in graph.nodes
            )
            
            # Obté les cançons principals de l'artista a Espanya
            top_tracks = sp.artist_top_tracks(artist_id, country="ES")["tracks"]
            
            for track in top_tracks:
                # Obté les característiques d'àudio de la cançó
                audio_features = sp.audio_features([track["id"]])[0]
                
                # Afegeix les dades de la cançó
                track_data.append({
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
            print(f"S'ha produït una SpotifyException per a l'artista {artist_id}: {e}")
        except Exception as e:
            print(f"S'ha produït un error per a l'artista {artist_id}: {e}")

    # Crea un DataFrame a partir de les dades recopilades
    df = pd.DataFrame(track_data)

    # Desa el DataFrame en un fitxer CSV
    df.to_csv(out_filename, index=False)

    return df

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
    ll_grafs = list()
    ll_grafs.append(crawler(sp, "Bruno Mars", 100, "BFS", "BM_BFS.graphml"))
    ll_grafs.append(crawler(sp, "Bruno Mars", 100, "DFS", "BM_DFS.graphml"))
    data_frame = get_track_data(sp, ll_grafs, "BrunoMars_100.csv")
    # ------------------- END OF MAIN ------------------------ #

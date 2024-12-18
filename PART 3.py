import networkx as nx
from networkx.algorithms.community import girvan_newman
#from community import community_louvain
import community as community_louvain



# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def get_artist_id_by_name(g, artist_name):
    """Obté l'ID de l'artista a partir del seu nom."""
    for node_id, data in g.nodes(data=True):  # Iterem sobre els nodes del graf i les seves dades
        if data.get('name') == artist_name:  # Comprovem si el nom de l'artista coincideix
            return node_id  # Retornem l'ID del node si trobem una coincidència
    return None  # Retornem None si no trobem l'artista

def determine_min_clique_size(gb_un: nx.Graph, gd_un: nx.Graph, min_size_range: range = range(3, 11)) -> int:
    """
    Determina la mida mínima de la clique que resulta en almenys 2 cliques en ambdós grafs.

    :param gb_un: El graf g'B (no dirigit).
    :param gd_un: El graf g'D (no dirigit).
    :param min_size_range: El rang de mides de clique a provar (per defecte de 3 a 10).
    :return: La mida mínima de clique que genera almenys 2 cliques en ambdós grafs.
    """
    for min_clique_size in min_size_range:  # Iterem sobre el rang de mides possibles de clique
       # Troba les cliques per a la mida mínima actual
       cliques_b, _ = find_cliques(gb_un, min_size_clique=min_clique_size)  # Trobar les cliques a g'B
       cliques_d, _ = find_cliques(gd_un, min_size_clique=min_clique_size)  # Trobar les cliques a g'D

       # Comprova si hi ha almenys 2 cliques en ambdós grafs
       if len(cliques_b) >= 2 and len(cliques_d) >= 2:
           valid_clique_size = min_clique_size  # Si trobem 2 cliques, actualitzem el valor de la mida de la clique

    if valid_clique_size is not None:  # Si trobem una mida vàlida per a la clique
       print(f"El màxim valor de min_clique_size que genera almenys 2 cliques en ambdós grafs és: {valid_clique_size}")
       return valid_clique_size  # Retornem la mida mínima vàlida
    else:  # Si no trobem cap mida vàlida
       print("No s'ha trobat una mida de clique que generi almenys 2 cliques en ambdós grafs.")
       return None  # Retornem None si no s'ha trobat cap mida vàlida

def get_artist_names(g: nx.Graph, node_ids: list) -> list:
    """
    Converteix una llista d'ID de nodes als seus noms d'artistes corresponents.
    
    :param g: Graf de networkx.
    :param node_ids: Llista d'ID de nodes.
    :return: Llista de noms d'artistes.
    """
    artist_names = []  # Inicialitzem una llista buida per emmagatzemar els noms dels artistes
    for node_id in node_ids:  # Iterem sobre els IDs de nodes
        # Utilitzem get() per evitar el KeyError si 'name' no està present
        name = g.nodes[node_id].get('name', "Desconegut")  # Obtenim el nom de l'artista, si no existeix retornem "Desconegut"
        artist_names.append(name)  # Afegim el nom de l'artista a la llista
    return artist_names  # Retornem la llista de noms d'artistes

def min_ad_cost(g: nx.Graph, cost_per_artist: int = 100):
    """
    Calcula el cost mínim per assegurar que un anunci arribi a qualsevol usuari.
    Ens basem en la cobertura de nodes clau en la xarxa.

    :param g: Graf de recomanacions.
    :param cost_per_artist: Cost per posar un anunci a cada artista.
    :return: Cost mínim en euros.
    """
    # Identificar els nodes més centrals per cobrir tot el graf
    central_nodes = get_k_most_central(g, 'degree', num_nodes=len(g.nodes) // 10)  # Obtenim els nodes més centrals segons el grau
    num_artists = len(central_nodes)  # Comptem el nombre de nodes centrals

    # Calcular el cost
    total_cost = num_artists * cost_per_artist  # El cost total és el nombre d'artistes per el cost per artista
    return total_cost, central_nodes  # Retornem el cost total i els nodes centrals seleccionats

def best_ad_spread(g: nx.Graph, budget: int = 400, cost_per_artist: int = 100):
    """
    Selecciona els artistes amb millor centralitat per assegurar la millor propagació de l'anunci
    amb el pressupost de 400 euros.

    :param g: Graf de recomanacions.
    :param budget: Pressupost disponible per a la campanya publicitària.
    :param cost_per_artist: Cost per posar un anunci a cada artista.
    :return: Llista d'artistes seleccionats per a la publicitat.
    """
    num_artists = budget // cost_per_artist  # Determinem quants artistes podem pagar amb el pressupost disponible
    top_artists = get_k_most_central(g, 'degree', num_nodes=num_artists)  # Seleccionem els artistes més centrals

    return top_artists  # Retornem els artistes seleccionats

def find_shortest_path(g, start_artist, end_artist):
    """Troba el camí més curt entre dos artistes per ID."""
    start_node = get_artist_id_by_name(g, start_artist)  #obtenim l'ID de l'artista d'inici
    end_node = get_artist_id_by_name(g, end_artist)  #obtenim l'ID de l'artista final
    
    if start_node is None:  #si no trobem l'artista d'inici al graf
        return f"Error: '{start_artist}' no es troba al graf."
    if end_node is None:  #si no trobem l'artista final al graf
        return f"Error: '{end_artist}' no es troba al graf."
    
    try:
        path = nx.shortest_path(g, source=start_node, target=end_node)  #trobar el camí més curt
        path_names = get_artist_names(g, path)  #convertim els IDs dels nodes en noms d'artistes
        return path_names  #retornem el camí com a llista de noms d'artistes
    except nx.NetworkXNoPath:  #si no hi ha camí entre els dos artistes
        return None  #retornem None si no es pot trobar un camí

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if len(arg) < 2: #si no hi ha grafs a la tupla arg o només 1 no es poden trobar nodes comuns
        raise ValueError("Error: Com a mínim es necessiten dos grafs per poder trobar nodes comuns.")
    node_sets = [set(graph.nodes) for graph in arg] #recorrem cada graf de la tupla, seleccionem els seus nodes i els guardem en un conjunt. Com a resultat obtenim una llista de conjunts de nodes
    common_nodes = set.intersection(*node_sets) #fem la intersecció de tots els conjunts de nodes per seleccionar els comuns, * permet obtenir tots els conjunts de la llista
    return len(common_nodes) #retornem la longitud del conjunt de nodes comuns per saber quants n'hi ha de comuns
    # ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    degree_count = {} #creem un diccionari per poder guardar la quantitat de nodes amb x degree
    for node, degree in g.degree(): #iterem sobre el graf i obtenim un node i el grau d'aquest
        if degree not in degree_count: #comprovem que el grau no estigui al diccionari
            degree_count[degree] = 0  #l'inicialitzem si no està en el diccionari a 0
        degree_count[degree] += 1   #incrementem el comptador (tant si s'acaba d'afegir el grau o ja existeix com a clau)
    return degree_count #retornem el diccionari
    # ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    centrality_functions = { 
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality,
    } #diccionari amb les mètriques disponibles a la funció
    
    if metric not in centrality_functions: #comprovem si la mètrica està en el diccionari, per tant, si és vàlida
        raise ValueError("Error: Mètric invàlida, utilitza 'degree', 'betweenness', 'closeness', o 'eigenvector'.")

    centrality = centrality_functions[metric](g) #calculem la centralitat del graf, seleccionant la mètrica del diccionari i ho guardem en el diccionari centrality (claus: nodes, valors: valor centralitat)

    return sorted(centrality, key=centrality.get, reverse=True)[:num_nodes] #ordenem els nodes segons els valors de centralitat més alts (reverse=True) i els guardem en una llista, després seleccionem els primers num_nodes
    # ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    all_cliques = list(nx.find_cliques(g)) #trobem totes les cliques del graf, cada clique emmagatzemada en una llista (de nodes que formen la clique) i totes les llistes es troben en una mateixa llista anomenada "all_cliques" 
    filtered_cliques = [clique for clique in all_cliques if len(clique) >= min_size_clique] #filtrem la llista de llistes/cliques per quedar-nos només amb les que tinguin igual o major nombre de nodes que "min_size_cliques"
    nodes_in_cliques = set(node for clique in filtered_cliques for node in clique) #recorrem cada node de cada clique i l'afegim al conjunt "nodes_in_cliques" per quedar-nos sense cap repetit
    return filtered_cliques, list(nodes_in_cliques) #retornem la llista "filtered_cliques" i el conjunt "nodes_in_cliques" el retornem com una llista
    # ----------------- END OF FUNCTION --------------------- #


def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
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
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == '__main__':
    # Definim els grafs
    gb = nx.read_graphml("BrunoMars_100_BFS.graphml")
    gd = nx.read_graphml("BrunoMars_100_DFS.graphml")
    gb_un = nx.read_graphml("BrunoMars_100_BFS_undirected.graphml")
    gd_un = nx.read_graphml("BrunoMars_100_DFS_undirected.graphml")
    
    # Exercici 1: nodes comuns
    print("# --- Exercici 1 --- #")
    comuns_dir = num_common_nodes(gb, gd)
    comuns_bfs = num_common_nodes(gb, gb_un)
    print(f"Nombre de nodes comuns entre els dos grafs BDS i DFS dirigits: {comuns_dir}")
    print(f"Nombre de nodes comuns entre els dos grafs BDS dirigit i no dirigit: {comuns_bfs}\n")
    
    # Exercici 2: 25 nodes centrals i nodes en comú:
    print("# --- Exercici 2 --- #")
    top_nodes_degree = get_k_most_central(gb_un, metric='degree', num_nodes=25)
    top_nodes_bet = get_k_most_central(gb_un, metric='betweenness', num_nodes=25)
    top_names_degree = ', '.join(get_artist_names(gb_un, top_nodes_degree))
    top_names_bet = ', '.join(get_artist_names(gb_un, top_nodes_bet))
    
    comuns_25 = len(set(top_nodes_bet).intersection(set(top_nodes_degree)))
    print(f"Els 25 nodes amb més degree centrality: {top_names_degree}")
    print(f"Els 25 nodes amb més betweenness centrality: {top_names_bet}\n")
    print(f"En total hi ha {comuns_25} nodes que comparteixen els dos sets\n")
    
    # Exercici 3: buscar cliques
    print("# --- Exercici 3 --- #")
    min_clique_size = determine_min_clique_size(gb_un, gd_un)
    
    cliques_b, nodes_in_cliques_b = find_cliques(gb_un, min_size_clique=min_clique_size)
    cliques_d, nodes_in_cliques_d = find_cliques(gd_un, min_size_clique=min_clique_size)
    
    print(f"Nombre de cliques trobades en g'B (mida mínima {min_clique_size}): {len(cliques_b)}")
    print(f"Nombre de nodes únics a g'B: {len(nodes_in_cliques_b)}")
    print(f"Nombre de cliques trobades en g'D (mida mínima {min_clique_size}): {len(cliques_d)}")
    print(f"Nombre de nodes únics a g'D: {len(nodes_in_cliques_d)}")
    
    print(f"Nombre de nodes únics comuns a ambdós grafs: {len(set(nodes_in_cliques_b).intersection(set(nodes_in_cliques_d)))}\n")
    
    # Exercici 4: analitzar cliques
    print("# --- Exercici 4 --- #")
    largest_clique_b = max(cliques_b, key=len)
    largest_clique_d = max(cliques_d, key=len)

    if len(largest_clique_b) >= len(largest_clique_d):
        largest_clique = largest_clique_b
        selected_graph = gb_un
    else:
        largest_clique = largest_clique_d
        selected_graph = gd_un
    artist_names = get_artist_names(selected_graph, largest_clique)
    print("La clique més gran té els següents nodes:")
    print(', '.join(artist_names))

    # Exercici 5: detectar comunitats
    print("\n# --- Exercici 5 --- #")
    method = 'louvain'

    communities, modularity = detect_communities(gd_un, method)
    
    print(f"Comunitats detectades amb el mètode {method}:")
    print(f"Modularitat de la partició: {modularity:.4f}")
    
    if modularity > 0.3:
        print("\nLa partició sembla ser bona, ja que la modularitat és superior a 0.3.\n")
    else:
        print("\nLa partició no és massa bona, ja que la modularitat és inferior a 0.3.\n")
        
    # Exercici 6: anuncis
    print("\n# --- Exercici 6 --- #")
    
    # Part (a) - Cost mínim per a l'anunci
    print("# --- Part (a) --- #")
    min_cost, central_nodes_gb = min_ad_cost(gb)
    num_central_nodes = len(central_nodes_gb)
    print(f"Cost mínim per assegurar-se que l'anunci arribi a un usuari: {min_cost} euros")
    print(f"Total de nodes centrals seleccionats per l'anunci: {num_central_nodes}\n")
    min_cost_gd, central_nodes_gd = min_ad_cost(gd)
    num_central_nodes_gd = len(central_nodes_gd)
    print(f"Cost mínim per assegurar-se que l'anunci arribi a un usuari (graf gD): {min_cost_gd} euros")
    print(f"Total de nodes centrals seleccionats per l'anunci (graf gD): {num_central_nodes_gd}\n")
    
    # Part (b) - Selecció d'artistes amb pressupost de 400 euros
    print("# --- Part (b) --- #")
    best_artists = best_ad_spread(gb)
    best_artist_names = ', '.join(get_artist_names(gb, best_artists))
    print(f"Els millors artistes per la propagació de l'anunci amb 400 euros: {best_artist_names}")
    best_artists_gd = best_ad_spread(gd)
    best_artist_names_gd = ', '.join(get_artist_names(gd, best_artists_gd))
    print(f"Els millors artistes per la propagació de l'anunci amb 400 euros (graf gD): {best_artist_names_gd}\n")
    
    # Exercici 7: artistes preferits
    print("\n# --- Exercici 7 --- #")
    start_artist = "Bruno Mars"
    end_artist = "Alicia Keys"
    
    path = find_shortest_path(gb, start_artist, end_artist)
    path_str = ', '.join(path)

    if path:
        print(f"El camí més curt de {start_artist} a {end_artist} és: {path_str}")
        print(f"Número de hops: {len(path) - 1}")
    else:
        print(f"No hi ha camí entre {start_artist} i {end_artist}.")

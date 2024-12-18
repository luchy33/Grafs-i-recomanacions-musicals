if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #definim els grafs:
    gb = nx.read_graphml("BrunoMars_100_BFS.graphml")
    gd = nx.read_graphml("BrunoMars_100_DFS.graphml")
    gb_un = nx.read_graphml("BrunoMars_100_BFS_undirected.graphml")
    gd_un = nx.read_graphml("BrunoMars_100_DFS_undirected.graphml")
    
    
    # Test de la funció 'num_common_nodes'
    print("# --- Nodes comuns --- #")
    comuns_dir = num_common_nodes(gb, gd)
    comuns_bfs = num_common_nodes(gb, gb_un)
    print(f"Nombre de nodes comuns entre els dos grafs BDS i DFS dirigits: {comuns_dir}")
    print(f"Nombre de nodes comuns entre els dos grafs BDS dirigit i no dirigit: {comuns_bfs}\n")
    
    # Test de la funció 'get_degree_distribution'
    print("# --- Distribució de graus --- #")
    degree_dist = get_degree_distribution(gb)
    print(f"Distribució de graus del graf: {degree_dist}\n")
    
    # Test de la funció 'get_k_most_central'
    print("# --- Nodes més centrals --- #")
    top_nodes_degree = get_k_most_central(gb, metric='degree', num_nodes=25)
    top_nodes_bet = get_k_most_central(gb, metric='betweenness', num_nodes=25)
    print(f"Els 25 nodes amb més degree centrality: {top_nodes_degree}")
    print(f"Els 25 nodes amb més betweenness centrality: {top_nodes_bet}\n")
    
    
    # Test de la funció 'find_cliques'
    print("# --- Cliques del graf --- #")
    min_clique_size = 25
    cliques_b, nodes_in_cliques_b = find_cliques(gb_un, min_size_clique=min_clique_size)
    cliques_d, nodes_in_cliques_d = find_cliques(gd_un, min_size_clique=min_clique_size)
    print(f"Cliques al graf no dirigit BFS amb mida mínima {min_clique_size}: {cliques_b}")
    print(f"Nodes presents a qualsevol clique: {nodes_in_cliques_b}")
    print(f"Cliques al graf no dirigit DFS amb mida mínima {min_clique_size}: {cliques_d}")
    print(f"Nodes presents a qualsevol clique: {nodes_in_cliques_d}\n")
    
    # Test de la funció 'detect_communities' amb 'girvan-newman'
    print("# --- Detecció de comunitats (Girvan-Newman) --- #")
    communities, modularity = detect_communities(gd, method='girvan-newman')
    print(f"Comunitats detectades: {communities}")
    print(f"Modularitat: {modularity:.4f}")
    print()
    
    # Test de la funció 'detect_communities' amb 'louvain'
    print("# --- Detecció de comunitats (Louvain) --- #")
    try:
        communities, modularity = detect_communities(gb, method='louvain')
        print(f"Comunitats detectades: {communities}")
        print(f"Modularitat: {modularity:.4f}")
    except ImportError:
        print("El mòdul 'community' no està instal·lat. No es pot provar l'algoritme Louvain.")
    # ------------------- END OF MAIN ------------------------ #

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
def assign_random_numbers(G):
    for node in G.nodes:
        G.nodes[node]['random_number'] = np.random.random()  # Assigning a random number to each node
    return G

def occupied(G,p):
    for node in G.nodes:
        G.nodes[node]['occupied'] = G.nodes[node]['random_number'] < p
    return G

def find_clusters(G):
    occupied_nodes = [node for node in G.nodes if G.nodes[node]['occupied']]
    clusters = list(nx.connected_components(G.subgraph(occupied_nodes)))
    return clusters

p=0.5
G = nx.triangular_lattice_graph(2,2)
G = assign_random_numbers(G)
G = occupied(G,p)
clusters = find_clusters(G)
labels = {node: f"{G.nodes[node]['occupied']:.2f}" for node in G.nodes}  # Creating labels

pos = {node:G.nodes[node]['pos'] for node in G}
node_colors = [G.nodes[node]['occupied'] for node in G.nodes]
occupied_nodes = [node for node in G.nodes if G.nodes[node]['occupied']]
# Draw the graph
nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, cmap=plt.get_cmap('winter'), node_size=25)
for i, cluster in enumerate(clusters):
        cluster_edges = G.subgraph(cluster).edges()
        nx.draw_networkx_edges(G, pos=pos, edgelist=cluster_edges, edge_color='black',style='solid',width=1.5)
    

selected_nodes = [node for node in G.nodes if G.nodes[node]['pos'] == (0,0)] 
print(selected_nodes['pos'])
plt.axis('square')
plt.tight_layout()
plt.show()
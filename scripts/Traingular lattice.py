import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
def assign_random_numbers(G):
    for node in G.nodes:
        G.nodes[node]['random_number'] = np.random.random()  # Assigning a random number to each node
    return G

def occupied(G,p):
    for node in G.nodes:
        G.nodes[node]['occupied'] = G.nodes[node]['random_number'] <= p
    return G

def find_clusters(G):
    occupied_nodes = [node for node in G.nodes if G.nodes[node]['occupied']]
    clusters = list(nx.connected_components(G.subgraph(occupied_nodes)))
    return clusters

p=0.5
G = nx.triangular_lattice_graph(10,20)
G = assign_random_numbers(G)
G = occupied(G,p)
clusters = find_clusters(G)
print(clusters)
labels = {node: f"{G.nodes[node]['occupied']:.2f}" for node in G.nodes}  # Creating labels

pos = {node:G.nodes[node]['pos'] for node in G}
node_colors = [G.nodes[node]['occupied'] for node in G.nodes]

#nx.draw(G, pos=pos, node_color=node_colors, cmap=plt.cm.binary, with_labels=False, node_size=100)
#plt.axis('equal')
#plt.show()

pos = {node: attrs['pos'] for node, attrs in G.nodes(data=True)}
node_colors = [0 if G.nodes[node]['occupied'] else 1 for node in G.nodes]

for i, cluster in enumerate(clusters):
    nx.draw(G, pos=pos, nodelist=cluster, node_color=node_colors, cmap=plt.get_cmap('tab20c'), with_labels=False, node_size=200, label=f"Cluster {i + 1}")

plt.title('Triangular Lattice with Occupied Node Clusters')
plt.show()
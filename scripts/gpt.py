import networkx as nx
import random
import matplotlib.pyplot as plt

# Create a triangular lattice
rows = 5  # Number of rows
cols = 5  # Number of columns
G = nx.triangular_lattice_graph(rows, cols)

# Assign random numbers and occupation based on probability
p = 0.6  # Probability threshold for occupation

for node in G.nodes():
    random_number = random.uniform(0, 1)  # Generate a random number between 0 and 1
    G.nodes[node]['random_number'] = random_number  # Assign the random number to the node

    occupation = int(random_number < p)  # Assign occupation based on the probability 'p' (1 for 'Occupied', 0 for 'Unoccupied')
    G.nodes[node]['occupation'] = occupation  # Assign the occupation to the node

# Renormalize the lattice based on the average of 3 nodes in a triangle
new_G = nx.Graph()
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    neighbor_occupations = [G.nodes[neighbor]['occupation'] for neighbor in neighbors]
    if len(neighbor_occupations) == 3:
        average_occupation = sum(neighbor_occupations) / 3  # Calculate the average of 3 neighboring nodes
        new_G.add_node(node, occupation=int(average_occupation + 0.5))  # Add nodes with re-normalized occupation

for edge in G.edges():
    if edge[0] in new_G.nodes() and edge[1] in new_G.nodes():
        new_G.add_edge(edge[0], edge[1])  # Add edges to the new lattice

# Draw the renormalized graph with node labels based on occupation
pos = {node: new_G.nodes[node]['pos'] for node in new_G}  # Define node positions
node_labels = {node: new_G.nodes[node]['occupation'] for node in new_G.nodes()}  # Assign node labels based on occupation
nx.draw(new_G, pos, with_labels=True)
nx.draw_networkx_labels(new_G, pos, labels=node_labels)  # Label the nodes based on their occupation
plt.show()
import networkx as nx
import random
import matplotlib.pyplot as plt

# Function to generate a triangular lattice
def generate_triangular_lattice(rows, cols):
    G = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0:
                G.add_node((r, c), pos=(c, -r))
            else:
                G.add_node((r, c), pos=(c + 0.5, -r))
            if c > 0:
                G.add_edge((r, c), (r, c - 1))
            if r > 0:
                if r % 2 == 0:
                    G.add_edge((r, c), (r - 1, c))
                else:
                    if c < cols - 1:
                        G.add_edge((r, c), (r - 1, c + 1))
                    G.add_edge((r, c), (r - 1, c))
    return G

# Function to assign random numbers to each node
def assign_random_numbers(G):
    for node in G.nodes:
        G.nodes[node]['random_number'] = random.random()  # Assigning a random number to each node

# Generating a triangular lattice
rows = 6  # Number of rows
cols = 6  # Number of columns
lattice = generate_triangular_lattice(rows, cols)

# Assigning random numbers to each node
assign_random_numbers(lattice)

# Plotting the graph
pos = {node: attrs['pos'] for node, attrs in lattice.nodes(data=True)}
node_colors = [lattice.nodes[node]['random_number'] for node in lattice.nodes]
nx.draw(lattice, pos=pos, node_color=node_colors, cmap=plt.cm.plasma, with_labels=False, node_size=100)
plt.title('Triangular Lattice with Random Numbers')
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma))
plt.show()
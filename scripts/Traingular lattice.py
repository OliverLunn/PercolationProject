import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

G = nx.triangular_lattice_graph(10,20)
pos = {n: G.nodes[n]['pos'] for n in G}
nx.draw(G,pos)
plt.axis('equal')
plt.show()
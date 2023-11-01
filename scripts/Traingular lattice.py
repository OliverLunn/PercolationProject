import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.spatial as spatial
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
m=5
n=10
G = nx.triangular_lattice_graph(m,n)
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
    
positions = np.asarray(G.nodes)
occs = np.asarray([G.nodes[node]['occupied'] for node in G.nodes])
xs = positions[:,0]
ys = positions[:,1]
len_bot_row = np.count_nonzero(ys==0)
len_sec_row = np.count_nonzero(ys==1)
print(len_bot_row,len_sec_row)
height = np.max(ys)
print(height)

#scan on long row sites
occupations = np.zeros((height+1,len_bot_row))
for j in range(0,int(np.ceil(height/2))):
    off = j*(len_sec_row+len_bot_row)
    occupations[2*j,0:len_bot_row] = occs[off:off+len_bot_row]

#scan short row sites
for i in range(0,int(np.floor(height/2))+1):
    off = len_bot_row + i*(len_sec_row+len_bot_row)
    occupations[2*i+1,0:len_sec_row] = occs[off:off+len_sec_row]


#find on edge triangle occupations
for j in range(0,height//2+1):
    for i in range(0,(len_bot_row+1)//3):
        tri = [occupations[2*j,i*3],occupations[2*j,(i*3)+1],occupations[1+j*2,i*3]]
        new_occ = np.average(tri) > 0.5
#find off edge triangle occupations
for j in range(0,height//2):
    for i in range(0,len_sec_row//3):
        tri = [occupations[1+j*2,1+i*3],occupations[1+j*2,2+i*3],occupations[2+j*2,2+i*3]]
        new_occ = np.average(tri) > 0.5

plt.axis('square')
plt.tight_layout()

pos_array = []
pos = [G.nodes[node]["pos"] for node in G.nodes]
pos = np.append(pos_array,pos)
#print(nx.triangles(G))

#tri = Delaunay(pos_array)
#plt.imshow(tri)
plt.show()
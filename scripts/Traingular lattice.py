import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.spatial as spatial
from scipy import ndimage
import scipy.optimize as opt
from tqdm import tqdm

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

def assign_positions(G):
    nodes = np.asarray(G.nodes)
    i=0
    for node in G.nodes:
        if node[1]%2 == 0:
            G.nodes[node]['pos'] = (nodes[i][0],(np.sqrt(3)/2)*nodes[i][1])
        else:
            G.nodes[node]['pos'] = (0.5+nodes[i][0],(np.sqrt(3)/2)*nodes[i][1])
        i+=1
    return G
def add_edges(G):
    origional_nodes = G.nodes
    for node in G.nodes:
        x=node[0]
        y=node[1]
        if y%2 == 0:
            new_nodes = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        else:
            
            new_nodes = [(x-1,y),(x+1,y),(x,y-1),(x,y+1),(x+1,y+1),(x+1,y-1)]
        for new_node  in new_nodes:
            if new_node in origional_nodes:
                G.add_edge((x,y),new_node)

    new_nodes = G.nodes
    return G

def plot(G,axis,title):

    axis.set_title(title)
    pos = {node:G.nodes[node]['pos'] for node in G}
    node_colors = [G.nodes[node]['occupied'] for node in G.nodes]
    clusters = find_clusters(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, cmap=plt.get_cmap('winter'), node_size=25, ax=axis)
    for i, cluster in enumerate(clusters):
        cluster_edges = G.subgraph(cluster).edges()
        nx.draw_networkx_edges(G, pos=pos, edgelist=cluster_edges, edge_color='black',style='solid',width=1.5,ax=axis)
    axis.axis('square')
    plt.tight_layout()

def renormalise(G,m,n):
    positions = np.asarray(G.nodes)
    occs = np.asarray([G.nodes[node]['occupied'] for node in G.nodes])
    xs = positions[:,0]
    ys = positions[:,1]
    height = m+1
    len_bot_row = (n+1)//2 + 1
    len_sec_row = np.count_nonzero(ys==1)

    occupations = np.zeros((height,len_bot_row))
    for i in range(0,len(occs)):
        occupations[ys[i],xs[i]] = occs[i]

    #find on edge triangle occupations
    on_edge = []
    for j in range(0,(height//2)):
        for i in range (0,(len_bot_row-(len_bot_row//3))//2):
            on_edge = np.append(on_edge,(np.average([occupations[0+j*2,0+i*3],occupations[0+j*2,1+i*3],occupations[1+j*2,0+i*3]]) > 0.5))
    on_edge = np.reshape(on_edge,(j+1,i+1))

    #find off edge triangle occupations
    off_edge = []
    for j in range(0,((height-1)//2)):
        for i in range (0,(len_sec_row-1-((len_sec_row-1)//3))//2):
            off_edge = np.append(off_edge,(np.average([occupations[1+j*2,1+i*3],occupations[1+j*2,2+i*3],occupations[2+j*2,2+i*3]]) > 0.5))
    off_edge = np.reshape(off_edge,(j+1,i+1))

    #genarete new empty graph
    H = nx.triangular_lattice_graph(0,0)
    #add on_edge nodes to new graph
    for j in range(0,len(on_edge[:,0])):
        for i in range(0,len(on_edge[0,:])):
            H.add_node((i,2*j))

            H.nodes[(i,2*j)]['occupied'] = bool(on_edge[j,i])

    #add off edge nodes to new graph
    for j in range(0,len(off_edge[:,0])):
        for i in range(0,len(off_edge[0,:])):
            H.add_node((i,1+2*j))
            
            H.nodes[(i,1+2*j)]['occupied'] = bool(off_edge[j,i])

    H = assign_positions(H)
    H = add_edges(H)
    return H

def clsuter_size(G):
    clusters = find_clusters(G)
    sizes = []
    for i in range(0,len(clusters)):
        sizes = np.append(sizes,len(clusters[i]))
    return sizes

def average_clust_size(G,m,n):
    num_nodes = G.number_of_nodes()
    positions = np.asarray(G.nodes)
    occs = np.asarray([G.nodes[node]['occupied'] for node in G.nodes])
    xs = positions[:,0]
    ys = positions[:,1]
    height = m+1
    len_bot_row = (n+1)//2 + 1
    len_sec_row = np.count_nonzero(ys==1)

    lattice = np.zeros((height,len_bot_row))
    for i in range(0,len(occs)):
        lattice[ys[i],xs[i]] = occs[i]

    labeled_lattice = np.zeros((height,len_bot_row))
    clusters = np.asarray(find_clusters(G))
    for i in range(0,len(clusters)):
        cluster = list(clusters[i])
        for j in range(0,len(cluster)):
            x = cluster[j][0]
            y = cluster[j][1]
            labeled_lattice[y,x] = i+1
    labelList = np.arange(labeled_lattice.max() + 1) #create list of labels
    area = ndimage.sum_labels(lattice, labeled_lattice, labelList)
    # Remove spanning cluster by setting its area to zero
    perc_x = np.intersect1d(labeled_lattice[0,:],labeled_lattice[-1,:])
    perc = perc_x[np.where(perc_x>0)]
    
    if (len(perc)>0):
        area[int(perc[0])] = 0
    S = (sum(area*area))/num_nodes
    return S
        

case = 's'
if case == 's':
    probs=np.arange(0.1,0.7,0.005)
    m=100
    n=100
    runs=25
    S = np.zeros((runs,len(probs)))

    for run in tqdm(range(0,runs)):
        i=0
        G = nx.triangular_lattice_graph(m,n)
        G = assign_random_numbers(G)
        for p in probs:
            G = occupied(G,p)

            S[run,i]  = average_clust_size(G,m,n)
            i += 1

    fig, (ax1,ax2)= plt.subplots(1,2)
    
    ax1.set_xlabel('$P$')
    ax1.set_ylabel('$S(p,L)$')
    ax1.plot(probs,np.average(S,axis=0),color='black',label=f'Average over {runs} runs')
    ax1.vlines(0.5,np.min(np.average(S,axis=0)),np.max(np.average(S,axis=0)),color='black',linestyle='--',label='P_c')
    ax1.legend()

    ax2.set_xlabel('$log(|P-P_c|)$')
    ax2.set_ylabel('$log(S(p.L))$')

    def func(x,m,c):
        return m*x + c
    
    indx = np.where(probs<0.5)
    p_c = 0.5
    xdata = []
    ydata = []
    for i in indx:
        xdata = np.append(xdata,np.log(np.abs(probs[i]-p_c)))
        ydata = np.append(ydata,np.log(np.average(S[:,i],axis=0)))

    ppot,pcov = opt.curve_fit(func,xdata,ydata)
    ax2.plot(xdata,ydata,color='blue')
    ax2.plot(xdata,func(xdata,*ppot),color='black')
    print(ppot)

if case == 'r':
    p=0.5
    m=12
    n=12
    G = nx.triangular_lattice_graph(m,n)
    G = assign_random_numbers(G)
    G = occupied(G,p)

    H = renormalise(G,m,n)
    fig, (ax1,ax2) = plt.subplots(1,2)
    plot(G,ax1,'origional lattice')
    plot(H,ax2,'renormalised lattice')


plt.show()
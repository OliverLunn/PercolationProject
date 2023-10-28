import numpy as np
import matplotlib.pyplot as plt
from random import uniform

def lattice_random(size, p):
    rows, columns = size, size
    lattice = [[0 for i in range(rows)] for j in range(columns)]

    for i in range(rows):
        for j in range(columns):
            lattice[i][j] = uniform(0,1) <= p

    lattice = np.array(lattice)
    lattice = np.where(lattice==0, 0, 1)
    return lattice

def find_cluster(a, b, size):
	
	size = a.shape
	row = np.array([], dtype=np.int64)
	col = np.array([], dtype=np.int64)
	
	for i in range(0, size):
		for j in range(0, size):
			
			if a[i,j] == b:
				row = np.append(row, i)
				col = np.append(col, j)
				
	return [[row],[col]]


def search(i, j, lattice):
    if i > 0 and j > 0:
        up = lattice[i-1, j]
        across = lattice[i, j-1]
    elif i > 0 and j == 0: 
        up = lattice[i-1,j]
        across = 0
    elif i == 0 and j > 0:
        up = 0
        across = lattice[i,j-1]
    else:
        up, across = 0, 0   
    return (up,across)

def label(size, probability):
	
    lattice = lattice_random(size, probability)
    id = 1
    label = np.zeros((size,size))

    for i in range(0,size):
          for j in range(0, size):
                if lattice[i,j]:
                    l_a = search(i,j,lattice)
                    up = l_a[0]
                    across = l_a[1]

                    if up == 0 and across == 0:
                        label[i,j] = id
                        id += 1
                    elif up == 0 and across != 0:
                         label[i,j] = label[i, j-1]
                    elif up != 0 and across == 0:
                         label[i,j] = label[i-1, j]
                    else:
                         label_rest = finder(label, size, i, i-1, j, j-1)
                         label = label_rest
                         label[i,j] = label[i-1,j]
    return label

def finder(lattice, size, a, b, c, d):
    x = lattice[a,b]
    y = lattice[c,d]

    find_clust = find_cluster(lattice, x, size)
    row = find_clust[0]
    col = find_clust[1]
    
    for i in range(0, len(col)):
        aa = row[i]
        bb = row[i]
        lattice[aa, bb] = y

    return lattice

                     
size, p = 200, 0.55
lattice = label(size, p)
print(lattice)
plt.imshow(lattice, cmap="viridis")
plt.show()


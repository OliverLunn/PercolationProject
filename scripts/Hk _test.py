import numpy as np
import matplotlib.pyplot as plt
from random import uniform

class UnionFind:
    def __init__(self, max_labels):
        self.labels = [0] * max_labels
        self.labels[0] = 0
        self.n_labels = max_labels
    def find(self, x):
        y = x
        while self.labels[y] != y:
            y = self.labels[y]
        while self.labels[x] != x:
            z = self.labels[x]
            self.labels[x] = y
            x = z
        return y
    
    def union(self, x, y):
        self.labels[self.find(x)] = self.find(y)
        return self.find(x)
    
    def make_set(self):
        self.labels[0] += 1
        assert self.labels[0] < self.n_labels
        self.labels[self.labels[0]] = self.labels[0]
        return self.labels[0]
    
class percolation():
    def __init__(self,size,p):
        self.size = size
        self.p = p

    def lattice_random(self):
        rows, columns = self.size, self.size

        lattice = [[0 for i in range(rows)] for j in range(columns)]

        for i in range(rows):
            for j in range(columns):
                lattice[i][j] = uniform(0,1) <= self.p

        lattice = np.array(lattice)
        lattice = np.where(lattice==0, 0, 1)
        return lattice
    
def hk(matrix):
    m,n = np.shape(matrix)
    uf = UnionFind(m * n // 2)
    labeled = matrix
    for i in range(m):
        for j in range(n):
            if labeled[i][j] != 0:
                up = labeled[i - 1][j] if i > 0 else 0
                left = labeled[i][j - 1] if j > 0 else 0
                
                if up == 0 and left == 0:
                    labeled[i][j] = uf.make_set()
                elif up == 0 and left != 0:
                    labeled[i][j] = uf.find(left)
                elif up != 0 and left == 0:
                    labeled[i][j] = uf.find(up)
                else:
                    labeled[i][j] = uf.union(up, left)
                print(up,left,uf.union(up,left))
    return labeled

size = 5
p=0.5
perc = percolation(size,p)
lattice = perc.lattice_random()
labeled_lattice = hk(lattice)


fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(lattice)
ax2.imshow(labeled_lattice)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

from random import uniform
from scipy.ndimage import label

def lattice_random(p, size):
    
    """
    generates a 2d lattice of randomly distributed numbers 0 < x < 1.
    A random number < probability is the condition for occupation of a site
    Inputs:
        p : occupation probaility [type : float]
        size : lattice dimensions [type : int]
            only ave capabiity for square lattice currently
    Returns:
        lattice : array of occupied/non-occupied sites
    -------
    """
    
    rows, columns = size, size

    lattice = [[0 for i in range(rows)] for j in range(columns)]
    for i in range(rows):
        for j in range(columns):
            lattice[i][j] = uniform(0,1) <= p
    
    return lattice


def cluster_search(lattice):
    """
    Searches lattice array of occupied sites for clusters.
    
    Inputs:
        lattice : lattice of random numbers
        
    Returns:
        labeled_lattice : lattice with individual clusters labelled 

    """
    labeled_lattice, num = label(lattice)
    return labeled_lattice    
    
def max_cluster(lattice, size):
    """
    Searches lattice array of occupied sites for largest cluster.
    
    Inputs:
        lattice : lattice of occupied/non-occupied sites [2d array]
        size : dimensions of lattice [float]
        
    Returns:
        lmax_cluster : lattice with only max cluster [2d array] 

    """
    
    count = np.bincount(np.reshape(lattice, size*size))
    count[0] = 0
    max_cluster_id = np.argmax(count)
    max_cluster = np.where(lattice == max_cluster_id,1,0)
    
    return max_cluster


if __name__ == '__main__':
    p = 0.59274621  #transition prob
    size = 1000
    
    lattice_1 = lattice_random(p, size)             #generate lattice
    labeled_lattice_1 = cluster_search(lattice_1)           #label lattice
    max_cluster_1 = max_cluster(labeled_lattice_1, size)    #find max cluster
    
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)    #plot 
    ax1.imshow(labeled_lattice_1, cmap="viridis")
    ax2.imshow(max_cluster_1, cmap="binary")
    
    plt.show()
    

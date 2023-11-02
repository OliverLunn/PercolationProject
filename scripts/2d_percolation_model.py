import numpy as np
import matplotlib.pyplot as plt
from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object
    p = 0.6 #transition prob
    size = 2000

    lattice, labeled_lattice, max_cluster = percolation2d.generate(size, p)   #generate lattice
    
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)    #plot 

    y_occupied,x_occupied = np.where(lattice==1)
    ax1.imshow(max_cluster,cmap = "binary")
    ax2.imshow(lattice, cmap = "binary")
    ax3.imshow(labeled_lattice)

    plt.show()
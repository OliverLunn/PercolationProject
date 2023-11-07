import numpy as np
import matplotlib.pyplot as plt
from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object
    p =  0.59274605079210#transition prob
    size = 250

    lattice, labeled_lattice, max_cluster = percolation2d.generate(size, p)   #generate lattice
    
    fig,(ax1,ax2) = plt.subplots(1,2)    #plot 

    y_occupied,x_occupied = np.where(lattice==1)
    ax2.imshow(max_cluster,cmap = "binary")
    ax1.imshow(lattice, cmap = "binary")
    ax1.axis("off")
    ax2.axis("off")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from random import uniform
from scipy.ndimage import label
from matplotlib.widgets import Slider, Button

class Percolation_2D():
    """
    Class contiaing functions for simulating 2 dimensional percolation transitions
    """
    def lattice_rand(p, size):
        
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
    
    def update_lattice(val):
        
        p = prob_slider.val
        ax1.cla()
        ax2.cla()
        
        lattice_1 = Percolation_2D.lattice_rand(p, size)             #generate lattice
        labeled_lattice_1 = Percolation_2D.cluster_search(lattice_1)           #label lattice
        max_cluster_1 = Percolation_2D.max_cluster(labeled_lattice_1, size)    #find max cluster
        
        ax1.imshow(labeled_lattice_1)
        ax2.imshow(max_cluster_1, cmap="binary")
        
    def reset(event):
        prob_slider.reset()

    
if __name__ == '__main__':
    
    p = 0.618  #transition prob
    size = 1000
    plt.ion()
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)    #plot 
    plt.subplots_adjust(bottom=0.35)

    # Make a horizontal slider to control the probability (button to reset to crit prob)
    ax_prob = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    
    prob_slider = Slider(ax=ax_prob, label='Probability',
                         valmin=0.575, valmax=0.615, valinit=p)
    button = Button(ax_reset, 'Reset', hovercolor='0.975')
        
    lattice_1 = Percolation_2D.lattice_rand(prob_slider.val, size)         #generate lattice
    labeled_lattice_1 = Percolation_2D.cluster_search(lattice_1)           #label lattice
    max_cluster_1 = Percolation_2D.max_cluster(labeled_lattice_1, size)    #find max cluster
        
    ax1.imshow(labeled_lattice_1)
    ax2.imshow(max_cluster_1, cmap="binary")
    
    prob_slider.on_changed(Percolation_2D.update_lattice)
    button.on_clicked(Percolation_2D.reset)

    plt.show()
    

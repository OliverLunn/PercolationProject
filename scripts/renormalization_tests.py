import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage

class Percolation_2D:
    """
    Class contiaing functions for simulating 2 dimensional percolation transitions
    """
    def __init__(self,size,p):
        """
        initiliasiation function for the Percloation_2D object
        Inputs:
            size    : Side length of the lattice
            p       : probabilty of site being occupied
        """
        self.size = size
        self.p = p

    def lattice_random(self):
        
        """
        generates a 2d lattice of randomly distributed numbers 0 < x < 1.
        A random number < probability is the condition for occupation of a site
        Inputs:
            None
        Returns:
            lattice : array of occupied/non-occupied sites
        -------
        """
        
        rows, columns = self.size, self.size

        lattice = [[0 for i in range(rows)] for j in range(columns)]
        for i in range(rows):
            for j in range(columns):
                lattice[i][j] = uniform(0,1) <= self.p
        lattice = np.array(lattice) 
        lattice = np.where(lattice==0, -1, 1)
        return lattice


    def cluster_search(self,lattice):
        """
        Searches lattice array of occupied sites for clusters.
        
        Inputs:
            lattice : lattice of random numbers
            
        Returns:
            labeled_lattice : lattice with individual clusters labelled 

        """
        lattice = np.where(lattice==-1, 0, 1)
        labeled_lattice, num = ndimage.label(lattice)
        return labeled_lattice    
        
    def max_cluster(self,lattice):
        """
        Searches lattice array of occupied sites for largest cluster.
        
        Inputs:
            lattice : lattice of occupied/non-occupied sites [2d array]
            
        Returns:
            max_cluster : lattice with only max cluster [2d array] 

        """
        count = np.bincount(np.reshape(lattice, self.size*self.size))
        count[0] = 0
        max_cluster_id = np.argmax(count)
        max_cluster = np.where(lattice == max_cluster_id,1,0)
        
        return max_cluster
    
    def generate(self):
        """
        Function that generates a lattice and finds the maxium cluster size within the lattice
        
        """
        lattice = self.lattice_random()             #generate lattice
        labeled_lattice = self.cluster_search(lattice)           #label lattice
        max_cluster = self.max_cluster(labeled_lattice)    #find max cluster

        return lattice, max_cluster
    
    def coarse_graining(self, b, lattice):
        """
        This function implements a majority rule coarse graining transformation on a lattice of N x N dimensions.
        Inputs:
            b : transformation scaling factor (multiple of 3) [type : Int]
            lattice : an array of lattice values [type: numpy array]

        Returns: 
        scaled_lattice : transformed lattice [type : numpy array]

        """
        size = len(lattice[0,:])
        scaled_lattice = np.zeros((int(size/b),int(size/b)))
        i_new = 0

        for i in range(1,size-1,b):
            j_new = 0
            for j in range(1,size-1,b):
                lattice1 = lattice[i-1:i+2,j-1:j+2]
                norm_lattice = np.sign(np.mean(lattice1))
                scaled_lattice[i_new,j_new] = norm_lattice
                j_new +=1
            i_new+=1
        return scaled_lattice

    def renorm_group(self, b, lattice):
        size = len(lattice[0,:])
        scaled_lattice = np.zeros((int(size/b), int(size/b)))
        
        i_new = 0

        for i in range(0, size+1, b):
            j_new = 0

            for j in range(0, size+1, b):

                lattice1 = lattice[i:i+1,j:j+1]
                
                count = sum(lattice1)
                print(lattice1)
                ab = sum(lattice1[0,:])
                ac = sum(lattice1[:,0])
                array = [ab,ac]

                if sum(count) > 0:
                    norm_lattice = 1
            
                elif sum(count) == 0:
                    if 2 in array:
                        norm_lattice = 1
                    else:
                        norm_lattice = 0
                else:
                    norm_lattice = 0
                scaled_lattice[i_new, j_new] = norm_lattice

                j_new += 1
            i_new += 1

        return scaled_lattice

    


if __name__ == '__main__':

    p = 0.59274605079210  #transition prob
    size = 4
    b = 2 #normalization scaling value
    difference = 1

    fig, (ax1,ax2) = plt.subplots(1,2)
    gen = Percolation_2D(size,p)
    lattice, max_cluster = gen.generate()
    scaled_lattice = gen.renorm_group(b, lattice)
    #scaled_lattice1 = gen.coarse_graining(b, scaled_lattice)

    ax1.imshow(lattice, cmap="binary")
    ax2.imshow(scaled_lattice, cmap="binary")
    #ax3.imshow(scaled_lattice1, cmap="binary")


    plt.show()
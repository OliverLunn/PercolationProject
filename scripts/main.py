import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt

class Percolation2D:
    """
    Class contiaing functions for simulating 2 dimensional percolation transitions
    """
    def __init__(self):
        """
        initiliasiation function for the Percloation_2D object
        Inputs:
            none 
        """

    def lattice_random(self,size, p):
        
        """
        generates a 2d lattice of randomly distributed numbers 0 < x < 1.
        A random number < probability is the condition for occupation of a site
        Inputs:
            size = lattice size [int]
            p = probabiltity    [float]
        Returns:
            lattice : array of occupied/non-occupied sites
        -------
        """
        
        rows, columns = size, size

        lattice = [[0 for i in range(rows)] for j in range(columns)]
        for i in range(rows):
            for j in range(columns):
                lattice[i][j] = uniform(0,1) <= p
        lattice = np.array(lattice) 
        lattice = np.where(lattice==0, -1, 1)
        return lattice

    def cluster_search(self, lattice):
        """
        Searches lattice array of occupied sites for clusters.
        
        Inputs:
            lattice : lattice of random numbers [array]
            
        Returns:
            labeled_lattice : lattice with individual clusters labelled [array]

        """
        lattice = np.where(lattice==-1, 0, 1)
        labeled_lattice, num = ndimage.label(lattice)
        return labeled_lattice    
        
    def max_cluster(self, size, lattice):
        """
        Searches lattice array of occupied sites for largest cluster.
        
        Inputs:
            size : lattice size [int]
            lattice : lattice of occupied/non-occupied sites [2d array]
            
        Returns:
            max_cluster : lattice with only max cluster [2d array] 

        """
        count = np.bincount(np.reshape(lattice, size*size))
        count[0] = 0
        max_cluster_id = np.argmax(count)
        max_cluster = np.where(lattice == max_cluster_id,1,0)
        
        return max_cluster
    
    def generate(self, size, p):
        """
        Function that generates a lattice and finds the maxium cluster size within the lattice
        Inputs:
            size : lattice size [int]
            p : probability [float]
        Outputs:
        Lattice, labeled_lattice, max_cluster
        """
        lattice = self.lattice_random(size, p)
        labeled_lattice = self.cluster_search(lattice)
        max_cluster = self.max_cluster(size, labeled_lattice)

        return lattice, labeled_lattice, max_cluster
    
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

        for i in range(1,size,b):
            j_new = 0
            for j in range(1,size,b):
                lattice1 = lattice[i-1:i+2,j-1:j+2]
                norm_lattice = np.sign(np.mean(lattice1))
                    
                if norm_lattice==0:
                    scaled_lattice[i_new, j_new] = 1
                else:
                    scaled_lattice[i_new,j_new] = norm_lattice
                j_new +=1
            i_new+=1
        return scaled_lattice

    def renorm_group(self, b, lattice):
        """
        Scales a lattice by grouping elements in blocks of size 'b' and applying a normalization rule.

        Inputs:
            b (int): Block size for grouping elements.
            size (int): Size of the lattice along each dimension.
            lattice (numpy.ndarray): Input lattice as a 2D numpy array.

        Returns:
            numpy.ndarray: Scaled lattice with elements grouped and normalized.

        The function takes a lattice represented as a 2D numpy array and groups its elements
        in blocks of size 'b'. For each block, it applies a normalization rule based on the
        sum of elements in the block and specific conditions.
        """
        size = self.size
        scaled_lattice = np.zeros((int(size/b), int(size/b)))
        i_new = 0
        for i in range(0, size-1, b):
            j_new = 0
            for j in range(0, size-1, b):
                
                lattice1 = lattice[i:i+2,j:j+2]
                count = sum(lattice1)

                ab = lattice1[0,0:2]
                ac = lattice1[0:2,0]
                bd = lattice1[1,0:2]
                cd = lattice1[0:2,1]
                array = np.array([sum(ab),sum(ac),sum(bd), sum(cd)])

                if sum(count) > 0:
                    norm_lattice = 1
            
                elif sum(count) == 0:
                    if 2 in array:
                        norm_lattice = 1
                    else:
                        norm_lattice = -1
                else:
                    norm_lattice = -1

                scaled_lattice[i_new, j_new] = norm_lattice
                j_new += 1  
            i_new += 1

        return scaled_lattice
    
    def average_cluster_size(self, labeled_lattice):
        
        clust_id = np.arange(1,np.max(labeled_lattice)+1)
        clust_size = np.zeros(np.max(labeled_lattice))
        
        for id in clust_id:
            clust_size[id-1] = int(len(np.where(labeled_lattice==id)[0]))
        
        cluster_number = np.zeros(int(np.max(clust_size))+1)
        occupation_prob = 0

        for s in clust_size:
            s=int(s)
            cluster_number[s] = cluster_number[s] + 1
            occupation_prob = occupation_prob + (s*cluster_number[s])
        
        average_size = 0
        for s in clust_size:
            s=int(s)
            average_size = average_size + (1/occupation_prob)*(s**2)*cluster_number[s]
        
        return average_size
    
    def renorm_group_prediction(self, probs, array):
        """
        Renormalization group predicition for the 2x2 blocking regime
        """
        for p in probs:
            p_prime = p**4+4*p**3*(1-p)+2*p**2*(1-p)**2
            array = np.append(array, p_prime)
        return array
    
    def func(self, x,gamma):
        return (x)**(-gamma)

    def diverge(self, probs, p_c, array):
        for p in probs:
            zeta = np.abs((p-p_c))**-1.34
            array = np.append(array, zeta)
        return array
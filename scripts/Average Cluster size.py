import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
import scipy.spatial as spatial
from tqdm import tqdm
import scipy.optimize as opt

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
    
    def occupied_ratio(self, lattice):

        occupied = np.count_nonzero(lattice==1)
        non_occupied = np.count_nonzero(lattice==-1)
        ratio = int(occupied)/int(non_occupied)

        return ratio
    
def average_cluster_size(labeled_lattice):

    clust_id = np.arange(1,np.max(labeled_lattice)+1)
    clust_size = np.zeros(np.max(labeled_lattice))
    for id in clust_id:
        clust_size[id-1] =int(len(np.where(labeled_lattice==id)[0]))
    
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
def func(x,gamma):
    return (x)**(-gamma)

def diverge(probs, p_c, array):
    for p in probs:
        zeta = np.abs((p-p_c))**-1.34
        array = np.append(array, zeta)
    return array

if __name__ == '__main__':

    size = 10
    probs = np.arange(0.5927,0.5928,0.000000025)
    reps = 2
    avg_sizes = np.zeros((len(probs),reps))
    p_c = 0.59274621
    zeta_array = []

    for r in tqdm(range(0,reps)):
        i=0
        for p in probs:
            gen = Percolation_2D(size,p)
            lattice = gen.lattice_random()
            labeled_lattice = gen.cluster_search(lattice)
            avg_sizes[i,r] = average_cluster_size(labeled_lattice)
            i += 1
        
 
    fig, (ax1,ax2) = plt.subplots(1,2)

    ax1.plot(probs,np.average(avg_sizes,axis=1))
    ydata = np.average(avg_sizes,axis=1)
    ax1.vlines(p_c,np.min(avg_sizes)-2,np.max(avg_sizes)+2,linestyles='--',color='black')
    #ax1.ylim(np.min(avg_sizes)-2,np.max(avg_sizes)+2)
    ppot,pcov = opt.curve_fit(func,np.abs(probs-p_c),ydata,2.38)
    #print(ppot)
    ax1.plot(probs, func(np.abs(probs-p_c), *ppot))

    ax2.plot(probs, diverge(probs, p_c, zeta_array))
    plt.show()
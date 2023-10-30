import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
<<<<<<< HEAD
import scipy.spatial as spatial
from tqdm import tqdm
=======
import scipy.optimize as opt
>>>>>>> 4d5de2d54d059330907d623e1670c01d94ffd8be

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
<<<<<<< HEAD

if __name__ == '__main__':

    size = 100
    probs = np.arange(0.1,0.99,0.005)
    reps = 5
=======
def func(x,gamma):
    return (x)**(-gamma)
if __name__ == '__main__':

    size = 100
    probs = np.arange(0.55,0.6,0.00025)
    reps = 10
>>>>>>> 4d5de2d54d059330907d623e1670c01d94ffd8be
    avg_sizes = np.zeros((len(probs),reps))

    for r in tqdm(range(0,reps)):
        i=0
        for p in probs:
            gen = Percolation_2D(size,p)
            lattice = gen.lattice_random()
            labeled_lattice = gen.cluster_search(lattice)
            avg_sizes[i,r] = average_cluster_size(labeled_lattice)
            i += 1
<<<<<<< HEAD

    plt.plot(probs,np.average(avg_sizes,axis=1))
    plt.vlines(0.59274621,np.min(avg_sizes)-2,np.max(avg_sizes)+2,linestyles='--')
    plt.ylim(np.min(avg_sizes)-2,np.max(avg_sizes)+2)
=======
        
    p_c = 0.59274621
    indx = np.argwhere(probs<p_c)
    xdata = np.reshape(probs[indx],len(indx))
    ydata = np.reshape(np.average(avg_sizes,axis=1)[indx],len(indx))
    
    plt.plot(probs[indx],np.average(avg_sizes,axis=1)[indx])
    #plt.vlines(p_c,np.min(avg_sizes)-2,np.max(avg_sizes)+2,linestyles='--',color='black')
    #plt.ylim(np.min(avg_sizes)-2,np.max(avg_sizes)+2)
    ppot,pcov = opt.curve_fit(func,np.abs(xdata-p_c),ydata,2.38)
    print(ppot)
    plt.plot(xdata, func(np.abs(xdata-p_c), *ppot))
>>>>>>> 4d5de2d54d059330907d623e1670c01d94ffd8be
    plt.show()
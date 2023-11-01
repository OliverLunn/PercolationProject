import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
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
        lattice = self.lattice_random()
        labeled_lattice = self.cluster_search(lattice)
        max_cluster = self.max_cluster(labeled_lattice)

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

    def renorm_group(self, b, size, lattice):
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
        for p in probs:
            p_prime = p**4+4*p**3*(1-p)+2*p**2*(1-p)**2
            array = np.append(array, p_prime)
        return array

def func(p,a,b):
    return a*p**4+b*p**3*(1-p)
    
if __name__ == '__main__':

    p = 0.59274605079210  #transition prob
    p=0.65
    size, size1 = 50, 25
    b = 2 #normalization scaling value
    rep = 5
    probs = np.arange(0.05,0.995,0.01)
    avg_sizes = np.zeros((len(probs), rep))
    avg_size1 = np.zeros((len(probs), rep))
    renorm_array = []

    '''
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    gen = Percolation_2D(size,p)
    lattice, max_cluster = gen.generate()
    scaled_lattice = gen.renorm_group(b, size, lattice)
    scaled_lattice1 = gen.renorm_group(b, size1, scaled_lattice)
    
    ax1.imshow(lattice, cmap="binary")
    ax2.imshow(scaled_lattice, cmap="binary")
    ax3.imshow(scaled_lattice1, cmap="binary")
    '''

    for r in tqdm(range(0,rep)):
        i=0
        for p in probs:
            gen = Percolation_2D(size,p)
            lattice = gen.lattice_random()
            labeled_lattice = gen.cluster_search(lattice)
            avg_sizes[i,r] = gen.average_cluster_size(labeled_lattice)
            lattice_renorm = gen.renorm_group(b, size, lattice)
            lattice_renorm_lab = gen.cluster_search(lattice_renorm)
            avg_size1[i,r] = gen.average_cluster_size(lattice_renorm_lab)

            i += 1
    
    ydata = np.average(avg_sizes,axis=1)
    ppot,pcov = opt.curve_fit(func,probs,ydata)
    plt.plot(probs, avg_size1/np.max(avg_size1), "k.", label="simulation results")
    plt.plot(probs, probs, "b--", label="$p=p'$")
    plt.plot(probs, gen.renorm_group_prediction(probs, renorm_array), label="R(p)")
    plt.ylabel("Average Cluster Size, $\zeta_p$")
    plt.xlabel("Probability, $p$")
    #plt.plot(probs, func(probs, *ppot)/np.max(avg_size1))
    plt.legend()
    plt.show()

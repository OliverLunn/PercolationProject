import numpy as np
import matplotlib.pyplot as plt
from random import uniform

import scipy.optimize as opt
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
        
        return lattice


    def cluster_search(self,lattice):
        """
        Searches lattice array of occupied sites for clusters.
        
        Inputs:
            lattice : lattice of random numbers
            
        Returns:
            labeled_lattice : lattice with individual clusters labelled 

        """
        labeled_lattice, num = ndimage.label(lattice)
        return labeled_lattice    
        
    def max_cluster(self,lattice):
        """
        Searches lattice array of occupied sites for largest cluster.
        
        Inputs:
            lattice : lattice of occupied/non-occupied sites [2d array]
            
        Returns:
            lmax_cluster : lattice with only max cluster [2d array] 

        """
        
        count = np.bincount(np.reshape(lattice, self.size*self.size))
        count[0] = 0
        max_cluster_id = np.argmax(count)
        max_cluster = np.where(lattice == max_cluster_id,1,0)
        
        return max_cluster
    
    def generate(self):
        """
        
        
        """
        lattice = self.lattice_random()             #generate lattice
        labeled_lattice = self.cluster_search(lattice)           #label lattice
        max_cluster = self.max_cluster(labeled_lattice)    #find max cluster

        return lattice,max_cluster


def f(x,a,c):
    return a*x + c

if __name__ == '__main__':
    N = 5
    p = 0.59274621  #transition prob
    #0.59274621 is the critical prob
    size = 500
    dimensions = np.zeros(N)
    for run in range (0,N):
        gen = Percolation_2D(size,p)
        lattice,max_cluster = gen.generate()    #generate lattice

        #fig = plt.figure()  #plot 
        #ax1 = plt.axes()
        #y_occupied,x_occupied = np.nonzero(lattice)
        #ax1.scatter(x_occupied,y_occupied,color='black',s=0.02)
        
        #ax1.imshow(max_cluster, cmap="binary")
        
        cm = ndimage.center_of_mass(max_cluster)
        #ax1.scatter(int(cm[1]),int(cm[0]),s=0.5,color='red')
        
        mass = np.zeros(int(size/4))
        box_size = np.zeros(int(size/4))
        for i in range(0,int(size/4)):
            mask =np.ones_like(lattice)
            width = 3 + 2*i #must be odd
            offset = (width-1)/2

            mask[int(cm[0] - offset):int(cm[0] + offset + 1), int(cm[1] - offset):int(cm[1] + offset + 1)] = 0

            masked_lattice = np.ma.masked_array(max_cluster,mask)
            mass[i] = len(np.ma.nonzero(masked_lattice)[0])
            if mass[i] == 0:
                mass[i] = 1
            box_size[i] = width

        #fig=plt.figure()
        #ax2 = plt.axes()
        #ax2.plot(np.log(box_size),np.log(mass))
        indexes = np.where(np.gradient(mass,box_size)>0,True,False)
        params,cov = opt.curve_fit(f,np.log(box_size[indexes]),np.log(mass[indexes]))
        #ax2.plot(np.log(box_size), f(np.log(box_size), params[0],params[1]),color='red')
        dimensions[run] = params[0]
        print('looped')
    print(dimensions)
    print(f'fractal dimension is measured as {np.average(dimensions):.3} for a probabilty of {p:.4} over {N} runs')
        
        #plt.show()

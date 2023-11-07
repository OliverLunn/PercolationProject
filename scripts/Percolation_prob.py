import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt

from main import Percolation2D

percolation2D = Percolation2D()

sizes = [50,75,100,150,200,250]
probs=np.linspace(0.4,0.75,50)

perc_prob = np.zeros((len(sizes),len(probs)))

j=0
for size in sizes:
    runs=int(2000*25/size)
    for r in tqdm(range(0,runs)):
        i=0
        lattice = percolation2D.rand_lattice(size)
        for p in probs:
            lattice_occupied= percolation2D.occupied(lattice,size,p)
            perc_prob[j,i] = perc_prob[j,i] + percolation2D.percolation_probability(lattice_occupied)
            i+=1
    
    perc_prob[j,:] = perc_prob[j,:]/runs
    j+=1
np.savetxt('scripts\\percolation probabilty.txt',perc_prob)


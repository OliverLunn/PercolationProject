#import modules
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt

from main import Percolation2D

if __name__ == '__main__':

    percolation2d = Percolation2D() #create class object

    p = 0.59274605079210  #transition prob
    size = 432
    b = 2 #renormalization scaling value
    lattice, labeled_lattice, max_cluster = percolation2d.generate(size, p)
    #coarse graining
    coarse_grain_1 = percolation2d.coarse_graining(3, lattice, size)
    coarse_grain_2 = percolation2d.coarse_graining(3, coarse_grain_1, size/3)

    #block spin renorm
    block_spin_1 = percolation2d.renorm_group(b, size, lattice)
    block_spin_2 = percolation2d.renorm_group(b, size/b, block_spin_1)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig1, (ax4,ax5,ax6) = plt.subplots(1,3)

    ax1.imshow(lattice, cmap="binary")
    ax2.imshow(coarse_grain_1, cmap="binary")
    ax3.imshow(coarse_grain_2, cmap="binary")
    ax4.imshow(lattice, cmap="binary")
    ax5.imshow(block_spin_1, cmap="binary")
    ax6.imshow(block_spin_2, cmap="binary")

    ax1.text(10, 25," N="+str(int(size)), ha='left', va='top', fontsize="20", bbox={'facecolor': 'white', 'pad': 10})
    ax2.text(3, 3,"N="+str(int(size/3)), fontsize="20", bbox={'facecolor': 'white', 'pad': 10})
    ax3.text(1, 1, "N="+str(int(size/9)), fontsize="20", bbox={'facecolor': 'white', 'pad': 10})
    ax4.text(12, size-20, "N="+str(int(size)), fontsize="20", bbox={'facecolor': 'white', 'pad': 10})
    ax5.text(6, size/2-8, "N="+str(int(size/2)), fontsize="20", bbox={'facecolor': 'white', 'pad': 10})
    ax6.text(2, size/4-2, "N="+str(int(size/4)), fontsize="20", bbox={'facecolor': 'white', 'pad': 10})
    
    ax1.set_xlim(0, size)
    ax1.set_ylim(0, size)
    ax2.set_xlim(0, size/3)
    ax2.set_ylim(0, size/3)
    ax3.set_xlim(0, size/9)
    ax3.set_ylim(0, size/9)
    
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    plt.tight_layout()
    plt.show()
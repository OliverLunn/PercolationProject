"""
File contains code to simulate the Bak-Sneppen evolution model.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def bak_sneppen(npoints, max_gen, dist_array, neighbour_size=2, neighbour_prob=1):
    """
    Function that simulates the Bak-Sneppen algorithm

    Inputs:
        npoints : size of population [int]
        max_gen : "lifetime of population" [int]
        neighbour_size : size of neighbour species to be eliminated [int, default=2]
        neighbour_prob : probability that neighbour is eliminated [int:, default=1]

    Outputs:
        [x,ages_start, ages_end]
    
    """
    age_size = 300
    ages = np.zeros(npoints)
    ages_start = np.zeros((age_size, npoints))
    ages_end = np.zeros((age_size, npoints))

    x = np.zeros(max_gen)       #define population
    B = np.random.rand(npoints)     #assign random B(x)
    p1 = 1
    idx_array = [0]

    for t in range(max_gen):        #iterate over time
        ages += 1
        idx = np.argmin(B)  #find min B(x)

        if np.random.rand(1) < p1:

            B[idx] = np.random.rand(1)
            ages[idx] = 0  #reset "age" of point

            dist = abs(idx_array[-1] - idx)
            dist_array = np.append(dist_array, dist)
            idx_array = np.append(idx_array, idx)

        for d in range(1, 1+math.floor(neighbour_size / 2)):
                        if np.random.rand(1) < neighbour_prob:

                            B[(idx + d) % npoints] = np.random.rand(1)
                            ages[(idx + d) % npoints] = 0
                        if np.random.rand(1) < neighbour_prob:
                            B[(idx - d) % npoints] = np.random.rand(1)
                            ages[(idx - d) % npoints] = 0

        x[t] = np.mean(B)

        if t < age_size:
            for i in range(npoints):
                ages_start[t, i] = ages[i]  #store values

        if max_gen - t < age_size:
            for i in range(npoints):
                ages_end[t-max_gen, i] = ages[i]
                
    return [x, ages_start, ages_end, B, dist_array]


if __name__ == '__main__':

    npoints, max_gen = 3000, 100000
    dist_array = []
    [x, ages_start, ages_end, B, dist_array] = bak_sneppen(npoints, max_gen, dist_array)

    fig, (ax1) = plt.subplots(1,1)

    fig1, (ax2,ax3) = plt.subplots(1,2)
    fig2, (ax4) = plt.subplots(1,1)

    ax1.imshow(ages_start / np.max(ages_start), cmap="jet")
    ax1.set_xticks([])
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Generations')
    ax1.set_aspect("equal")

    ax2.plot(range(len(x)), x)
    ax2.set_xlabel("survival time")
    ax2.set_ylabel("fitness barrier, B(x)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax3.plot(B, "k.")
    ax3.hlines(2/3, 0, npoints)
    ax3.set_xlim(0,npoints+1)
    ax3.set_ylim(0,1.1)
    ax3.set_xlabel("points")
    ax3.set_ylabel("fitness barrier, B(x)")

    ax4.hist(B, bins=100)

    plt.show()
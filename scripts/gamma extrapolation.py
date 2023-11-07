import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
#square lattice
sizes = np.asarray([75,100,150,200,250,300])
gammas = np.asarray([-1.41812766,-1.59304725,-1.78615074,-1.91176002,-2.0021365,-2.0473317 ])
errors = np.asarray([0.07023682,0.06313086,0.0572554,0.05130077,0.04267882,0.04240928])

def func (x,m,c):
    return m*np.log(x) + c

ppot,pcov = opt.curve_fit(func,sizes,gammas,sigma=errors)
fig,(ax1) = plt.subplots(1)

ax1.errorbar(sizes,gammas,errors,0,color='black',marker='x',label='data')
ax1.plot(sizes,func(sizes,*ppot),color='blue',label='fit')
ax1.legend()
ax1.set_ylabel('$\gamma$ estimate')
ax1.set_xlabel('Lattice size, L')
print(f'extrapolating to a square lattice of size 500, gamma = {func(500,*ppot)}')

#trangular lattice
sizes = np.asarray([50,75,100,150,200,250])
gammas = np.asarray([-0.78438324, -1.035674,-1.20633426,-1.45864379,-1.6877932,-1.74823582])
errors = np.asarray([0.06884929,0.07351625,0.07123603,0.06674984,0.064343,0.065679])

def func (x,m,c):
    return m*np.log(x) + c

ppot,pcov = opt.curve_fit(func,sizes,gammas,sigma=errors)
fig,(ax2) = plt.subplots(1)

ax2.errorbar(sizes,gammas,errors,0,color='black',marker='x',label='data')
ax2.plot(sizes,func(sizes,*ppot),color='blue',label='fit')
ax2.legend()
ax2.set_ylabel('$\gamma$ estimate')
ax2.set_xlabel('Lattice size, L')
print(f'extrapolating to a triangular lattice of size 500, gamma = {func(500,*ppot)}')
plt.show()
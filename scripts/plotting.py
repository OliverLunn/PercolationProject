import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('data\gamma estimation.txt')
sizes = X[0]
gammas = X[1]
errs = X[2]
S = np.loadtxt('data\Triangular lattice average size.txt')
probs=np.linspace(0.3,0.7,100)
fig,(ax1) = plt.subplots(1)

colors=['red','orange','yellow','green','blue','purple']
gammas = np.zeros(len(sizes))
errs = np.zeros(len(sizes))
p_c =0.5
indx = np.max(np.where(probs<p_c))
for j in range(len(sizes)):
    ax1.plot(probs,S[j,:],color = colors[j],label=f'L={int(sizes[j])}')

ax1.vlines(0.5,np.min(S[j,:]),np.max(S[j,:]),color='black',linestyle='--',label='P_c')
ax1.legend(fontsize="18")
ax1.tick_params(axis="x", labelsize=18)
ax1.tick_params(axis="y", labelsize=18)
ax1.set_xlabel('$P$',fontsize="22")
ax1.set_ylabel('$S(p,L)$',fontsize="22")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

sizes = [50,75,100,200,250]
probs=np.linspace(0.4,0.75,100)
S = np.loadtxt('data\percolation probabilty.txt')
fig,(ax1,ax2) = plt.subplots(1,2)

colors=['red','orange','yellow','green','blue','purple']
gammas = np.zeros(len(sizes))
errs = np.zeros(len(sizes))
p_c =0.5

for j in range(len(sizes)):
    ax1.plot(probs,S[j,:],color = colors[j],label=f'L={int(sizes[j])}')



probs1=np.linspace(0.3,0.7,100)
S1 = np.loadtxt('data\percolation probabilty.txt')
for j in range(len(sizes)):
    ax2.plot(probs1,S1[j,:],color = colors[j],label=f'L={int(sizes[j])}')



#ax1.vlines(0.59274605079210,np.min(S[j,:]),np.max(S[j,:]),color='black',linestyle='--',label='P_c')
ax1.legend(fontsize="15",loc=4)
ax1.tick_params(axis="x", labelsize=12)
ax1.tick_params(axis="y", labelsize=12)
ax1.set_xlabel('$P$',fontsize="20")
ax1.set_ylabel('$\Pi(p,L)$',fontsize="20")
ax1.text(0.4,1150,'A',fontsize=20,color ='black')

ax2.legend(fontsize="15",loc=4)
ax2.tick_params(axis="x", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)
ax2.set_xlabel('$P$',fontsize="20")
ax2.set_ylabel('$\Pi(p,L)$',fontsize="20")
ax2.text(0.3,327,'B',fontsize=20,color ='black')

plt.show()
plt.tight_layout


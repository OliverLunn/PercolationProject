import matplotlib.pyplot as plt
import numpy as np

def diverge(probs, p_c, array):
    for p in probs:
        zeta = np.abs((p-p_c))**-1.34
        array = np.append(array, zeta)
    return array
    
def renorm_group(probs, array):
    for p in probs:
        p_prime = p**4+4*p**3*(1-p)+2*p**2*(1-p)**2
        array = np.append(array, p_prime)
    return array

fig, (ax1,ax2) = plt.subplots(1,2)
p=0.59274605079210
interval=0.00025
probs = np.arange(0.0,1-interval, interval)
zeta_array = np.array([])
renorm_array = np.array([])
ax1.plot(probs, diverge(probs, p, zeta_array))
ax1.vlines(p,0,1000, linestyles='--', color='black', label="$p_c$")
ax1.set_ylim(0,200)
ax1.set_ylabel("Mean cluster size, $\zeta_p$")
ax1.set_xlabel("Probabilty,p")
ax2.plot(probs, renorm_group(probs, renorm_array), label="R(p)")
ax2.plot(probs, probs,"k--", label="p")
ax2.set_ylabel("R(p)")
ax2.set_xlabel("Probabilty,p")


plt.legend()
plt.show()
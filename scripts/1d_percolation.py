import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    gamma_array = []
    probs = np.arange(0.1,1,0.0001)

    def gamma(probs, L, gamma_array):
        for p in probs:
           gamma = p**L
           gamma_array = np.append(gamma_array, gamma)

        return gamma_array

    ax1 = plt.axes()
    ax1.plot(probs, gamma(probs, 1, gamma_array), label="L=1")
    ax1.plot(probs, gamma(probs, 2, gamma_array), label="L=2")
    ax1.plot(probs, gamma(probs, 3, gamma_array), label="L=3")
    ax1.plot(probs, gamma(probs, 5, gamma_array), label="L=5")
    ax1.plot(probs, gamma(probs, 20, gamma_array), label="L=20")
    ax1.plot(probs, gamma(probs, 50, gamma_array), label="L=50")
    ax1.plot(probs, gamma(probs, 100, gamma_array), label="L=100")
    ax1.legend()
    ax1.set_ylabel("$\Pi(p,L)$")
    ax1.set_xlabel("Occupation probabilty, p")
    plt.show()
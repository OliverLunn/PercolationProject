import numpy as np
import matplotlib.pyplot as plt

def initial_lattice(L):
    lattice = np.zeros(L**2)
    initial_infected = 1
    for i in range(initial_infected): 
        lattice[i] = 1
    np.random.shuffle(lattice)
    lattice = lattice.reshape((L,L))
    return lattice

def rand_site(L):
    x=np.random.randint(0,L)
    y=np.random.randint(0,L)
    return x,y

def list_infected(lattice):
    pos = np.where(lattice==1)
    I = np.stack(pos,axis=1)
    return I

def pick_neigh(x,y):
    r=np.random.randint(0,4)
    x = x + neigh_choices[0,r]
    y = y + neigh_choices[1,r]
    return x,y

L = 10 #size of lattice
lattice = initial_lattice(L)

neigh_choices = np.array([[1,0,-1,0],[0,1,0,-1]])

lam=2
c=1/(1+lam)

I = list_infected(lattice)
while len(I) != L**2:
    I = list_infected(lattice)
    if len(I) <= 0:
        print('No more infected')
        break
    i = np.random.randint(0,len(I))
    y,x = I[i]
    rand = np.random.random()
    if rand < c:
        lattice[y,x] = -1 #recovered
    else:
        neigh_x,neigh_y = pick_neigh(x,y)
        lattice[neigh_y%L,neigh_x%L] = 1 #infected


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:55:15 2024

@author: ruira
"""

import numpy as np
import matplotlib.pyplot as plt



def transitionFunctionValues(t,h):
    '''
    Calculates all the possible values for the transition function based on the spin of the central point and
    the spins of its four neighbours. Stores them in an array
   
    t : reduced temperature
    h : reduced external magnetic field

    Returns: array of possible values for the transition function 
    '''
    delta = []
    for d in range(-4,6,2):
        delta.append([d-h, d+h])
    
    values = []
    for n in delta:
        if n[0] <= 0:
            i = 1
        else:
            i = np.exp(-2*n[0]/t)
        if n[1] <= 0:
            j = 1
        else:
            j = np.exp(-2*n[1]/t)
        values.append([i,j])
    
    return np.array(values)





def init(size, flag=-1):
    '''
    Initializes two square grids
    
    size : size of the grid
    flag : -1 to start with all spins down, 1 to start with spin up

    Returns: grids with dimension size**2
    '''
    if flag == -1:
        grid1 = np.full((size,size),-1)
        grid2 = np.full((size,size),-1)
    elif flag==1:
        grid1 = np.full((size,size),1)
        grid2 = np.full((size,size),1)
        
    return grid1, grid2




def cycle(gridOld, gridNew, size, t, h, w):    
    '''Does a full cycle, meaning it iterates through all the points in the grid and either flips or not based on the
    probability of fliping (transition function) given the spins of the neibhours. Does this updating on the new grid.
    
    (x+1)%10 is equal to x for x=1:8, equal to 0 for x=9.
    (sum_neib/2 + 2) maps from -4,-2,0,2,4 to 0,1,2,3,4 so that we acess the transitionFunctionValues array in the 
    correct spot.
    (spin/2 + 1/2) maps from -1,1 to 0,1 so that we acess the right position inside the spot we acessed with sum_neib
     
    grid : our grid
    size : size of the grid
    t: reduced temperature
    h: reduced external magnetic field
    w: transition function values

    Returns: grid after cycle
    '''
    for x in range(size):
        for y in range(size):
            spin = gridOld[x,y] 
            sum_neib = (gridOld[(x+1)%size, y] + gridOld[x-1, y] + gridOld[x, (y+1)%size] + gridOld[x, y-1]) * spin
            
            if np.random.random() < w[int(sum_neib/2 + 2)][int(spin/2 + 1/2)]:
                gridNew[x,y] = -spin  
            else:
                gridNew[x,y] = spin
    
    return gridOld, gridNew



def sim(size, num_cycles, t, h):
    '''
    Performs n cycles and calculates the magnetic momentum in each, storing it in an array. Keeps switching between 
    new and old grid, so that the spins in the old grid are only updated in the new grid
    
    size : size of the grid
    num_cycles : number of cycles we do
    t : reduced temperature
    h : reduced external magnetic field

    Returns: the array containing the magnetic momenta in each cycle, and the array containing
    the total energy in each cycle
    '''
    gridOld = init(size)[0]
    gridNew = init(size)[1]
    w = transitionFunctionValues(t, h)
    mag_momentum = np.zeros(num_cycles)
    energy = np.zeros(num_cycles)
    
    for i in range(num_cycles): 
        gridOld, gridNew = cycle(gridOld, gridNew, size, t, h, w)
        mag_momentum[i] = abs(2*np.sum(gridNew==1) - size**2)
        
        top = np.roll(gridNew, 1, axis=0)
        bottom = np.roll(gridNew, -1, axis=0)
        left = np.roll(gridNew, 1, axis=1)
        right = np.roll(gridNew, -1, axis=1)
        sum_neib = top + bottom + left + right
        e = -0.5*sum_neib*gridNew - gridNew*h
        
        energy[i] = np.sum(e)
        gridOld, gridNew = gridNew, gridOld
        
    mag_momentum = mag_momentum / (size**2)
    energy = energy / (size**2)
    return mag_momentum, energy

 
#mag_momentum, energy = sim(10, 1000, 1.2, 0)
#plt.plot(mag_momentum) 




###########################################################################################################





def changingT(size, num_cycles, h, start_n, temperatures):
    '''
    Performs a simulation on each temperature from 0.1 to 6 with a step of 0.2. For each simulation it stores
    the relevant variables in an array (average magnetic_momentum, average energy, susceptibility, heat capacity)
    
    size : Tgrid size
    num_cycles : number of cycles per temperature
    h : reduced external magnetic field
    start_n : number of cycles we reject to calculate the mean

    Returns: lists of magnetic_momentum, energy, susceptibility, heat capacity at each temperature
    '''

    n_points = temperatures.size
    mag_list = np.zeros(n_points)
    sus_list = np.zeros(n_points)
    energy_list = np.zeros(n_points)
    cap_list = np.zeros(n_points)
    
    i = 0
    for t in temperatures:
        mag_momentum, energy = sim(size, num_cycles, t, h)
        
        mag_momentum_m = mag_momentum[start_n:].mean()
        energy_m = energy[start_n:].mean()
        sus = (mag_momentum.var() * size**2) / t
        cap = energy.var() / (t**2 * size**2)
        
        mag_list[i] = mag_momentum_m
        energy_list[i] = energy_m
        sus_list[i] = sus
        cap_list[i] = cap
        i+=1
    
    return mag_list, energy_list, sus_list, cap_list
 



def plotting(mag_list, energy_list, sus_list, cap_list, temperatures):
    '''
    Plots magnetic_momentum, energy, susceptibility, heat capacity as a function of reduced temperature
    '''
    
    fig, axs = plt.subplots(2,2)
    
    axs[0,0].plot(temperatures, mag_list)
    axs[0,0].set_xlabel('t')
    axs[0,0].set_ylabel('magnetic momentum')
    
    axs[0,1].plot(temperatures, energy_list)
    axs[0,1].set_xlabel('t')
    axs[0,1].set_ylabel('energy')
    
    axs[1,0].plot(temperatures, sus_list)
    axs[1,0].set_xlabel('t')
    axs[1,0].set_ylabel('magnetic susceptibility')
    
    axs[1,1].plot(temperatures, cap_list)
    axs[1,1].set_xlabel('t')
    axs[1,1].set_ylabel('heat capacity')
    
    plt.tight_layout()
    plt.show()


    
temperatures = np.arange(0.1, 6, 0.1)

mag_list, energy_list, sus_list, cap_list = changingT(10, 1000, 0, 0, temperatures) #size, num_cycles, h, start_n
plotting(mag_list, energy_list, sus_list, cap_list, temperatures)
    
    
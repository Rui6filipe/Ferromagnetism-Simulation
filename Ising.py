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
        delta.append([d-h, d+h]) #delta = [[j + i * h for i in range(-1, 3, 2)] for j in range(-4, 6, 2)]
    
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
        values.append([i,j]) #values = [[1 if elem <= 0 else np.exp(-2 * elem / t) for elem in row] for row in deltaE]
    
    return np.array(values)





def init(size, inicial_state=-1):
    '''
    Initializes the square grid
    
    size : size of the grid
    flag : -1 to start with all spins down, 1 to start with spins up

    Returns: grid with dimension size**2
    '''
    if inicial_state == -1:
        grid = np.full((size,size),-1)
    elif inicial_state == 1:
        grid = np.full((size,size),1)
        
    return grid




def cycle(grid, size, t, h, w):
    '''
    Does a full cycle, meaning it iterates through all the points in the grid and either flips or not based on the
    probability of fliping (transition function) given the spins of the neibhours.
    
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
            spin = grid[x,y] 
            sum_neib = (grid[(x+1)%size, y] + grid[x-1, y] + grid[x, (y+1)%size] + grid[x, y-1]) * spin
            
            if np.random.random() < w[int(sum_neib/2 + 2)][int(spin/2 + 1/2)]:
                grid[x,y] = -spin  

    return grid




def sim(size, num_cycles, t, h, grid_option=None, inicial_state=-1, flag=1):
    '''
    Performs n cycles and calculates the magnetic momentum and energy in each, storinging them
    
    size : size of the grid
    num_cycles : number of cycles we do
    t : reduced temperature
    h : reduced external magnetic field
    grid option: if we want to start the grid in the previous configuration when changing temperature
    inicial_state: inicial spin orientation
    flag: needed because we cant properly see the hysteresis cycle if working with absolute values
    
    Returns: final grid, the array containing the magnetic momenta in each cycle, and the array containing
    the total energy in each cycle
    '''
    if grid_option is not None:
        grid = grid_option
    else:
        grid = init(size, inicial_state)
        
    w = transitionFunctionValues(t, h)
    mag_momentum = np.zeros(num_cycles)
    energy = np.zeros(num_cycles)
    
    for i in range(num_cycles):
        grid = cycle(grid, size, t, h, w)
        
        if (flag==1):
            mag_momentum[i] = abs(2*np.sum(grid==1) - size**2)
        else:
            mag_momentum[i] = 2*np.sum(grid==1) - size**2
        
        top = np.roll(grid, 1, axis=0)
        bottom = np.roll(grid, -1, axis=0)
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)
        sum_neib = top + bottom + left + right
        e = -0.5*sum_neib*grid - grid*h
        energy[i] = np.sum(e)
        
    mag_momentum = mag_momentum / (size**2)
    energy = energy / (size**2)
    return grid, mag_momentum, energy




def changingT(size, num_cycles, h, start_n, temperatures, independent=0):
    '''
    Performs a simulation on each temperature value. For each simulation it stores
    the relevant variables in an array (average magnetic_momentum, average energy, susceptibility, heat capacity)
    
    size : Tgrid size
    num_cycles : number of cycles per temperature
    h : reduced external magnetic field
    start_n : number of cycles we reject to calculate the mean
    temperatures: array containing the temperatures to use
    indepedent: if 1 makes it so that the starting grid on a new temperature is the last grid from previous temperature

    Returns: lists of magnetic_momentum, energy, susceptibility, heat capacity at each temperature
    '''
 
    points = temperatures.size
    mag_list = np.zeros(points)
    sus_list = np.zeros(points)
    energy_list = np.zeros(points)
    cap_list = np.zeros(points)
    
    grid_option = None
    
    i = 0
    for t in temperatures:
        grid, mag_momentum, energy = sim(size, num_cycles, t, h, grid_option) 
        
        mag_momentum_m = mag_momentum[start_n:].mean()
        energy_m = energy[start_n:].mean()
        sus = (mag_momentum.var() * size**2) / t 
        cap = energy.var() / (t**2 * size**2)
        
        mag_list[i] = mag_momentum_m
        energy_list[i] = energy_m
        sus_list[i] = sus
        cap_list[i] = cap
        i+=1
        
        if independent==1:
            grid_option = grid
    
    return mag_list, energy_list, sus_list, cap_list
 
 


def plotting(size, num_cycles, h, start_n, temperatures, independent):
    '''
    Plots the relevant variables
    '''
    
    mag_list, energy_list, sus_list, cap_list = changingT(size, num_cycles, h, start_n, temperatures, independent)
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
    
    
    
    
def changingH(fields, size, num_cycles, t, independent):
    '''
    Performs a simulation on each field value. For each simulation it stores the average magnetic_momentum
    
    fields : array containing the fields to use
    indepedent: if 1 makes it so that the starting grid on a new field is the last grid from previous field

    Returns: list of magnetic momenta for the different fields
    '''
    points = fields.size
    mag_list = np.zeros(points)
    grid_option = None
    i = 0
    inicial_state = -1
    
    for h in fields:
        grid, mag_momentum, energy = sim(size, num_cycles, t, h, grid_option, inicial_state, 0) 
        mag_list[i] = mag_momentum[start_n:].mean()
        i+=1
        
        if i > (fields.size/2):
            inicial_state = 1
            
        if independent==1:
            grid_option = grid
    
    return mag_list



def hysterisis(fields, size, num_cycles, temperatures, independent):
    '''
    Plots the magnetic momentum as a function of the external field, for different temperatures.
    '''
    fig, ax = plt.subplots()

    for t in temperatures:
        mag_list = changingH(fields, size, num_cycles, t, independent)
        ax.plot(fields, mag_list, label=f'Temperature {t}')

    ax.set_xlabel('Magnetic Field (h)')
    ax.set_ylabel('Magnetic Momentum')
    ax.legend()
    plt.tight_layout()
    plt.show()




# Meta parameters
size = 10
num_cycles = 500
start_n = 100
h = 0
independent = 0

temperatures = np.arange(0.1, 6, 0.1) 
temperaturesh = np.arange(1, 4, 1) 
array1 = np.arange(-3, 3, 0.5)
array2 = np.arange(3, -3, -0.5)
fields = np.concatenate((array1, array2))

plotting(size, num_cycles, h, start_n, temperatures, independent)
hysterisis(fields, size, num_cycles, temperaturesh, independent)
    
    
    
    
    
    
    
    
    



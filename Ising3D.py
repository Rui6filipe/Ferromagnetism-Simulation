# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:22:27 2024

@author: ruira
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
import time


# Start time
start_time = time.time()


#Functions
def transitionFunctionValues(t,h):
    '''
    Calculates all the possible values for the transition function based on the spin of the central point and
    the spins of its four neighbours. Stores them in an array
    
    Changes for 3D: sum_neib can go from -6 to 6
   
    t : reduced temperature
    h : reduced external magnetic field

    Returns: array of possible values for the transition function 
    '''
    delta = []
    for d in range(-6,8,2):
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



def init(size, inicial_state=-1):
    '''
    Initializes the square grid
    
    size : size of the grid
    flag : -1 to start with all spins down, 1 to start with spins up
    
    Changes for 3D: grid has 3 dimensions

    Returns: grid with dimension size**2
    '''
    if inicial_state == -1:
        grid = np.full((size,size,size),-1)
    elif inicial_state == 1:
        grid = np.full((size,size,size),1)
        
    return grid



def cycle(grid, size, t, h, w):
    '''
    Does a full cycle, meaning it iterates through all the points in the grid and either flips or not based on the
    probability of fliping (transition function) given the spins of the neibhours.
    
    (x+1)%10 is equal to x for x=1:8, equal to 0 for x=9.
    (sum_neib/2 + 3) maps from -6,-4,-2,0,2,4,6 to 0,1,2,3,4,5,6 so that we acess the transitionFunctionValues 
    array in the correct spot.
    (spin/2 + 1/2) maps from -1,1 to 0,1 so that we acess the right position inside the spot we acessed with sum_neib
     
    Changes for 3D: iterate through z. Change the mapping function sum_neib
        
    grid : our grid
    size : size of the grid
    t: reduced temperature
    h: reduced external magnetic field
    w: transition function values

    Returns: grid after cycle
    '''
    for x in range(size):
        for y in range(size):
            for z in range(size):
                spin = grid[x,y,z]
                sum_neib = (grid[(x+1)%size, y,z] + grid[x-1, y,z] + grid[x, (y+1)%size,z] + grid[x, y-1,z]
                            + grid[x, y, (z+1)%size] + grid[x, y, z-1]) * spin 
                
                if np.random.random() < w[int(sum_neib/2 + 3)][int(spin/2 + 1/2)]:
                    grid[x,y,z] = -spin  

    return grid




def sim(size, num_cycles, t, h, grid_option=None, inicial_state=-1, flag=1):
    '''
    Performs n cycles and calculates the magnetic momentum and energy in each, storinging them.
    Calculates the energy by creating a grid where each point contains the sum_neib of that point in the original grid,
    and uses the grids for the calculation of the energy at each point
    
    size : size of the grid
    num_cycles : number of cycles we do
    t : reduced temperature
    h : reduced external magnetic field
    grid option: if we want to start the grid in the previous configuration when changing temperature
    inicial_state: inicial spin orientation
    flag: needed because we cant properly see the hysteresis cycle if working with absolute values
    
    Change for 3D: size**3. Roll the grid in 3rd dimension
    
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
            mag_momentum[i] = abs(np.sum(grid))
        else:
            mag_momentum[i] = np.sum(grid)
        
        top = np.roll(grid, 1, axis=0)
        bottom = np.roll(grid, -1, axis=0)
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)
        closer = np.roll(grid, 1, axis=2)
        further = np.roll(grid, -1, axis=2)
        
        sum_neib = top + bottom + left + right + closer + further
        e = -0.5*sum_neib*grid - grid*h
        energy[i] = np.sum(e)
        
    mag_momentum = mag_momentum / (size**3)
    energy = energy / (size**3)
    return grid, mag_momentum, energy



def changingT(size, num_cycles, h, start_n, temperatures, independent=0):
    '''
    Performs a simulation on each temperature value. For each simulation it stores
    the relevant variables in an array (average magnetic_momentum, average energy, susceptibility, heat capacity)
    
    size : grid size
    num_cycles : number of cycles per temperature
    h : reduced external magnetic field
    start_n : number of cycles we reject to calculate the mean
    temperatures: array containing the temperatures to use
    indepedent: if 1 makes it so that the starting grid on a new temperature is the last grid from previous temperature

    Changes for 3D: size**3

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
        sus = (mag_momentum[start_n:].var() * size**3) / t 
        cap = energy[start_n:].var() / (t**2 * size**3)
        
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
    
    
    
    
def changingH(fields, size, num_cycles, temperatures, start_n, independent):
    '''
    Performs a simulation on each field value. For each simulation it stores the average magnetic_momentum
    
    fields : array containing the fields to use
    indepedent: if 1 makes it so that the starting grid on a new field is the last grid from previous field

    Returns: list of magnetic momenta for the different fields
    '''
    points = fields.size
    mag_lists = [] 
    
    for t in temperatures:
        mag_list = np.zeros(points)
        grid_option = None
        i = 0
        inicial_state = -1
        for h in fields:
            grid, mag_momentum, energy = sim(size, num_cycles, t, h, grid_option, inicial_state, 0) 
            mag_list[i] = mag_momentum[start_n:].mean()
            i+=1
            
            if i >= (fields.size/2):
                inicial_state = 1
            if independent==1:
                grid_option = grid
                
        mag_lists.append(mag_list)
        
    return np.array(mag_lists)



def hysterisis(fields, size, num_cycles, temperatures, start_n, independent):
    '''
    Plots the magnetic momentum as a function of the external field, for different temperatures.
    '''
    fig, ax = plt.subplots()
    
    mag_lists = changingH(fields, size, num_cycles, temperatures, start_n, independent)
    for i, t in enumerate(temperatures):
        mag_list = mag_lists[i]
        ax.plot(fields, mag_list, label=f'Temperature {t}')

    ax.set_xlabel('Magnetic Field (h)')
    ax.set_ylabel('Magnetic Momentum')
    ax.legend()
    plt.tight_layout()
    plt.show()




#Meta parameters
size = 10
num_cycles = 100
start_n = 10
h = 0
independent = 0
temperatures = np.arange(1, 8, 0.1) 
temperaturesh = np.arange(2, 7, 1) 
array1 = np.arange(-4, 4.5, 0.5)
array2 = np.arange(4, -4.5, -0.5)
fields = np.concatenate((array1, array2))

#Run and Plot
#_, mag_momentum,_ = sim(10, 1000, 5, 0)
#plt.plot(mag_momentum)
plotting(size, num_cycles, h, start_n, temperatures, independent)
hysterisis(fields, size, num_cycles, temperaturesh, start_n, independent)


# End time
end_time = time.time()
print("Runtime:", end_time - start_time, "seconds")

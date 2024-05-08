# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:31:19 2024

@author: danie
"""

# %%   
#Imports
import numpy as np
import matplotlib.pyplot as plt
import time




# %%
def transitionFunctionValues(t, h):
    '''
    Calculates all the possible values for the transition function based on 
    the spin of the central point and the spins of its four neighbours;
    stores them into an array.
   
    t : reduced temperature
    h : reduced external magnetic field

    Returns: array of possible values for the transition function 
    '''
    transition = np.zeros((7, 2))
    deltas = range(-6, 8, 2)
    for d in range(7):
        numerator = [deltas[d] - h, deltas[d] + h]
        for s in range(2):
            if numerator[s] < 0:
               transition[d][s] = 1
            else:
               transition[d][s] = np.exp( (-2 * numerator[s])/t )
    
    return transition





# %%
def init(size, initial_state=-1):
    '''
    Initializes the 3-dimensional grid
    
    size : size of the grid
    initial_state : -1 to start with all spins down, 1 to start with spins up

    Returns: grid with dimension size**3
    '''
    if initial_state == -1:
        grid = np.full((size,size,size),-1)
    elif initial_state == 1:
        grid = np.full((size,size,size),1)
        
    return grid




# %%
def cycle(grid, size, w):
    '''
    Does a full cycle, meaning it iterates through all the points in the grid 
    and either flips each spin or not based on the probability of flipping 
    (transition function) given the spins of the neighbors.
    
    (x+1)%10 is equal to x for x=1:8 and equal to 0 for x=9.
    (sum_neib/2 + 3) maps from -6,-4,-2,0,2,4,6 to 0,1,2,3,4,5,6 and
    (spin/2 + 1/2) maps from -1,1 to 0,1 ; this way, we pass the correct 
    indexes to the w array, which contains the possible values for the 
    transition function.
        
    grid : previously initialized grid
    size : size of the grid
    w : transition function values array

    Returns: grid after cycle
    '''
    for x in range(size):
        for y in range(size):
            for z in range(size):
                spin = grid[x,y,z]
                sum_neib = (grid[(x+1)%size, y,z] + grid[x-1, y,z] + \
                            grid[x, (y+1)%size,z] + grid[x, y-1,z] + \
                            grid[x, y, (z+1)%size] + grid[x, y, z-1]) * spin 
                
                if np.random.random() < w[int(sum_neib/2 + 3)]\
                                         [int(spin/2 + 1/2)]:
                    grid[x,y,z] = -spin  

    return grid




# %%
def ising(size, num_cycles, t, h, grid_option=None, initial_state = -1, 
          statistics = True):
    '''
    Performs n cycles and calculates the magnetic momentum and energy
    in each, storing them in arrays.
    Calculates the energy by creating a grid where each point contains the 
    sum_neib of that point in the original grid, and uses the grids for the 
    calculation of the energy on each point
    
    size : size of the grid
    num_cycles : number of cycles to be done
    grid option : if we want to start the grid in the previous configuration 
    when changing temperature
    initial_state : initial spin orientation for the whole grid
    statistics : True for Curie temperature and False for hysteresis because for 
    hysteresis, the signed value for the total magnetic momentum must be used 
    and also there is no need to calculate the total energy
    t, h : arguments to be passed to the transitionFunctionValues() function
    only
    
    Returns: final grid, the array containing the magnetic momenta in each 
    cycle, and the array containing the total energy in each cycle
    '''
    if grid_option is not None:
        grid = grid_option
    else:
        grid = init(size, initial_state)
        
    w = transitionFunctionValues(t, h)
    mag_momentum = np.zeros(num_cycles)
    energy = np.zeros(num_cycles)
    
    for i in range(num_cycles):
        grid = cycle(grid, size, w)
        
        if (statistics == True):
            
            mag_momentum[i] = abs(np.sum(grid))
            
            top = np.roll(grid, 1, axis=0)
            bottom = np.roll(grid, -1, axis=0)
            left = np.roll(grid, 1, axis=1)
            right = np.roll(grid, -1, axis=1)
            closer = np.roll(grid, 1, axis=2)
            farther = np.roll(grid, -1, axis=2)
            
            sum_neib = top + bottom + left + right + closer + farther
            
            e = -0.5*sum_neib*grid - grid*h
            energy[i] = np.sum(e)
            
        else:
            mag_momentum[i] = np.sum(grid)
        
    mag_momentum = mag_momentum / (size**3)
    energy = energy / (size**3)
    
    return grid, mag_momentum, energy





# %%
def curie_temp(size, num_cycles, h, start_n, temperatures, independent):
    '''
    Performs an ising simulation on each temperature value, at a fixed external
    field value. For each simulation it stores the relevant variables in an 
    array (average magnetic_momentum, average energy, susceptibility, heat 
    capacity).
    
    size : size of the grid
    start_n : number of initial cycles we reject to calculate the mean
    temperatures : array containing the reduced temperatures to use
    independent : if False, the starting grid on a new temperature is the last
    grid from the simulation of the previous temperature
    num_cycles, h : arguments to be passed to the ising() function only

    Returns: lists of magnetic_momentum, energy, susceptibility, heat capacity
    at each temperature
    '''
    points = temperatures.size
    mag_list = np.zeros(points)
    sus_list = np.zeros(points)
    energy_list = np.zeros(points)
    cap_list = np.zeros(points)
    
    grid_option = None
    
    for i in range(points):
        t = temperatures[i]
        grid, mag_momentum, energy = ising(size, num_cycles, t, h, grid_option) 
        
        mag_momentum_m = mag_momentum[start_n:].mean()
        energy_m = energy[start_n:].mean()
        sus = (mag_momentum[start_n:].var() * size**3) / t 
        cap = energy[start_n:].var() / (t**2 * size**3)
        
        mag_list[i] = mag_momentum_m
        energy_list[i] = energy_m
        sus_list[i] = sus
        cap_list[i] = cap
        
        if independent == False:
            grid_option = grid
    
    return mag_list, energy_list, sus_list, cap_list




def plotting_curie_temp(size, num_cycles, h, start_n, temperatures, 
                        independent = True):
    '''
    Plots the relevant variables for each temperature.
    
    temperatures : array containing the reduced temperatures to use as x axis
    size, num_cycles, h, start_n, independent : arguments to be passed to the
    curie_temp() function only
    '''
    mag_list, energy_list, sus_list, cap_list = \
    curie_temp(size, num_cycles, h, start_n, temperatures, 
               independent)
    
    index = np.argmax(sus_list)
    curie_t = temperatures[index]
    print("Curie Temperature:", round(curie_t, 1))
    
    fig, axs = plt.subplots(2, 2)
    labels = ['magnetic momentum', 'energy', 'magnetic susceptibility',
              'heat capacity']
    data_lists = [mag_list, energy_list, sus_list, cap_list]
    
    for ax, data, label in zip(axs.flatten(), data_lists, labels):
        ax.plot(temperatures, data)
        ax.set_xlabel('t')
        ax.set_ylabel(label)
    
    plt.tight_layout()
    
    
    
    
# %%
def hysteresis(fields, size, num_cycles, temperatures, start_n, independent):
    '''
    Performs a simulation on each field value. For each simulation it stores 
    the average magnetic_momentum
    
    fields : array containing the reduced fields to use
    temperatures : reduced temperatures to be used to make each hysteresis plot
    start_n : number of initial cycles we reject to calculate the mean
    independent : if False, the starting grid on a new temperature is the last
    grid from the simulation of the previous temperature
    size, num_cycles : arguments to be passed to the ising() function only

    Returns: list of magnetic momenta for the different fields
    '''
    points = fields.size
    mag_lists = [] 
    
    for t in temperatures:
        mag_list = np.zeros(points)
        grid_option = None
        i = 0
        initial_state = -1
        for h in fields:
            grid, mag_momentum, energy = ising(size, num_cycles, t, h, 
                                         grid_option, initial_state, False) 
            mag_list[i] = mag_momentum[start_n:].mean()
            i+=1
            
            if i >= (fields.size/2):
                initial_state = 1
                
            if independent == False:
                grid_option = grid
                
        mag_lists.append(mag_list)
        
    return np.array(mag_lists)




def plotting_hysteresis(fields, size, num_cycles, temperatures, start_n, 
                        independent = True):
    '''
    Plots the magnetic momentum as a function of the external field, for 
    different temperatures, representing different hysteresis cycles.
    
    fields : reduced external fields to be used as x axis
    temperatures : different reduced temperatures to plot each hysteresis cycle
    size, num_cycles, start_n, independent : arguments to be passed to the 
    hysteresis() function only
    '''
    fig, ax = plt.subplots()
    
    mag_lists = hysteresis(fields, size, num_cycles, temperatures, start_n, 
                           independent)
    
    for i in range(temperatures.size):
        mag_list = mag_lists[i]
        ax.plot(fields, mag_list, label=f'Temperature {i+1}')

    ax.set_xlabel('Magnetic Field (h)')
    ax.set_ylabel('Magnetic Momentum')
    ax.legend()
    plt.tight_layout()
    

    

# %%
def main():
    
    grid_size = 10
    #start_n = 10 * grid_size
    start_n = 0
    ncycles = 10
    
    temperatures = np.arange(0.1, 9, 0.1)
    h = 0
    
    forward = np.arange(-4, 4.5, 0.5)
    backward = np.arange(4, -4.5, -0.5)
    external_fields = np.concatenate((forward, backward))
    hysteresis_temperatures = np.arange(2, 7, 1)
    
    independent = True
        
    plotting_curie_temp(grid_size, ncycles, h, start_n, temperatures, 
                        independent)
    plotting_hysteresis(external_fields, grid_size, ncycles, 
                        hysteresis_temperatures, start_n, independent)
    end_time = time.time()
    print("Runtime:", end_time - start_time, "seconds")



start_time = time.time()
main()

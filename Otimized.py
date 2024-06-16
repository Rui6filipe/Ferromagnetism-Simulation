# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:25:33 2024

@author: danie
"""

# %%   
#Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from numba import jit




# %%
def transitionFunctionValues(t,h):
    '''
    Calculates all the possible values for the transition function based on 
    the spin of the central point and the spins of its four neighbours;
    stores them into an array.
   
    t : reduced temperature
    h : reduced external magnetic field

    Returns: array of possible values for the transition function 
    '''
    delta = [[d-h, d+h] for d in range(-6, 8, 2)]
    
    values = [[1 if n[0] <= 0 else np.exp(-2*n[0]/t),
               1 if n[1] <= 0 else np.exp(-2*n[1]/t)] for n in delta]
    
    return np.array(values)




# %%
def init(size, initial_state=-1):
    '''
    Initializes the 3-dimensional grid.
    
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
@jit(nopython=True)
def cycle(grid, size, w):
    '''
    Computes a full cycle, meaning it iterates through all the points in the 
    grid and either flips each spin or not based on the probability of flipping 
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
                sum_neib = (grid[(x+1)%size, y,z] + grid[x-1, y,z] + 
                            grid[x, (y+1)%size,z] + grid[x, y-1,z] + 
                            grid[x, y, (z+1)%size] + grid[x, y, z-1]) * spin 
                
                if np.random.random() < w[int(sum_neib/2 + 3)]\
                                         [int(spin/2 + 1/2)]:
                    grid[x,y,z] = -spin  

    return grid




# %%
def ising(size, num_cycles, t, h, initial_state = -1, statistics = True):
    '''
    Performs n cycles and calculates the magnetic momentum and energy
    in each, storing them in arrays.
    Calculates the energy by creating a grid where each point contains the 
    sum_neib of that point in the original grid, and uses the grids for the 
    calculation of the energy on each point.
    
    size : size of the grid
    num_cycles : number of cycles to be computed
    initial_state : initial spin orientation for the whole grid, to be passed to
    the init() function
    statistics : must be True for Curie temperature and False for hysteresis 
    because for hysteresis, the signed value of the total magnetic momentum 
    must be used and also there is no need to calculate the total energy
    t, h : arguments to be passed to the transitionFunctionValues() function
    only
    
    Returns: final grid, the array containing the magnetic momenta in each 
    cycle, and the array containing the total energy in each cycle
    '''
    grid = init(size, initial_state)
        
    w = transitionFunctionValues(t, h)
    mag_momentum = np.zeros(num_cycles)
    energy = np.zeros(num_cycles)
    size_cubed = size**3
    
    for i in range(num_cycles):
        
        grid = cycle(grid, size, w)
        
        if statistics == True:
            mag_momentum[i] = abs(2*np.sum(grid==1) - size_cubed)
            
            sum_neib = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) \
                     + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) \
                     + np.roll(grid, 1, axis=2) + np.roll(grid, -1, axis=2)
                        
            e = -0.5*sum_neib*grid - grid*h
            energy[i] = np.sum(e)
        else:
            mag_momentum[i] = 2*np.sum(grid==1) - size_cubed
        
    mag_momentum /= size_cubed
    energy /= size_cubed
    
    return grid, mag_momentum, energy




# %%
def curie_temp_parallel(args):
    '''
    Performs a single ising simulation on a given set of arguments. Stores the
    relevant variables in arrays (average magnetic_momentum, average energy, 
    susceptibility, heat capacity). This function runs in parallel during the 
    multiprocessing.
    
    The args array contains the following:
    size : size of the grid
    start_n : number of cycles we reject to calculate the mean
    t : reduced temperature
    num_cycles, h : arguments to be passed to the ising() function only
   
    Returns: lists of magnetic_momentum, energy, susceptibility, heat capacity
    at the given temperature
    '''
    size, num_cycles, h, start_n, t = args
    size_cubed = size ** 3
    
    grid, mag_momentum, energy = ising(size, num_cycles, t, h)
    
    mag_momentum_m = mag_momentum[start_n:].mean()
    energy_m = energy[start_n:].mean()
    sus = (mag_momentum[start_n:].var() * size_cubed) / t
    cap = energy[start_n:].var() / (t ** 2 * size_cubed)
    
    return mag_momentum_m, energy_m, sus, cap




def curie_temp_mp(num_processes, size, num_cycles, h, start_n, temperatures):
    '''
    Multiprocessing function for Curie temperature simulation; calls the 
    curie_temp_parallel function in pool and map method for each temperature.
    
    num_processes : number of processes to be run on the multiprocessing
    temperatures : array containing the reduced temperatures to use 
    size, num_cycles, h, start_n : arguments to be passed to the 
    curie_temp_parallel() function only
    
    Returns the arrays containing the relevant variables for each temperature
    '''
    pool = mp.Pool(processes = num_processes)  
    results = pool.map(curie_temp_parallel, [(size, num_cycles, h, start_n, t) 
                                              for t in temperatures])
    mag_list, energy_list, sus_list, cap_list = zip(*results)
    
    return np.array(mag_list), np.array(energy_list), np.array(sus_list), \
           np.array(cap_list)




def plotting_curie_temp(num_processes, size, num_cycles, h, start_n,
                         temperatures):
    '''
    Plots the relevant variables for each temperature.
    
    temperatures : array containing the reduced temperatures to use as x axis
    num_processes, size, num_cycles, h, start_n : arguments to be passed to the
    curie_temp_mp() function only
    '''
    mag_list, energy_list, sus_list, cap_list = \
    curie_temp_mp(num_processes, size, num_cycles, h, start_n, temperatures)
    
    
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
def hysteresis_parallel(args):
    '''
    Performs a simulation for given set of arguments. Stores the average 
    magnetic_momentum in an array. This function runs in parallel during the 
    multiprocessing.
    
    The args array contains the following:
    start_n : number of cycles we reject to calculate the mean
    size, num_cycles, t, h, initial_state : arguments to be passed to the 
    ising() function only
  
    Returns: list of magnetic momenta for the different fields and temperatures
    '''
    size, num_cycles, t, h, start_n, initial_state = args
    
    _, mag_momentum, _ = ising(size, num_cycles, t, h, initial_state, False)
    
    mag_list = mag_momentum[start_n:].mean()
    
    return mag_list   




def hysteresis_mp(num_processes, fields, size, num_cycles, temperatures, 
                  start_n):
    '''
    Multiprocessing function for hysteresis simulation; calls the 
    hysteresis_parallel function in pool and map method for each pair of 
    external field and temperature.
    
    Goes forward from strong negative fields to strong positive fields with the 
    initial spins down; goes backward from strong positive fields to strong 
    negative fields with the initial spins up.
    
    num_processes : number of processes to be run on the multiprocessing
    fields : array containing the reduced fields to use
    temperatures : array containing the reduced temperatures to use
    size, num_cycles, start_n : arguments to be passed to the 
    hysteresis_parallel() function only

    Returns: list of magnetic momenta lists for each (fields, temperature) pair
    '''
    points = fields.size
    half_len = len(fields) // 2
    params_list = [(size, num_cycles, t, h, start_n, -1 if idx < half_len
                    else 1) for t in temperatures
                    for idx, h in enumerate(fields)]

    pool = mp.Pool(processes = num_processes)  
    results = pool.map(hysteresis_parallel, params_list)
    mag_lists = np.array(results).reshape(temperatures.size, points)
    
    return mag_lists




def plotting_hysteresis(num_processes, fields, size, num_cycles, temperatures,
                        start_n):
    '''
    Plots the magnetic momentum as a function of the external field, for 
    different temperatures, representing different hysteresis cycles.
    
    fields : reduced external fields to be used as x axis
    temperatures : different reduced temperatures to plot each hysteresis cycle
    size, num_cycles, start_n, independent : arguments
    to be passed to the hysteresis_mp() function only
    '''
    fig, ax = plt.subplots()

    mag_lists = hysteresis_mp(num_processes, fields, size, num_cycles, 
                              temperatures, start_n)
    
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
    start_n = 10 * grid_size
    ncycles = 1000
    
    temperatures = np.arange(0.1, 9, 0.1)
    h = 0
    
    forward = np.arange(-4, 4.5, 0.5)
    backward = np.arange(4, -4.5, -0.5)
    external_fields = np.concatenate((forward, backward))
    hysteresis_temperatures = np.arange(2, 7, 1)
    
    num_processes = 4
    
    if __name__ == "__main__":
        
        plotting_curie_temp(num_processes, grid_size, ncycles, h, start_n,
                             temperatures)
        plotting_hysteresis(num_processes, external_fields, grid_size, ncycles, 
                            hysteresis_temperatures, start_n)
        end_time = time.time()
        print("Runtime:", end_time - start_time, "seconds")



start_time = time.time()
main()

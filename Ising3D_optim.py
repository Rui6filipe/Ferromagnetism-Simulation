# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:22:27 2024

@author: ruira
"""

# %%   
#Imports
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from numba import jit


# Start time
start_time = time.time()



# %%
def transitionFunctionValues(t,h):
    '''
    Calculates all the possible values for the transition function based on the spin of the central point and
    the spins of its four neighbours. Stores them in an array
    
    Changes for 3D: sum_neib can go from -6 to 6
   
    t : reduced temperature
    h : reduced external magnetic field

    Returns: array of possible values for the transition function 
    '''
    delta = [[d-h, d+h] for d in range(-6, 8, 2)]
    
    values = [[1 if n[0] <= 0 else np.exp(-2*n[0]/t),
               1 if n[1] <= 0 else np.exp(-2*n[1]/t)] for n in delta]
    
    return np.array(values)


# %%
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



# %%
@jit(nopython=True)
def cycle(grid, size, w):
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



# %%
def sim(size, num_cycles, t, h, inicial_state=-1, flag=1):
    '''
    Performs n cycles and calculates the magnetic momentum and energy in each, storinging them.
    Calculates the energy by creating a grid where each point contains the sum_neib of that point in the original grid,
    and uses the grids for the calculation of the energy at each point
    
    size : size of the grid
    num_cycles : number of cycles we do
    t : reduced temperature
    h : reduced external magnetic field
    inicial_state: inicial spin orientation
    flag: needed because we cant properly see the hysteresis cycle if working with absolute values and 
    when doing the hysteresis we are not interested in energies.
    
    Change for 3D: size**3. Roll the grid in 3rd dimension
    
    Returns: final grid, the array containing the magnetic momenta in each cycle, and the array containing
    the total energy in each cycle
    '''
    grid = init(size, inicial_state)
        
    w = transitionFunctionValues(t, h)
    mag_momentum = np.zeros(num_cycles)
    energy = np.zeros(num_cycles)
    size_cubed = size**3
    
    for i in range(num_cycles):
        grid = cycle(grid, size, w)
        if (flag==1):
            mag_momentum[i] = abs(2*np.sum(grid==1) - size_cubed)
            #mag_momentum[i] = abs(np.sum(grid))
            
            sum_neib = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + \
                        np.roll(grid, -1, axis=1) + np.roll(grid, 1, axis=2) + np.roll(grid, -1, axis=2)
                        
            e = -0.5*sum_neib*grid - grid*h
            energy[i] = np.sum(e)
        
        
        else:
            mag_momentum[i] = 2*np.sum(grid==1) - size_cubed
            #mag_momentum[i] = np.sum(grid)
        
    mag_momentum /= size_cubed
    energy /= size_cubed
    return grid, mag_momentum, energy



# %%
# For first graph

def changingT(size, num_cycles, h, start_n, temperatures): # Not being used
    '''
    Performs a simulation on each temperature value. For each simulation it stores
    the relevant variables in an array (average magnetic_momentum, average energy, susceptibility, heat capacity)
    
    size : grid size
    num_cycles : number of cycles per temperature
    h : reduced external magnetic field
    start_n : number of cycles we reject to calculate the mean
    temperatures: array containing the temperatures to use
 
    Changes for 3D: size**3

    Returns: lists of magnetic_momentum, energy, susceptibility, heat capacity at each temperature
    '''
 
    points = temperatures.size
    mag_list = np.zeros(points)
    sus_list = np.zeros(points)
    energy_list = np.zeros(points)
    cap_list = np.zeros(points)
    size_cubed = size**3
    
    i = 0
    for t in temperatures:
        grid, mag_momentum, energy = sim(size, num_cycles, t, h) 
        
        mag_momentum_m = mag_momentum[start_n:].mean()
        energy_m = energy[start_n:].mean()
        sus = (mag_momentum.var() * size_cubed) / t 
        cap = energy.var() / (t**2 * size_cubed)
        
        mag_list[i] = mag_momentum_m
        energy_list[i] = energy_m
        sus_list[i] = sus
        cap_list[i] = cap
        i+=1
    
    return mag_list, energy_list, sus_list, cap_list
 
def simulate_temperature(args):
    '''
    Performs a simulation a given set of arguments. Stores the relevant variables in arrays 
    (average magnetic_momentum, average energy, susceptibility, heat capacity)
    
    size : grid size
    num_cycles : number of cycles per temperature
    h : reduced external magnetic field
    start_n : number of cycles we reject to calculate the mean
    temperature: reduced temperature
   
    Returns: lists of magnetic_momentum, energy, susceptibility, heat capacity at the given temperature
    '''
    size, num_cycles, h, start_n, temperature = args
    size_cubed = size ** 3
    
    grid, mag_momentum, energy = sim(size, num_cycles, temperature, h)
    
    mag_momentum_m = mag_momentum[start_n:].mean()
    energy_m = energy[start_n:].mean()
    sus = (mag_momentum[start_n:].var() * size_cubed) / temperature
    cap = energy[start_n:].var() / (temperature ** 2 * size_cubed)
    
    return mag_momentum_m, energy_m, sus, cap



def changingT_parallel(size, num_cycles, h, start_n, temperatures):
    '''
    Parallel execution of simulate temperature function for the reduced temperatures in the array temperatures
    
    Returns the arrays containing the relevant variables for each temperature
    '''
    pool = mp.Pool()  
    results = pool.map(simulate_temperature, [(size, num_cycles, h, start_n, t) for t in temperatures])
    mag_list, energy_list, sus_list, cap_list = zip(*results)
    pool.close()
    pool.join()
    
    return np.array(mag_list), np.array(energy_list), np.array(sus_list), np.array(cap_list)



def plotting(size, num_cycles, h, start_n, temperatures):
    '''
    Plots the relevant variables for each temperature.
    '''
    mag_list, energy_list, sus_list, cap_list = changingT_parallel(size, num_cycles, h, start_n, temperatures)
    
    fig, axs = plt.subplots(2, 2)
    labels = ['magnetic momentum', 'energy', 'magnetic susceptibility', 'heat capacity']
    data_lists = [mag_list, energy_list, sus_list, cap_list]
    
    for ax, data, label in zip(axs.flatten(), data_lists, labels):
        ax.plot(temperatures, data)
        ax.set_xlabel('t')
        ax.set_ylabel(label)
    
    plt.tight_layout()
    #plt.show()
    
    
    
    
    
    
    
# %% 
# For second graph
def changingH(fields, size, num_cycles, temperatures, start_n): # Not being used
    '''
    Performs a simulation on each field value for different temperatures. For each simulation, 
    it stores the average magnetic_momentum. 
    Goes from strong negative fields to strong positive fields with the starting spins down. 
    Goes from strong positive to strong negative with starting spins up
    
    fields : array containing the fields to use
    temperatures: array containing the temperatures to simulate
  
    Returns: list of magnetic momenta for the different fields and temperatures
    '''
    points = fields.size
    mag_lists = [] 

    for t in temperatures:
        mag_list = np.zeros(points)
        i = 0
        inicial_state = -1
        for h in fields:
            grid, mag_momentum, energy = sim(size, num_cycles, t, h, inicial_state, 0) 
            mag_list[i] = mag_momentum[start_n:].mean()
            i += 1

            if i >= (fields.size/2):
                inicial_state = 1
        
        mag_lists.append(mag_list)
    
    return np.array(mag_lists)


def simulate_field(args):
    '''
    Performs a simulation for given t and h values. Stores the average magnetic_momentum
    
    size : grid size
    num_cycles : number of cycles per temperature
    t: reduced temperature
    h : reduced external magnetic field
    start_n : number of cycles we reject to calculate the mean
    inicial_state: inicial spin orientation
  
    Returns: list of magnetic momenta for the different fields and temperatures
    '''
    size, num_cycles, t, h, start_n, inicial_state = args
    _, mag_momentum,_ = sim(size, num_cycles, t, h, inicial_state, 0)
    mag_list = mag_momentum[start_n:].mean()
    
    return mag_list



def changingH_parallel(fields, size, num_cycles, temperatures, start_n):
    '''
    Parallel execution of the simulate_field function for the t's and h's in the arrays temperatures and fields.
    
    Goes from strong negative fields to strong positive fields with the starting spins down. 
    Goes from strong positive to strong negative with starting spins up
    
    fields : array containing the fields to use
    temperatures: array containing the temperatures to simulate

    Returns: list of magnetic momenta list for each (fields, temperature) pair
    '''
    points = fields.size
    half_len = len(fields) // 2
    params_list = [(size, num_cycles, t, h, start_n, -1 if idx < half_len else 1)
                   for t in temperatures
                   for idx, h in enumerate(fields)]

    pool = mp.Pool()  
    results = pool.map(simulate_field, params_list)
    pool.close()
    pool.join()
    mag_lists = np.array(results).reshape(len(temperatures), points)
    
    return mag_lists



def hysterisis(fields, size, num_cycles, temperatures, start_n):
    '''
    Plots the magnetic momentum as a function of the external field, for different temperatures.
    '''
    fig, ax = plt.subplots()

    mag_lists = changingH_parallel(fields, size, num_cycles, temperatures, start_n)
    for i, t in enumerate(temperatures):
        mag_list = mag_lists[i]
        ax.plot(fields, mag_list, label=f'Temperature {t}')

    ax.set_xlabel('Magnetic Field (h)')
    ax.set_ylabel('Magnetic Momentum')
    ax.legend()
    plt.tight_layout()
    #plt.show()





# %%
# RUN
if __name__ == "__main__":
    
    #Meta parameters
    size = 10
    num_cycles = 1000
    start_n = 10
    h = 0
    num_processes = 8
    temperatures = np.arange(0.1, 9, 0.1) 
    temperatures_h = np.arange(2, 7, 1) 
    array1 = np.arange(-4, 4.5, 0.5)
    array2 = np.arange(4, -4.5, -0.5)
    fields = np.concatenate((array1, array2))
    
    #Run and Plot
    #_, mag_momentum,_ = sim(size, num_cycles, 5, 0)
    #plt.plot(mag_momentum)
    plotting(size, num_cycles, h, start_n, temperatures)
    hysterisis(fields, size, num_cycles, temperatures_h, start_n)
    
    # End time
    end_time = time.time()
    print("Runtime:", end_time - start_time, "seconds")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial
import scipy.linalg

from mpi4py import MPI

#instantiate MPI communicator
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

### Helper Functions ###


# Simple constitutive model
def scalar_force_state_fun(exten_state, weighted_volume, bulk_modulus):
    """Computes the scalar force state.  This is the state-based version of a
       bond based material."""
    return 9.0 * bulk_modulus / weighted_volume * exten_state


# Internal force calculation
def compute_internal_force(pos_x, pos_y, disp_x, disp_y, families,
                           ref_mag_state, weighted_volume, volumes,
                           num_nodes, bulk_modulus):
    """ Computes the peridynamic internal force due to deformations."""

    #Allocate space for internal force
    force_x = np.zeros_like(disp_x)
    force_y = np.zeros_like(disp_y)

    #Compute the deformed positions of the nodes
    def_x = pos_x + disp_x
    def_y = pos_y + disp_y


    #Compute deformation state
    def_state_x = np.array([def_x[families[i]] - def_x[i]
                           for i in range(num_nodes)])
    def_state_y = np.array([def_y[families[i]] - def_y[i]
                           for i in range(num_nodes)])

    #Compute deformation magnitude state
    def_mag_state = np.array(def_state_x * def_state_x +
                             def_state_y * def_state_y) ** 0.5
    print def_mag_state[0]

    #Compute deformation unit state
    def_unit_state_x = def_state_x / def_mag_state
    def_unit_state_y = def_state_y / def_mag_state

    #Compute scalar extension state
    exten_state = def_mag_state - ref_mag_state

    #Compute scalar force state
    scalar_force_state = np.array([scalar_force_state_fun(exten_state[i],
                                  weighted_volume[i], bulk_modulus)
                                  for i in range(num_nodes)])
   
    #Compute the force state
    force_state_x = scalar_force_state * def_unit_state_x
    force_state_y = scalar_force_state * def_unit_state_y

    #Integrate nodal forces 
    for i in range(num_nodes):
        force_x[i] += np.dot(force_state_x[i], volumes[families[i]])
        force_y[i] += np.dot(force_state_y[i], volumes[families[i]])
        force_x[families[i]] -= force_state_x[i] * volumes[i]
        force_y[families[i]] -= force_state_y[i] * volumes[i]

    return [force_x, force_y]


# MPI communicator function
def exchange_grid_variable(my_x, rank, size, horizon, grid_size):
    """Function that performs MPI communication across grid"""

    step = horizon * grid_size

    if rank == 0:
        comm.send(my_x[-step:], dest=1)

    if 0 < rank < (size - 1):
        rows_above_x = comm.recv(source=(rank - 1))
        comm.send(my_x[-step:], dest=(rank + 1))

    if rank == (size - 1):
        rows_above_x = comm.recv(source=(rank - 1))
        comm.send(my_x[:step], dest=(rank - 1))
        #Setup local grid
        return np.r_[rows_above_x, my_x].flatten()

    if 0 < rank < (size - 1):
        comm.send(my_x[:step], dest=(rank - 1))
        rows_below_x = comm.recv(source=(rank + 1))
        #Setup local grid
        return np.r_[rows_above_x, my_x, rows_below_x].flatten()

    if rank == 0:
        rows_below_x = comm.recv(source=1)
        #Setup local grid
        return np.r_[my_x, rows_below_x].flatten()


# Strip off ghost rows before MPI commuincation
def strip_ghost_rows(my_x, rank, size, horizon, grid_size):
    """Function that strips off the overlapped ghost rows, intended to be used
       before exchange_grid_varibles()""" 

    step = horizon * grid_size

    if rank == 0:
        return my_x[:-step]

    if 0 < rank < (size - 1):
        return my_x[step:-step]

    if rank == (size - 1):
        return my_x[step:]

### Main Program ####
GRIDSIZE = 20
HORIZON = 2.

num_nodes = GRIDSIZE * GRIDSIZE

#Grid buffer variables
x_sendbuf = []
y_sendbuf = []
disp_x_sendbuf = []
disp_y_sendbuf = []

if rank == 0:
    print("pop: The Peridynamic OPtimization tool\n")

    #Set up the grid
    grid = np.mgrid[0:GRIDSIZE:1., 0:GRIDSIZE:1.]

    #Partition grid
    chunks = GRIDSIZE / size

    x_partitioned = [grid[0][i * chunks:(i + 1) * chunks] for i in range(size)]
    y_partitioned = [grid[1][i * chunks:(i + 1) * chunks] for i in range(size)]

    disp_x = np.zeros_like(x_partitioned)
    disp_y = np.zeros_like(y_partitioned)
    
    x_sendbuf = x_partitioned
    y_sendbuf = y_partitioned
    disp_x_sendbuf = disp_x
    disp_y_sendbuf = disp_y
else:
    x_sendbuf = None
    y_sendbuf = None
    disp_x_sendbuf = None
    disp_y_sendbuf = None

#Scatter initial positions to all ranks
my_x = comm.scatter(x_sendbuf, root=0)
my_y = comm.scatter(y_sendbuf, root=0)
my_disp_x = comm.scatter(disp_x_sendbuf, root=0)
my_disp_y = comm.scatter(disp_y_sendbuf, root=0)

#Flatten the data before lots of communication happens
my_x = my_x.flatten()
my_y = my_x.flatten() 
my_disp_x = my_disp_x.flatten()
my_disp_y = my_disp_y.flatten()

# Communicate ghost rows and setup local grids
my_x = exchange_grid_variable(my_x, rank, size, HORIZON, GRIDSIZE)
my_y = exchange_grid_variable(my_y, rank, size, HORIZON, GRIDSIZE)
my_disp_x = exchange_grid_variable(my_disp_x, rank, size, HORIZON, GRIDSIZE)
my_disp_y = exchange_grid_variable(my_disp_y, rank, size, HORIZON, GRIDSIZE)
        
#Create the x,y tuples required by scipy.spatial.KDTree
my_nodes = np.array(zip(my_x, my_y), dtype=np.double)

#Get length of nodes on the local rank
my_number_of_nodes = len(my_nodes)

#Create a kdtree to do nearest neighbor search
my_tree = scipy.spatial.KDTree(my_nodes)

#Get all families
my_families = [my_tree.query_ball_point(my_nodes[i], HORIZON, p=2, eps=0.05) 
               for i in range(my_number_of_nodes)]

#Remove node indices from their own families
[fam.remove(ind) for ind, fam in enumerate(my_families) ]

#Initialize dummy volumes
my_volumes = np.ones(my_number_of_nodes, dtype=np.double)

#Compute reference position state of all nodes
my_ref_pos_state_x = np.array([ my_x[my_families[i]] - my_x[i] 
                              for i in range(my_number_of_nodes)])
my_ref_pos_state_y = np.array([ my_y[my_families[i]] - my_y[i] 
                              for i in range(my_number_of_nodes)])

##Compute reference magnitude state of all nodes
my_ref_mag_state = np.array(my_ref_pos_state_x * my_ref_pos_state_x +
                            my_ref_pos_state_y * my_ref_pos_state_y) ** 0.5

#Initialize influence state
my_influence_state = np.array([ np.ones_like(my_families[i]) 
                              for i in range(my_number_of_nodes)])

#Compute weighted volume 
my_weighted_volume = np.array([np.dot(my_influence_state[i]*my_ref_mag_state[i],
    my_ref_mag_state[i]*my_volumes[my_families[i]]) 
    for i in range(my_number_of_nodes) ])

#Time step
TIME_STEP = 1.e-6
VELOCITY = 10.
BULK_MODULUS = 70.e9
RHO = 7800

#Temparary arrays
my_velocity_x = np.zeros_like(my_disp_x)
my_velocity_y = np.zeros_like(my_disp_y)
my_accel_x = np.zeros_like(my_disp_x)
my_accel_y = np.zeros_like(my_disp_y)

#Time stepping loop
for iteration in range(1,2):
    
    #Print a information line
    if rank == 0:
        print "Iter = " + str(iteration) + " , time = " + str(iteration*TIME_STEP)

    #Enforce boundary conditions
    if rank == 0:
        my_disp_x[:HORIZON*GRIDSIZE] = 0.0
        my_disp_y[:HORIZON*GRIDSIZE] = TIME_STEP*VELOCITY 
        my_velocity_x[:HORIZON*GRIDSIZE] = 0.0
        my_velocity_y[:HORIZON*GRIDSIZE] = VELOCITY 
    if rank == (size - 1):
        my_disp_x[-HORIZON*GRIDSIZE:] = 0.0
        my_disp_y[-HORIZON*GRIDSIZE:] = -TIME_STEP*VELOCITY 
        my_velocity_x[-HORIZON*GRIDSIZE:] = 0.0
        my_velocity_y[-HORIZON*GRIDSIZE:] = -VELOCITY 
        
    #Compute the internal force
    my_force_x, my_force_y = compute_internal_force(my_x, my_y, my_disp_x,
                                                    my_disp_y, my_families,
                                                    my_ref_mag_state,
                                                    my_weighted_volume,
                                                    my_volumes,
                                                    my_number_of_nodes,
                                                    BULK_MODULUS)
 
    #Compute the nodal acceleration
    my_accel_x_old = my_accel_x.copy()
    my_accel_y_old = my_accel_y.copy()
    my_accel_x = my_force_x/my_volumes/RHO
    my_accel_y = my_force_y/my_volumes/RHO
    
    #Compute the nodal velocity
    my_velocity_x += 0.5*(my_accel_x_old + my_accel_x)*TIME_STEP
    my_velocity_y += 0.5*(my_accel_y_old + my_accel_y)*TIME_STEP
    
    #Compute the new displacements
    my_disp_x += my_velocity_x*TIME_STEP + 0.5*my_accel_x*TIME_STEP*TIME_STEP
    my_disp_y += my_velocity_y*TIME_STEP + 0.5*my_accel_y*TIME_STEP*TIME_STEP

    #Reshape for MPI communication
    #my_disp_x.reshape((-1, GRIDSIZE))
    #my_disp_y.reshape((-1, GRIDSIZE))

    if iteration == 9:
        print my_disp_x

    #Strip off the ghosted rows
    #my_disp_x = strip_ghost_rows(my_disp_x,rank,size,HORIZON)
    #my_disp_y = strip_ghost_rows(my_disp_y,rank,size,HORIZON)

    #Update new positions across processors
    #my_disp_x = exchange_grid_variable(my_disp_x,rank,size,HORIZON)
    #my_disp_y = exchange_grid_variable(my_disp_y,rank,size,HORIZON)

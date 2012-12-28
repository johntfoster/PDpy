#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import signal

import numpy as np
import scipy.spatial
import matplotlib.delaunay as delaun

from ensight import Ensight

### Peridynamic functions ###


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

### Main Program ####
GRIDSIZE = 30
HORIZON = 3.

#Set up the grid
grid = np.mgrid[0:GRIDSIZE:1., 0:GRIDSIZE:1.]

my_x = np.array(grid[0].flatten(), dtype=np.double)
my_y = np.array(grid[1].flatten(), dtype=np.double)

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
my_disp_x = np.zeros_like(my_x)
my_disp_y = np.zeros_like(my_y)
my_velocity_x = np.zeros_like(my_disp_x)
my_velocity_y = np.zeros_like(my_disp_y)
my_accel_x = np.zeros_like(my_disp_x)
my_accel_y = np.zeros_like(my_disp_y)


#Initialize output files
outfile = Ensight('output', ['displacement'])
outfile.write_geometry_file_timestep(my_x, my_y)
outfile.write_vector_variable_timestep('displacement', 
                                       [my_disp_x,my_disp_y], 0.0)

#Time stepping loop
max_iter = 11
for iteration in range(1,max_iter):
    
    #Print a information line
    time = iteration*TIME_STEP
    print "Iter = " + str(iteration) + " , time = " + str(time)

    #Enforce boundary conditions
    my_disp_x[:HORIZON*GRIDSIZE] = 0.0
    my_disp_y[:HORIZON*GRIDSIZE] = TIME_STEP*VELOCITY 
    my_velocity_x[:HORIZON*GRIDSIZE] = 0.0
    my_velocity_y[:HORIZON*GRIDSIZE] = VELOCITY 
    #
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

    if iteration % 5 == 0 or iteration == (max_iter - 1):
        outfile.write_geometry_file_timestep(my_x, my_y)
        outfile.write_vector_variable_timestep('displacement', 
                                               [my_disp_x,my_disp_y], time)
        outfile.append_timestep(time)


outfile.finalize()

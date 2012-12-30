#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import signal
import math

import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.delaunay as delaun

from progressbar import ProgressBar
from ensight import Ensight

### Peridynamic functions ###

# Simple constitutive model
def scalar_force_state_fun(exten_state, weighted_volume, bulk_modulus, 
        influence_state):
    """Computes the scalar force state.  This is the state-based version of a
       bond based material."""
    return 9.0 * bulk_modulus * influence_state / weighted_volume * exten_state


# Internal force calculation
def compute_internal_force(force_x, force_y, pos_x, pos_y, disp_x, disp_y, 
        families, ref_mag_state, weighted_volume, volumes, num_nodes, 
        bulk_modulus,influence_state):
    """ Computes the peridynamic internal force due to deformations."""

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
        weighted_volume[i], bulk_modulus, influence_state[i]) 
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

    return 


def compute_stable_time_step(families, ref_mag_state, volumes, num_nodes, 
        bulk_modulus,rho,horizon):

    spring_constant = 18.0 * bulk_modulus / math.pi / horizon**4.0

    crit_time_step_denom = np.array([spring_constant * volumes[families[i]] / 
            ref_mag_state[i] for i in range(num_nodes)])**0.5

    critical_time_steps = np.sqrt(2.0 * rho) / crit_time_step_denom
    
    nodal_min_time_step = [ np.amin(item) for item in critical_time_steps ]

    return np.amin(nodal_min_time_step)


def insert_crack(line_eqn, box, tree, horizon, x_pos, y_pos, ref_pos_state_x,
        ref_pos_state_y, influence_state):
    
    #Read in bounding box for crack
    min_x, min_y, max_x, max_y = box
    #Calculate crack length
    crack_length_x = abs(max_x - min_x)
    crack_length_y = abs(max_y - min_y)
    crack_length = np.sqrt(crack_length_x ** 2.0 + crack_length_y ** 2.0)
    
    #Number of discrete points along crack length
    number_points_along_crack = crack_length / horizon * 4.0
    
    #Find the points according to the equation of the line
    j = np.complex(0,1)
    x_points = np.r_[min_x:max_x:number_points_along_crack*j]
    y_points = [ line_eqn(x) for x in x_points ] 

    points_along_crack = zip(x_points,y_points)

    nodes_near_crack = [tree.query_ball_point(point, 1.5*horizon, p=2, 
        eps=0.05) for point in points_along_crack]

    nodes_near_crack_flat = list(set([  elem for iterable in nodes_near_crack 
            for elem in iterable ]))

                
                
    oldsettings = np.seterr(divide='ignore')
    slope_state = ref_pos_state_y / ref_pos_state_x

    #Loop over nodes near the crack to see if any bonds in the nodes family
    #cross the crack path
    for node_index in nodes_near_crack_flat:
        #The slopes for each bond in the family
        bonds_slopes_this_node = slope_state[node_index]
        #Loop over the bonds in the nodes family
        for bond_index,slope in enumerate(bonds_slopes_this_node):
            #Solve for the x location of the cross point
            if slope == nan:
                x_int = scipy.optimize.fsolve(lambda x: slope*(x - 
                    x_pos[node_index]) + y_pos[node_index] - 
                    line_eqn(x),x_pos[node_index])
            else:
                x_int = scipy.optimize.fsolve(lambda x: slope*(x - 
                    x_pos[node_index]) + y_pos[node_index] - 
                    line_eqn(x),x_pos[node_index])
            #Check to see if the x intersection point is within the crack
            #bounding box
            if min_x < x_int < max_x:
                #Find the y intersection point, coorespoinding to the x
                #intersection point
                y_int = line_eqn(x_int)
                #Check to see if the y intersection point is within the crack
                #bounding box
                if min_y < y_int < max_y:
                    #If we got to here that means we need to ``break'' the bond
                    influence_state[node_index][bond_index] = 0.0
    
    return


#####################
### Main Program ####
#####################
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

#INPUTS
TIME_STEP = 1.e-5
#TIME_STEP = None
VELOCITY = 10.
BULK_MODULUS = 70.e9
RHO = 7800
SAFTEY_FACTOR = 0.5
MAX_ITER = 1000
PLOT_DUMP_FREQ = 100
VERBOSE = True

#Temparary arrays
my_disp_x = np.zeros_like(my_x)
my_disp_y = np.zeros_like(my_y)
my_velocity_x = np.zeros_like(my_disp_x)
my_velocity_y = np.zeros_like(my_disp_y)
my_accel_x = np.zeros_like(my_disp_x)
my_accel_y = np.zeros_like(my_disp_y)


#Initialize output files
vector_variables = ['displacement']
outfile = Ensight('output', vector_variables)
#outfile.write_geometry_file_time_step(my_x, my_y)
#outfile.write_vector_variable_time_step('displacement', 
                                       #[my_disp_x,my_disp_y], 0.0)

print("PD.py version 0.1.0\n")
print("Output variables requested:")
for item in vector_variables:
    print("    " + item)

if TIME_STEP == None:
    time_step = SAFTEY_FACTOR*compute_stable_time_step(my_families, 
            my_ref_mag_state, my_volumes, my_number_of_nodes, BULK_MODULUS,
            RHO, HORIZON)
else:
    time_step = TIME_STEP

#insert crack
insert_crack(lambda x: x, [5, 5, 10, 10], my_tree, HORIZON, my_x, my_y, 
        my_ref_pos_state_x, my_ref_pos_state_y, my_influence_state)

print my_influence_state

print("\nRunning...")
if VERBOSE:
    iterable = range(MAX_ITER)
else:
    progress = ProgressBar()
    iterable = progress(range(MAX_ITER))

#Time stepping loop
for iteration in iterable:
    
    #Print a information line
    time = iteration*time_step

    if VERBOSE:
        print("iter = " + str(iteration) + " , time step = " + str(time_step) +
              " , sim time = " + str(time))

    #Enforce boundary conditions
    my_disp_x[:HORIZON*GRIDSIZE] = -time*VELOCITY 
    my_disp_y[:HORIZON*GRIDSIZE] = 0.0
    my_velocity_x[:HORIZON*GRIDSIZE] = -VELOCITY 
    my_velocity_y[:HORIZON*GRIDSIZE] = 0.0
    #
    my_disp_x[-HORIZON*GRIDSIZE:] = time*VELOCITY 
    my_disp_y[-HORIZON*GRIDSIZE:] = 0.0
    my_velocity_x[-HORIZON*GRIDSIZE:] = VELOCITY 
    my_velocity_y[-HORIZON*GRIDSIZE:] = 0.0
        
    #Compute the internal force
    my_force_x = np.zeros(my_number_of_nodes,dtype=np.double)
    my_force_y = np.zeros(my_number_of_nodes,dtype=np.double)
    
    compute_internal_force(my_force_x, my_force_y, my_x, my_y, my_disp_x, 
            my_disp_y, my_families, my_ref_mag_state, my_weighted_volume, 
            my_volumes, my_number_of_nodes, BULK_MODULUS,my_influence_state)
 
    #Compute the nodal acceleration
    my_accel_x_old = my_accel_x.copy()
    my_accel_y_old = my_accel_y.copy()
    my_accel_x = my_force_x/my_volumes/RHO
    my_accel_y = my_force_y/my_volumes/RHO
    
    #Compute the nodal velocity
    my_velocity_x += 0.5*(my_accel_x_old + my_accel_x)*time_step
    my_velocity_y += 0.5*(my_accel_y_old + my_accel_y)*time_step
    
    #Compute the new displacements
    my_disp_x += my_velocity_x*time_step + 0.5*my_accel_x*time_step*time_step
    my_disp_y += my_velocity_y*time_step + 0.5*my_accel_y*time_step*time_step

    #Compute stable time step
    #time_step = compute_stable_time_step(my_x, my_y, my_disp_x, my_disp_y, 
            #my_families, my_ref_mag_state, my_weighted_volume, my_volumes, 
            #my_number_of_nodes, BULK_MODULUS,RHO)

    #Dump plots
    if iteration % PLOT_DUMP_FREQ  == 0 or iteration == (MAX_ITER-1):
        if VERBOSE:
            print "Writing plot file..."
        outfile.write_geometry_file_time_step(my_x, my_y)
        outfile.write_vector_variable_time_step('displacement', 
                                               [my_disp_x,my_disp_y], time)
        outfile.append_time_step(time)
        outfile.write_case_file()


outfile.finalize()

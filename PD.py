#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pdb

import numpy as np
import numpy.ma as ma
import scipy.spatial

from progressbar import ProgressBar
from ensight import Ensight

### Peridynamic functions ###

# Simple constitutive model
def scalar_force_state_fun(exten_state, weighted_volume, bulk_modulus, 
        influence_state):
    """Computes the scalar force state.  This is the state-based version of a
       bond based material."""

    #Return the force
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
    def_state_x = ma.masked_array(def_x[families] - def_x[:,None],
            mask=families.mask)
    def_state_y = ma.masked_array(def_y[families] - def_y[:,None],
            mask=families.mask)

    #Compute deformation magnitude state
    def_mag_state = (def_state_x * def_state_x + 
            def_state_y * def_state_y) ** 0.5

    #Compute deformation unit state
    def_unit_state_x = def_state_x / def_mag_state
    def_unit_state_y = def_state_y / def_mag_state

    #Compute scalar extension state
    exten_state = def_mag_state - ref_mag_state

    #Apply a critical stretch damage model
    influence_state[exten_state > 0.005] = 0.0

    #Compute scalar force state
    scalar_force_state = scalar_force_state_fun(exten_state, weighted_volume,
            bulk_modulus, influence_state) 
   
    #Compute the force state
    force_state_x = scalar_force_state * def_unit_state_x
    force_state_y = scalar_force_state * def_unit_state_y

    #Integrate nodal forces 
    #Sum all the force contribution from j nodes back to i,the sum operation
    #automatically excludes the masked entries
    force_x += np.sum(force_state_x * volumes[families], axis=1)
    force_y += np.sum(force_state_y * volumes[families], axis=1)

    #Subtract the force contribution from i nodes from j, the bincount()
    #operation is a trick to keep it fast in Numpy.  See:
    #<http://stackoverflow.com/questions/9790436/numpy-accumulating-one-array-
    #in-another-using-index-array> for details
    tmp_x = np.bincount(families.compressed(), (force_state_x * 
        volumes[:,None]).compressed()) 
    tmp_y = np.bincount(families.compressed(), (force_state_y * 
        volumes[:,None]).compressed()) 
    force_x[:len(tmp_x)] -= tmp_x
    force_y[:len(tmp_y)] -= tmp_y

    return 


def compute_stable_time_step(families, ref_mag_state, volumes, num_nodes, 
        bulk_modulus,rho,horizon):

    spring_constant = 18.0 * bulk_modulus / math.pi / horizon**4.0

    crit_time_step_denom = np.array([spring_constant * volumes[families[i]] / 
            ref_mag_state[i] for i in range(num_nodes)])**0.5

    critical_time_steps = np.sqrt(2.0 * rho) / crit_time_step_denom
    
    nodal_min_time_step = [ np.amin(item) for item in critical_time_steps ]

    return np.amin(nodal_min_time_step)


def test_line_seg_intersect(line1,line2):
    """Tests to see if two lines segments intersect.  The lines are defined as:

       line1 = [ p0_x, p0_y, p1_x, p1_y ]
       line2 = [ p2_x, p2_y, p3_x, p3_y ]
       
    """
    #See http://stackoverflow.com/questions/563198/how-do-you-detect-where-
    #two-line-segments-intersect for algorithm details
    
    p0_x, p0_y, p1_x, p1_y = line1
    p2_x, p2_y, p3_x, p3_y = line2

    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    denom = (-s2_x * s1_y + s1_x * s2_y)

    num_s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y))
    num_t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x))

    #Detect if lines are parallel or coincident
    if -1e-10 < denom < 1e-10:
        if abs(num_s) - abs(num_t) < 1e-5:
            #Lines are coincident
            return True
        else:
            #Lines are parallel, but not coincident
            return False

    s =  num_s  / denom
    t =  num_t / denom

    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0: 
        #Lines intersect (or meet at endpoints)
        return True
    else:
        #Lines do not intersect
        return False

def insert_crack(crack, tree, horizon, x_pos, y_pos, families,influence_state):
    """
       Inserts crack by setting influence_state to zero for bonds that cross
       crack path.
    """
    
    #Read in crack endpoints
    min_x, min_y, max_x, max_y = crack
    #Calculate crack length
    crack_length_x = max_x - min_x
    crack_length_y = max_y - min_y
    crack_length = np.sqrt(crack_length_x ** 2.0 + crack_length_y ** 2.0)
    
    #Number of discrete points along crack length
    number_points_along_crack = int(math.ceil(crack_length / horizon * 4.0))

    #Find slope of crack line
    slope_denom = max_x - min_x
    j = np.complex(0,1)
    if -1e-10 < slope_denom < 1e-10:
        #Crack line is vertical,discrete points along crack path
        x_points = [min_x for _ in range(number_points_along_crack)]
        y_points = np.r_[min_y:max_y:number_points_along_crack*j]
    else:
        slope = (max_y - min_y) / slope_denom
        line_eqn = lambda x: slope * (x - min_x) + min_y
        #Find the discrete points along crack path
        x_points = np.r_[min_x:max_x:number_points_along_crack*j]
        y_points = [ line_eqn(x) for x in x_points ] 

    #Create a tuple required by the scipy nearest neighbor search
    points_along_crack = zip(x_points,y_points)

    #Find all nodes that could possibly have bonds that cross crack path
    distances, nodes_near_crack = tree.query(points_along_crack, 
            k=MAX_NEIGHBORS_RETURNED, eps=0.0, p=2, 
            distance_upper_bound=2.0*horizon) 
    
    #The search above will produce duplicate neighbor nodes, make them into a
    #unique 1-dimensional list
    nodes_near_crack_flat = list(set([  elem for iterable in nodes_near_crack 
            for elem in iterable ]))

    nodes_near_crack_flat.remove(tree.n)
                
    #Loop over nodes near the crack to see if any bonds in the nodes family
    #cross the crack path
    for node_index in nodes_near_crack_flat:
        #Loop over node family
        for bond_index,end_point_index in enumerate(families[node_index].compressed()):
            #Define the bond line segment as the line between the node and its
            #endpoint.
            bond_line_seg = [ x_pos[node_index], y_pos[node_index], 
                x_pos[end_point_index], y_pos[end_point_index] ]
            #Test for intersection
            if test_line_seg_intersect(crack,bond_line_seg):
                #If we got to here that means we need to ``break'' the bond
                families[node_index][bond_index] = ma.masked
    
    return



#####################
### Main Program ####
#####################
#INPUTS
GRIDSIZE = 100
HORIZON = 3.015
TIME_STEP = 1.e-5
#TIME_STEP = None
VELOCITY = 5.
BULK_MODULUS = 70.e9
RHO = 7800
SAFTEY_FACTOR = 0.5
MAX_ITER = 4000
PLOT_DUMP_FREQ = 100
VERBOSE = False
CRACKS = [[GRIDSIZE/2., -1., GRIDSIZE/2., GRIDSIZE/10.],[GRIDSIZE/2., 9*GRIDSIZE/10. , GRIDSIZE/2., GRIDSIZE+1.],[5.,10,20.,40.]]
MAX_NEIGHBORS_RETURNED = 300

print("PD.py version 0.2.0\n")

#Set up the grid
grid = np.mgrid[0:GRIDSIZE:1., 0:GRIDSIZE:1.]

my_x = np.array(grid[0].flatten(), dtype=np.double)
my_y = np.array(grid[1].flatten(), dtype=np.double)

#Create the x,y tuples required by scipy.spatial.KDTree
my_nodes = np.array(zip(my_x, my_y), dtype=np.double)

#Get length of nodes on the local rank
my_number_of_nodes = len(my_nodes)

#Print number of nodes
print("Total nodes: %d" % my_number_of_nodes)

#Create a kdtree to do nearest neighbor search
my_tree = scipy.spatial.cKDTree(my_nodes)

#Get all families
print("Performing nearest neighbor search... ")
my_ref_mag_state, my_families = my_tree.query(my_nodes, 
        k=MAX_NEIGHBORS_RETURNED, eps=0.0, p=2, distance_upper_bound=HORIZON) 
#Replace the default integers at the end of the arrays with -1's
my_families = np.where(my_families ==  my_number_of_nodes, -1, my_families)
#Delete node indices from their own families and create a masked array, first we
#mask all nodes outsize horizon up to MAX_SEARCH_SIZE. Using masked arrays 
#allows us to do fast Numpy operations we wouldn't otherwise be able to do. See
#<http://stackoverflow.com/questions/14104844/broadcasting-across-an-indexed-
#array-numpy/14124444> for details.
my_families_masked = ma.masked_equal(np.delete(my_families,0,1), -1)
#Find the maximum length of any family, we will use this to recreate the 
#families array such that it minimizes masked entries.
max_family_length = np.max(ma.count(my_families_masked, axis=1))
#Print a warning if MAX_NEIGHBORS_RETURNED needs to be increased
if max_family_length >= MAX_NEIGHBORS_RETURNED:
    print("Warning!: The family array has at least one row with length = \
           MAX_NEIGHBORS_RETURNED.  This constant likely should be increased")
else:
    print("Maximum size of any neighborhood: %d\n" % max_family_length)
#Recast the families array to be of minimum size possible
my_families = my_families_masked[:,:max_family_length]
#Harden the mask. Hardening the masks protects against mask removal if the 
#entries are reassigned by indexing into the array
my_families.harden_mask()

#Initialize dummy volumes
my_volumes = np.ones(my_number_of_nodes, dtype=np.double)

#Compute reference position state of all nodes
my_ref_pos_state_x = ma.masked_array(my_x[[my_families]] - my_x[:,None],
        mask=my_families.mask)
my_ref_pos_state_y = ma.masked_array(my_y[[my_families]] - my_y[:,None],
        mask=my_families.mask)

##Compute reference magnitude state of all nodes
my_ref_mag_state = (my_ref_pos_state_x * my_ref_pos_state_x +
        my_ref_pos_state_y * my_ref_pos_state_y) ** 0.5

#Initialize influence state
my_influence_state = np.ones_like(my_families,dtype=np.double) 

#insert crack
print("Inserting precracks...\n")
for crack in CRACKS:
    #Loop over and insert precracks.  The 1e-10 term is there to reduce the
    #chance of the crack directly intersecting any node and should not affect
    #the results at all for grid spacings on the order of 1.
    insert_crack(np.array(crack)+1e-10, my_tree, HORIZON, my_x, my_y, 
            my_families, my_influence_state)

#Compute weighted volume 
my_weighted_volume = (my_influence_state * my_ref_mag_state * 
        my_ref_mag_state * my_volumes[my_families])


#Temparary arrays
my_disp_x = np.zeros_like(my_x)
my_disp_y = np.zeros_like(my_y)
my_velocity_x = np.zeros_like(my_disp_x)
my_velocity_y = np.zeros_like(my_disp_y)
my_accel_x = np.zeros_like(my_disp_x)
my_accel_y = np.zeros_like(my_disp_y)
my_force_x = np.zeros_like(my_x)
my_force_y = np.zeros_like(my_y)
my_damage = np.zeros_like(my_x)

#Initialize output files
vector_variables = ['displacement']
scalar_variables = ['damage']
outfile = Ensight('output', vector_variables, scalar_variables)

print("Output variables requested:")
for item in vector_variables:
    print("    " + item)
for item in scalar_variables:
    print("    " + item)

if TIME_STEP == None:
    time_step = SAFTEY_FACTOR*compute_stable_time_step(my_families, 
            my_ref_mag_state, my_volumes, my_number_of_nodes, BULK_MODULUS,
            RHO, HORIZON)
else:
    time_step = TIME_STEP

#Begin the main explicit time stepping loop
print("\nRunning...")
if VERBOSE:
    loop_iterable = range(MAX_ITER)
else:
    progress = ProgressBar()
    loop_iterable = progress(range(MAX_ITER))

#Time stepping loop
for iteration in loop_iterable:
    
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
    my_force_x[:] = 0.0
    my_force_y[:] = 0.0
    
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

        #Compute the damage
        my_damage = 1.0 - ma.mean(my_influence_state,axis=1)

        outfile.write_geometry_file_time_step(my_x, my_y)
        outfile.write_vector_variable_time_step('displacement', 
                                               [my_disp_x,my_disp_y], time)
        outfile.write_scalar_variable_time_step('damage', 
                                               my_damage, time)
        outfile.append_time_step(time)
        outfile.write_case_file()


outfile.finalize()

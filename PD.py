#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Copyright 2018-2014 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import sys

import numpy as np
import numpy.ma as ma
import scipy.spatial
import matplotlib.path as path

from progressbar import ProgressBar
from progressbar import Percentage
from progressbar import Bar
from progressbar import RotatingMarker
from ensight import Ensight
import materials

from PyTrilinos import Epetra
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia

comm = Epetra.PyComm()
rank = comm.MyPID()
size = comm.NumProc()

### Peridynamic functions ###

# Internal force calculation
def compute_internal_force(force_x, force_y, pos_x, pos_y, disp_x, disp_y, 
        families, ref_mag_state, volumes, youngs_modulus, poisson_ratio, 
        influence_state, num_owned):
    """ Computes the peridynamic internal force due to deformations."""

    #Compute the deformed positions of the nodes
    def_x = pos_x + disp_x
    def_y = pos_y + disp_y

    #Compute deformation state
    def_state_x = ma.masked_array(def_x[families] - def_x[:num_owned,None],
            mask=families.mask)
    def_state_y = ma.masked_array(def_y[families] - def_y[:num_owned,None],
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
    
    #Compute dilatation
    dilatation = (3.0 * weighted_volume[:,None] * influence_state * 
            ref_mag_state * exten_state * volumes[families]).sum(axis=1)

    #Compute scalar force state
    #scalar_force_state = materials.elastic_material(youngs_modulus, 
            #poisson_ratio, dilatation, exten_state, ref_mag_state, 
            #weighted_volume, influence_state)   
    scalar_force_state = materials.bond_based_elastic_material(exten_state, 
            weighted_volume, youngs_modulus, poisson_ratio, influence_state)

    #Compute the force state
    force_state_x = scalar_force_state * def_unit_state_x
    force_state_y = scalar_force_state * def_unit_state_y

    #Integrate nodal forces 
    #Sum the force contribution from j nodes to i node
    force_x[:num_owned] += (force_state_x * volumes[families]).sum(axis=1)
    force_y[:num_owned] += (force_state_y * volumes[families]).sum(axis=1)

    #Subtract the force contribution from i nodes from j, the bincount()
    #operation is a trick to keep it fast in Numpy.  See:
    #<http://stackoverflow.com/questions/9790436/numpy-accumulating-one-array-
    #in-another-using-index-array> for details
    tmp_x = np.bincount(families.compressed(), (force_state_x * 
        volumes[:num_owned,None]).compressed()) 
    tmp_y = np.bincount(families.compressed(), (force_state_y * 
        volumes[:num_owned,None]).compressed()) 
    force_x[:len(tmp_x)] -= tmp_x
    force_y[:len(tmp_y)] -= tmp_y

    return 

#Compute stable time step function
def compute_stable_time_step(families, ref_mag_state, volumes, num_nodes, 
        bulk_modulus,rho,horizon):

    spring_constant = 18.0 * bulk_modulus / math.pi / horizon**4.0

    crit_time_step_denom = np.array([spring_constant * volumes[families[i]] / 
            ref_mag_state[i] for i in range(num_nodes)])**0.5

    critical_time_steps = np.sqrt(2.0 * rho) / crit_time_step_denom
    
    nodal_min_time_step = [ np.amin(item) for item in critical_time_steps ]

    return np.amin(nodal_min_time_step)

#Helper function that tests to see if two lines intersect
def test_line_seg_intersect(line1,line2):
    """Tests to see if two lines segments intersect.  The lines are defined as:

       line1 = [ p0_x, p0_y, p1_x, p1_y ]
       line2 = [ p2_x, p2_y, p3_x, p3_y ]
       
    """
    #See http://stackoverflow.com/questions/563198/how-do-you-detect-where-
    #two-line-segments-intersect for algorithm details
    
    #Read in individual point x,y positions from arguments
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

#Inserts a crack by removing neighbors from family lists
def insert_crack(crack, tree, horizon, x_pos, y_pos, families):
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
        #Crack line is vertical, discrete points along crack path
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
    _, nodes_near_crack = tree.query(points_along_crack, 
            k=MAX_NEIGHBORS_RETURNED, eps=0.0, p=2, 
            distance_upper_bound=2.0*horizon) 
    
    #The search above will produce duplicate neighbor nodes, make them into a
    #unique 1-dimensional list
    nodes_near_crack_flat = np.array(np.unique(nodes_near_crack),dtype=np.int)
    
    #Remove the dummy entries
    nodes_near_crack_flat = nodes_near_crack_flat[nodes_near_crack_flat != 
            tree.n]
                
    #Loop over nodes near the crack to see if any bonds in the nodes family
    #cross the crack path
    for node_index in nodes_near_crack_flat:
        #Loop over node family
        node_family = families[node_index][families[node_index] != -1]
        for bond_index,end_point_index in enumerate(node_family):
            #Define the bond line segment as the line between the node and its
            #endpoint.
            bond_line_seg = [ x_pos[node_index], y_pos[node_index], 
                x_pos[end_point_index], y_pos[end_point_index] ]
            #Test for intersection
            if test_line_seg_intersect(crack,bond_line_seg):
                #If we got to here that means we need to ``break'' the bond
                families[node_index][bond_index] = -1
    
    return


def boundary_condition_set(vertices,nodes,unbalanced_map):
    """Finds nodes enclosed by the polygon described with vertices"""
    
    #Create a polygon object with a list of vertices, vertices must be tuples
    polygon = path.Path(vertices,codes=None)
    #Returns an array with value True if point is inside polygon, False if not
    bool_arr = polygon.contains_points(nodes,radius=1.e-10)
    #List of the local node indices
    node_indices =  np.arange(unbalanced_map.NumMyElements(),dtype=np.int)
    #Returns local node indices that are inside the polygon
    return node_indices[bool_arr]

def influence_function(ref_mag_state,horizon):
    """Returns an influence state that has the form 1 - zeta/delta"""
    return 1. - ref_mag_state/horizon

#This line begins the main program.  This is not nescarry, but can be helpful
#if we want to load this file as a module from another Python script
if __name__ == "__main__":

    #####################
    ### Main Program ####
    #####################
    #INPUTS
    GRIDSIZE = 100
    HORIZON = 3.015
    TIME_STEP = 1.e-5
    #TIME_STEP = None
    YOUNGS_MODULUS = 200.0e9
    POISSON_RATIO = 0.29
    RHO = 7800
    SAFTEY_FACTOR = 0.5
    MAX_ITER = 4000
    PLOT_DUMP_FREQ = 100
    VERBOSE = False
    if sys.argv[-1] == '-v':
        VERBOSE = True
    CRACKS = [[GRIDSIZE/2., -1., GRIDSIZE/2., GRIDSIZE/10.],[GRIDSIZE/2., 
        9*GRIDSIZE/10. , GRIDSIZE/2., GRIDSIZE+1.],[5.,10,20.,40.]]
    MAX_NEIGHBORS_RETURNED = 300
    BC1_POLYGON = [(0.0,0.0),(HORIZON,0.0),(HORIZON,GRIDSIZE),
            (0.0,GRIDSIZE),(0.0,0.0)]
    BC2_POLYGON = [(GRIDSIZE-HORIZON,0.0),(GRIDSIZE,0.0),(GRIDSIZE,GRIDSIZE),
            (GRIDSIZE-HORIZON,GRIDSIZE),(GRIDSIZE-HORIZON,0.0)]
    BC1_VALUE = -5.
    BC2_VALUE = 5.
    VIZ_PATH='/Applications/paraview.app/Contents/MacOS/paraview'

    #Print version statement
    if rank == 0: print("PD.py version 0.4.0\n")

    #Set up the grid
    global_number_of_nodes = GRIDSIZE*GRIDSIZE

    #Populate the grid on the rank 0 processor only
    if rank == 0:
        #Create grid
        grid = np.mgrid[0:GRIDSIZE:1.,0:GRIDSIZE:1]

        #Create x,y tuple of node positions
        nodes = np.array(zip(grid[0].ravel(), grid[1].ravel()), 
                dtype=np.double)
        
        #Create a kdtree to do nearest neighbor search
        tree = scipy.spatial.cKDTree(nodes)

        #Get all families
        _, families = tree.query(nodes, k=100, eps=0.0, p=2, 
                distance_upper_bound=HORIZON) 
        #Replace the default integers at the end of the arrays with -1's
        families = np.delete(np.where(families ==  tree.n, -1, families),0,1)
        #Find the maximum length of any family, we will use this to recreate 
        #the families array such that it minimizes masked entries.
        max_family_length = np.max((families != -1).sum(axis=1))
        #Recast the families array to be of minimum size possible
        families = families[:,:max_family_length]

        #insert cracks
        if len(CRACKS) != 0: 
            print("Inserting precracks...\n")
        for crack in CRACKS:
            #Loop over and insert precracks.  The 1e-10 term is there to reduce the
            #chance of the crack directly intersecting any node and should not affect
            #the results at all for grid spacings on the order of 1.
            insert_crack(np.array(crack)+1e-10, tree, HORIZON, nodes[:,0], 
                    nodes[:,1], families)
        
    else:
        #Setup empty data on other ranks
        max_family_length = 0
        nodes = np.array([],dtype=np.double)
        families = np.array([],dtype=np.double)
    
    #Create node map with all the data on the rank 0 processor
    unbalanced_map = Epetra.Map(global_number_of_nodes, len(nodes), 0, comm)

    #Create and populate distributed Epetra vector to the hold the unbalanced
    #data.
    my_nodes = Epetra.MultiVector(unbalanced_map, 2)
    my_nodes[:] = nodes.T
    #Create and populate an Epetra mulitvector to store the families data
    max_family_length = comm.MaxAll(max_family_length)
    my_families = Epetra.MultiVector(unbalanced_map, max_family_length)
    my_families[:] = families.T

    #Load balance
    if rank == 0: print "Load balancing...\n"
    #Create Teuchos parameter list to pass parameter to ZOLTAN for load
    #balancing
    parameter_list = Teuchos.ParameterList()
    parameter_list.set("Partitioning Method","RCB")
    if not VERBOSE:
        parameter_sublist = parameter_list.sublist("ZOLTAN")
        parameter_sublist.set("DEBUG_LEVEL", "0")
    #Create a partitioner to load balance the grid
    partitioner = Isorropia.Epetra.Partitioner(my_nodes, parameter_list)
    #And a redistributer
    redistributer = Isorropia.Epetra.Redistributor(partitioner)
    #Redistribute nodes
    my_nodes_balanced = redistributer.redistribute(my_nodes)
    #The new load balanced map
    balanced_map = my_nodes_balanced.Map()
    #Create importer and exporters to move data between banlanced and 
    #unbalanced maps
    importer = Epetra.Import(balanced_map, unbalanced_map)
    exporter = Epetra.Export(balanced_map, unbalanced_map)
    #Create distributed vectors to store the balanced node positions
    my_x = Epetra.Vector(balanced_map)
    my_y = Epetra.Vector(balanced_map)
    my_families_balanced = Epetra.MultiVector(balanced_map, max_family_length)
    #Import the balanced node positions and family information
    my_x.Import(my_nodes[0],importer, Epetra.Insert)
    my_y.Import(my_nodes[1],importer, Epetra.Insert)
    my_families_balanced.Import(my_families,importer, Epetra.Insert)
    #Convert to integer data type for indexing purposes later
    my_families = np.array(my_families_balanced.T, dtype=np.int32)
    #Create a flattened list of all family global indices (locally owned 
    #+ ghosts)
    my_global_ids_required = np.unique(my_families[my_families != -1])
    #Create a list of locally owned global ids
    my_owned_ids = np.array(balanced_map.MyGlobalElements())
    #And its length
    my_num_owned = len(my_owned_ids)
    #The ghost indices required by the local processor is the relative complement 
    #of my_global_ids_required and my_owned_ids
    my_ghost_ids = np.setdiff1d(my_global_ids_required, my_owned_ids)
    #And its length
    my_num_ghosts = len(my_ghost_ids)
    #Get total length of worker array, this is len(owned) + len(ghosts)
    #summed over all processors
    length_of_global_worker_arr = comm.SumAll(len(my_owned_ids) 
            + len(my_ghost_ids))
    #Worker ids
    my_worker_ids = np.concatenate((my_owned_ids, my_ghost_ids))
    ##Create the map that will be used by worker vectors
    my_worker_map = Epetra.Map(length_of_global_worker_arr, 
            my_worker_ids, 0, comm)
    #Create the worker import/export operators to move data between the grid
    #data and the worker data
    worker_importer = Epetra.Import(my_worker_map, balanced_map)
    worker_exporter = Epetra.Export(my_worker_map, balanced_map)
    #Create worker vectors (owned + ghosts)
    my_x_worker = Epetra.Vector(my_worker_map)
    my_y_worker = Epetra.Vector(my_worker_map)
    #Import the needed components for local operations
    my_x_worker.Import(my_x, worker_importer, Epetra.Insert)
    my_y_worker.Import(my_y, worker_importer, Epetra.Insert)
    #Convert the global node ids in the family array to local ids
    my_families_local = np.array([my_worker_map.LID(i) 
        for i in my_families.flatten()])
    #Mask local family array
    my_families_local.shape = (len(my_families),-1)
    my_families_local = ma.masked_equal(my_families_local, -1)
    my_families_local.harden_mask()

    #Compute reference position state of all nodes
    my_ref_pos_state_x = ma.masked_array(my_x_worker[[my_families_local]] - 
            my_x_worker[:my_num_owned,None], mask=my_families_local.mask)
    my_ref_pos_state_y = ma.masked_array(my_y_worker[[my_families_local]] - 
            my_y_worker[:my_num_owned,None], mask=my_families_local.mask)

    #Compute reference magnitude state of all nodes
    my_ref_mag_state = (my_ref_pos_state_x * my_ref_pos_state_x +
            my_ref_pos_state_y * my_ref_pos_state_y) ** 0.5

    #Initialize influence state
    my_influence_state = influence_function(my_ref_mag_state,HORIZON)
    my_influence_state.harden_mask()
    #Create a reference copy, used for normalizing damage to a reference state
    my_ref_influence_state = my_influence_state.copy()

    #Initialize the dummy volumes
    my_volumes = np.ones_like(my_x_worker,dtype=np.double) 
    
    #Compute weighted volume 
    weighted_volume = (my_influence_state * my_ref_mag_state * 
            my_ref_mag_state * my_volumes[my_families_local]).sum(axis=1)

    #Create distributed vectors (owned only)
    my_disp_x = Epetra.Vector(balanced_map)
    my_disp_y = Epetra.Vector(balanced_map)
    my_force_x = Epetra.Vector(balanced_map)
    my_force_y = Epetra.Vector(balanced_map)

    #Create distributed worker vectors (owned + ghosts)
    my_disp_x_worker = Epetra.Vector(my_worker_map)
    my_disp_y_worker = Epetra.Vector(my_worker_map)
    my_force_x_worker = Epetra.Vector(my_worker_map)
    my_force_y_worker = Epetra.Vector(my_worker_map)

    #Temporary arrays
    my_velocity_x = np.zeros_like(my_disp_x)
    my_velocity_y = np.zeros_like(my_disp_y)
    my_accel_x = np.zeros_like(my_disp_x)
    my_accel_y = np.zeros_like(my_disp_y)
    my_damage = np.zeros_like(my_x)

    #Initialize output files
    vector_variables = ['displacement']
    scalar_variables = ['damage']
    #Instantiate output file object
    outfile = Ensight('output', vector_variables, scalar_variables, comm, 
            viz_path=VIZ_PATH)
    #Print the temporary output arrays
    if rank == 0: 
        print("Output variables requested:\n")
        for item in vector_variables:
            print("    " + item)
        for item in scalar_variables:
            print("    " + item)
        print(" ")

    #Find local nodes where boundary conditions should be applied
    bc1_local_node_set = boundary_condition_set(BC1_POLYGON,zip(my_x.ravel(),my_y.ravel()),balanced_map)
    bc2_local_node_set = boundary_condition_set(BC2_POLYGON,zip(my_x.ravel(),my_y.ravel()),balanced_map)

    #Calculate a stable time step or use the user defined
    if TIME_STEP == None:
        time_step = SAFTEY_FACTOR*compute_stable_time_step(my_families, 
                my_ref_mag_state, my_volumes, my_num_owned, YOUNGS_MODULUS,
                RHO, HORIZON)
    else:
        time_step = TIME_STEP

    #Begin the main explicit time stepping loop, the VERBOSE variable sets either
    #a progress bar or verbose output here.
    if rank == 0: 
        if VERBOSE:
            print("Running...")
        else:
            #Set up the progress bar
            widgets = ['Running: ', Percentage(), ' ', Bar(marker=RotatingMarker())]
            progress = ProgressBar(widgets=widgets, maxval=MAX_ITER).start()

    #Time stepping loop
    for iteration in range(MAX_ITER):
        
        #Set current time
        time = iteration * time_step

        #Print an information line
        if VERBOSE and rank ==0:
            print("iter = " + str(iteration) + " , time step = " + 
                    str(time_step) + " , sim time = " + str(time))

        #Enforce boundary conditions
        my_disp_x[[bc1_local_node_set]] = time * BC1_VALUE
        my_disp_y[[bc1_local_node_set]]= 0.0
        my_velocity_x[[bc1_local_node_set]] = BC1_VALUE
        my_velocity_y[[bc1_local_node_set]] = 0.0
        #
        my_disp_x[[bc2_local_node_set]] = time * BC2_VALUE
        my_disp_y[[bc2_local_node_set]]= 0.0
        my_velocity_x[[bc2_local_node_set]] = BC2_VALUE
        my_velocity_y[[bc2_local_node_set]] = 0.0
            
        #Clear the internal force vectors
        my_force_x[:] = 0.0
        my_force_y[:] = 0.0
        my_force_x_worker[:] = 0.0
        my_force_y_worker[:] = 0.0
        
        #Communicate the displacements (previous or boundary condition imposed)
        #to the worker vectors to be used at
        my_disp_x_worker.Import(my_disp_x, worker_importer, Epetra.Insert)
        my_disp_y_worker.Import(my_disp_y, worker_importer, Epetra.Insert)
        
        #Compute the internal force
        compute_internal_force(my_force_x_worker, my_force_y_worker, 
                my_x_worker, my_y_worker, my_disp_x_worker, my_disp_y_worker, 
                my_families_local, my_ref_mag_state, my_volumes, YOUNGS_MODULUS, 
                POISSON_RATIO, my_influence_state, my_num_owned)
        
        #Communicate values from worker vectors (owned + ghosts) back to owned only
        my_force_x.Export(my_force_x_worker, worker_exporter, Epetra.Add)
        my_force_y.Export(my_force_y_worker, worker_exporter, Epetra.Add)
     
        #Compute the nodal acceleration
        my_accel_x_old = my_accel_x.copy()
        my_accel_y_old = my_accel_y.copy()
        my_accel_x = my_force_x  / RHO
        my_accel_y = my_force_y  / RHO
        
        #Compute the nodal velocity
        my_velocity_x += 0.5 * (my_accel_x_old + my_accel_x) * time_step
        my_velocity_y += 0.5 * (my_accel_y_old + my_accel_y) * time_step
        
        #Compute the new displacements
        my_disp_x += my_velocity_x * time_step + (0.5 * my_accel_x * 
                time_step * time_step)
        my_disp_y += my_velocity_y * time_step + (0.5 * my_accel_y * 
                time_step * time_step)
        
        #Compute stable time step
        #time_step = compute_stable_time_step(my_x, my_y, my_disp_x, my_disp_y, 
                #my_families, my_ref_mag_state, my_weighted_volume, my_volumes, 
                #my_number_of_nodes, BULK_MODULUS,RHO)

        #Dump plots
        if iteration % PLOT_DUMP_FREQ  == 0 or iteration == (MAX_ITER-1):
            if VERBOSE and rank == 0:
                print "Writing plot file..."

            #Compute the damage
            my_damage = 1.0 - ma.mean(my_influence_state / 
                    my_ref_influence_state,axis=1)
            
            outfile.write_geometry_file_time_step(my_x, my_y)
            outfile.write_vector_variable_time_step('displacement', 
                                                   [my_disp_x,my_disp_y], time)
            outfile.write_scalar_variable_time_step('damage', 
                                                   my_damage, time)
            outfile.append_time_step(time)
            outfile.write_case_file(comm)

        #Update the progress bar
        if not VERBOSE and rank == 0:
            progress.update(iteration + 1)

    #Finalize plotfiles
    outfile.finalize()
    #Wrap up the progress bar printing
    if not VERBOSE and rank == 0:
        progress.finish()


# Simple constitutive model
def bond_based_elastic_material(exten_state, weighted_volume, youngs_modulus, 
        poisson_ratio, influence_state):
    """Computes the scalar force state.  This is the state-based version of a
       bond based material."""
    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))

    #Return the force
    return (9.0 * bulk_modulus * influence_state / weighted_volume[:,None] * 
            exten_state)

# Linear peridynamic solid model
def elastic_material(youngs_modulus, poisson_ratio, dilatation, exten_state,
        ref_mag_state, weighted_volume, influence_state):
    
    #Convert the elastic constants
    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    
    #Compute the pressure
    pressure = -bulk_modulus * dilatation
    
    #Compute the deviatoric extension state
    dev_exten_state = (exten_state - dilatation[:,None] * ref_mag_state / 3.0)
    
    #Compute the peridynamic shear constant
    alpha = 15.0 * shear_modulus / weighted_volume
 
    #Compute the isotropic and deviatoric components of the force scalar state
    iso_force_state = (-3.0 * pressure[:,None] / weighted_volume[:,None] * 
            influence_state * ref_mag_state)
    dev_force_state = alpha[:,None] * influence_state * dev_exten_state 
    
    #Return the force scalar-state
    return iso_force_state + dev_force_state


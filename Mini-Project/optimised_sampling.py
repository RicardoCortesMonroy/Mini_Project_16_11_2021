from scipy.optimize import minimize
import math
import numpy as np
import functions as f
import time

time_limit = 3 # Maximum algorithm time in seconds




def optimised_distance_sampling(centre, radius, n_s):
    samples = np.empty((0,2))
    
    t0 = time.time()
    
    while len(samples) < n_s and time.time() - t0 < time_limit:
        
        try: 
            s = generate_new(centre, radius, n_s, samples)
            samples = np.vstack((samples, s))
        except ValueError:
            continue
              
    return samples
  
## =============================================================================
## Sample generation with radial exclusion zones
## =============================================================================

  
# def generate_new(centre, radius, n_s, previous_samples):
    
#     sep_distance = 1.5*radius/math.sqrt(n_s)
        
#     def obj(x): return -objective(x, centre, previous_samples, sep_distance, radius)
#     def cons01(x): return trust_region(x, centre, radius)
#     s_0 = f.random_radial_sample(centre, radius)
#     cons = {'type': 'ineq', 'fun': cons01}
    
#     sol = minimize(obj, s_0, constraints = cons)
#     # if not sol.success: raise ValueError(sol.message) 
    
#     return sol.x  
  

## =============================================================================
## Sample generation with bounded exclusion zones
## =============================================================================

## Constraint that ensures that x is outside the bounds of every previous sample
# def exclusion_constraint(x,samples, sep_bounds):
#     results = [f.outside_bounds(x , f.bounds_centred_on(s_prev,bounds = sep_bounds)) 
#                for s_prev in samples]
#     return 1e-9 if len(results) == 0 else min(results)

# def generate_new(bounds, n_s, previous_samples, N):
    
#     sep_bounds = 1.5*bounds/n_s**(1/N)
    
#     # No objective, just makes sure that the sample is feasible
#     def obj(x): return 0
#     # centre = f.centre_of_bounds(bounds)
#     # def obj(x): return np.linalg.norm(centre - x)
#     def cons01(x): return f.inside_bounds(x, bounds)
#     def cons02(x): return exclusion_constraint(x, previous_samples, sep_bounds)
#     s_0 = f.random_bounded_sample(bounds)
#     cons = [{'type': 'ineq', 'fun': cons01},
#             {'type': 'ineq', 'fun': cons02}]
            
#     sol = minimize(obj, s_0, constraints = cons)
        
#     return sol.x

# =============================================================================
# Sample generation that maximises closest distance from other samples and bounds
# =============================================================================

a = 1.0 # weighting for closest sample distance
b = 0.1 # weighting for distance to centre
def objective(x, samples, centre, trust_bounds):
   
    nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples], default = 0)
    nearest_distance_to_bound = f.inside_bounds(x, trust_bounds)
    nearest_distance = min(nearest_distance_to_sample, nearest_distance_to_bound)
    distance_to_centre = np.linalg.norm(x-centre)
    
    obj = b * distance_to_centre - a * nearest_distance
    
    return obj

def generate_new(trust_bounds, previous_samples):
    
    centre = f.centre_of_bounds(trust_bounds)
    
    def obj(x): return objective(x, previous_samples, centre, trust_bounds)
    def cons01(x): return f.inside_bounds(x, trust_bounds)
    s_0 = f.random_bounded_sample(trust_bounds)
    cons = [{'type': 'ineq', 'fun': cons01}]
            
    sol = minimize(obj, s_0, constraints = cons)
        
    return sol.x


        

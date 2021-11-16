from scipy.optimize import minimize
import math
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time

trust_bounds = np.array([[-10,10],
                         [-20,10]])
n_s = 6
N = 2

time_limit = 10 # Maximum algorithm time in seconds
plot_margin = 1.2

annotation_offset = np.array([0.2,0.2])
grey = "#aaaaaa"
show_annotations = 0
do_plot = 1
iterations = 1 if do_plot else 500

a = 1.0 # weighting for closest sample distance
b = 0.1 # weighting for distance to centre
def objective(x, samples, centre, trust_bounds):
   
    nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples], default = 0)
    nearest_distance_to_bound = f.inside_bounds(x, trust_bounds)
    nearest_distance = min(nearest_distance_to_sample, nearest_distance_to_bound)
    distance_to_centre = np.linalg.norm(x-centre)
    
    obj = b * distance_to_centre - a * nearest_distance
    
    return obj

total_time = 0
for i in range(iterations):

    t0 = time.time()
    
    if do_plot:
        fig = plt.figure()
        fig.set_size_inches(8, 8, forward = True)
    
    samples = np.empty((0,2))
    
    # cons01 ensures the new sample is within the trust bounds
    # cons02 ensures the new sample is outside of all of the separation bounds of other samples
    # The object is to be as close to the centre as possible
    while len(samples) < n_s and time.time() - t0 < time_limit:
        
        centre = f.centre_of_bounds(trust_bounds)

        # No objective, just makes sure that the sample is feasible
        def obj(x): return objective(x, samples, centre, trust_bounds)
        def cons01(x): return f.inside_bounds(x, trust_bounds)
        s_0 = f.random_bounded_sample(trust_bounds)
        cons = [{'type': 'ineq', 'fun': cons01}]
                
        sol = minimize(obj, s_0, constraints = cons)
        s = sol.x
        if not sol.success: continue
        
        samples = np.vstack((samples, [s]))
        
        if do_plot:
            if show_annotations:
                j = len(samples)
                plt.plot(*s_0,'o', color = grey)
                plt.plot([s_0[0],s[0]],[s_0[1],s[1]], '-', color = grey)
                plt.annotate(j, s_0 + annotation_offset)
                plt.annotate(j, s   + annotation_offset)
        
            plt.plot(*s,'ro')
    
    t1 = time.time()
    total_time += t1-t0
    
if do_plot:

    plt.xlim(trust_bounds[0])
    plt.ylim(trust_bounds[1])

    plt.savefig("plot.pdf")   

print("Average time: {:.3f}s".format(total_time/iterations))    
        

        

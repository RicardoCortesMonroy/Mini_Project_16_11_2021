from scipy.optimize import minimize
import math
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time

bounds = np.array([[-10,10],
                   [-20,10]])
n_s = 16
N = 2

# Bounded excluded region around each sample
sep_bounds = 1.5*bounds/n_s**(1/N)
time_limit = 10 # Maximum algorithm time in seconds
plot_margin = 1.2

annotation_offset = max([max(b)-min(b) for b in bounds]) * 0.005 * np.ones(2)
grey = "#aaaaaa"
show_annotations = True
do_plot = True
iterations = 1 if do_plot else 500

# Constraint that ensures that x is outside the bounds of every previous sample
def exclusion_constraint(x,samples):
    results = [f.outside_bounds(x , f.bounds_centred_on(s_prev, bounds=sep_bounds)) 
               for s_prev in samples]
    return 1e-9 if len(results) == 0 else min(results)

total_time = 0
for i in range(iterations):

    t0 = time.time()
    
    if do_plot:
        fig = plt.figure()
        fig.set_size_inches(8, 8, forward = True)
        
        plt.plot(*f.rect(bounds),'k--')
    
    samples = np.empty((0,2))
    
    # cons01 ensures the new sample is within the trust bounds
    # cons02 ensures the new sample is outside of all of the separation bounds of other samples
    # The object is to be as close to the centre as possible
    while len(samples) < n_s and time.time() - t0 < time_limit:
        
        centre = f.centre_of_bounds(bounds)

        # No objective, just makes sure that the sample is feasible
        def obj(x): return 0
        def cons01(x): return f.inside_bounds(x, bounds)
        def cons02(x): return exclusion_constraint(x, samples)
        s_0 = f.random_bounded_sample(bounds)
        cons = [{'type': 'ineq', 'fun': cons01},
                {'type': 'ineq', 'fun': cons02}]
                
        sol = minimize(obj, s_0, constraints = cons)
        s = sol.x
        if not sol.success: continue
        
        samples = np.vstack((samples, [s]))
        
        if do_plot:
            if show_annotations:
                j = len(samples)
                s_bounds = f.bounds_centred_on(s, bounds=sep_bounds)
                plt.plot(*s_0,'o', color = grey)
                plt.plot([s_0[0],s[0]],[s_0[1],s[1]], '-', color = grey)
                plt.annotate(j, s_0 + annotation_offset)
                plt.annotate(j, s   + annotation_offset)
                plt.plot(*f.rect(s_bounds), '-', color = grey, linewidth = 0.8)
        
            plt.plot(*s,'ro')
    
    t1 = time.time()
    total_time += t1-t0
    
if do_plot:

    plt.xlim(bounds[0])
    plt.ylim(bounds[1])

    plt.savefig("plot.pdf")   

print("Average time: {:.3f}s".format(total_time/iterations))    
        

        

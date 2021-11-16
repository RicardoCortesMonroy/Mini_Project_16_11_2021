from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
import math
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time

radius = 10
centre = np.array([0,0])
n_s = 6

sep_distance = 1.5*radius/math.sqrt(n_s)
time_limit = 10 # Maximum algorithm time in seconds
plot_margin = 1.2

annotation_offset = radius * 0.025 * np.ones(2)
grey = "#aaaaaa"
show_annotations = False
do_plot = False
iterations = 1 if do_plot else 5

# Returns >0 if x is within trust region
def trust_region(x):
    distance = np.linalg.norm(x-centre)
    return radius - distance


a = 1.0 # Weighting coefficient for mpltimising inter-sample distance
b = 0.00 # Weighting coefficient for minimising distance from centre
def objective(x, centre, samples):
    output = 0
    for i in range(samples.shape[0]):
        distance_s = min(sep_distance,np.linalg.norm(x-samples[i]))
        output += a*distance_s
    distance_c = np.linalg.norm(x-centre)
    output /= max(0.01, b*distance_c/radius)
    return output

total_time = 0
for i in range(iterations):
    
    t0 = time.time()
    
    if do_plot:
        fig = plt.figure()
        fig.set_size_inches(8, 8, forward = True)
        
        plt.plot(*f.circle(centre,radius,100),'k--')
        
    samples = np.empty((0,2))
    
    while samples.shape[0] < n_s and time.time() - t0 < time_limit:
        j = samples.shape[0]
        def obj(x): return -objective(x, centre, samples)
        cons = {'type': 'ineq', 'fun': trust_region}
        nlc = NonlinearConstraint(trust_region, 0, radius**2)
        
        sol = differential_evolution(obj,
              [(-radius, radius) for i in range(2)],
              constraints = (nlc))
        s = sol.x
        if not sol.success: continue
        
        samples = np.vstack((samples, [s]))
    
        if do_plot:
            if show_annotations:
                plt.annotate(j, s   + annotation_offset)
                plt.plot(*f.circle(s,sep_distance,40), '-', color = grey, linewidth = 0.8)
            
            plt.plot(*s,'ro')
    
    t1 = time.time()
    total_time += t1-t0

if do_plot:

    plt.xlim(-plot_margin*radius + centre[0], plot_margin*radius + centre[0])
    plt.ylim(-plot_margin*radius + centre[1], plot_margin*radius + centre[1])
    
    plt.savefig("plot.pdf")

print("Average time: {:.3f}s".format(total_time/iterations))        
        

        

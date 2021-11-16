from scipy.optimize import minimize, NonlinearConstraint, differential_evolution
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
import functions as f
import optimised_sampling as ods

directory = "C:\\Users\\rmonr\\Desktop\\Research_Project_2021\\Mini-Project\\"

"""
TODO:
    [X] Keep track of all new samples generated
    [X] Only allow samples within trust region
    [X] Implement a maximum proportion of samples within the trust region, the rest of
        which will be newly generated
    [X] Generate samples according to bounds (not radially) so they can be scaled up to higher dimensions
    [X] Alternate: generate samples by maximising closest distance to previous samples
    [X] Improve distribution of bounded sampling (work on optimised_sampling.py)
    [X] Count total number of black box evaluations
    [X] Eliminate evaluation redundancy when evaluating repeated samples for sample_eval_k
    [X] Implement noise to step within a certain 'jitter radius' to improve exploration
    
Comments:
    (1) Perversely, regression optimisation seems to be much more stable at 
    higher n_s (>6) whereas it often fails at lower n_s, failing due to 
    reaching the maximum iteration limit. I'm not sure why. Surely a 
    least-squares objective function is easier to minimise for fewer samples, 
    but maybe it's a  quirk with the solver I'm using.
    
    (2) I've changed the sampling to work only with rectangles of bounds. This is because radially
    sampling cannot feasibly be scaled up to N-dimensions as the definition of polar coordinates
    becomes prohibitively complicated for even a minor increase in input dimensions.
    
    (3) The exclusion zones of the samples in sample generation are no longer radial.
    Instead, the distance to the closest neighbouring sample is maximised.
    This is because of the difficulty in defining an exclusion radius for N dimensions 
    (Fun fact: the fraction of volume that an n-dimensional sphere takes up in the smallest 
    n-dimensional cube that encloses it vanishes off to zero as n -> inf. 
    How fast it vanishes is important to determine how many spheres can fit in an 
    n-dimensional volume and therefore what exclusion radius should be chosen. 
    But I feel determining this is outside of the scope of this project (a rabbit
    hole led me to discover that it requires using the Euler-Gamma function)).
    
    (4) Overall I've been making sure that this algorithm can work for higher dimensions as
    optimisation problems typically have many more input variables than just 2.
    
    (5) The jitter seems to be pretty effective at nudging the point out of difficult areas.
    For example, when going along the centre of the valley, the point usually struggles
    to move because all but one very specific direction will lead to a decrease
    in the objective. There is a very small chance that the direction will be
    found randomly, so the steps keep getting rejected, and the radius keeps
    getting smaller. By forcing the step to move just a little bit, you're 
    effectively expanding the cone of viable directions for the next step,
    without decreasing the objective too much.
"""

N = 2                   # Number of input dimensions
radius = 5              # Radius of Trust region
radius_tol = 0.05       # Distance (scaled by radius) from the boundary x_k1 must be less than to count as "on the boundary"
min_radius = 1e-4       # Minimum allowable radius
n_s = 12                # Number of samples within trust region per iteration
n_i = 10                # Number of iterations
frac_new_s = 0.3        # Minimum fraction of samples in current iteration that are newly generated
beta_red = 0.2          # Reduction multiplier for trust-region radius
beta_inc = 2.0          # Increase multiplier for trust-region radius

# Exploration parameters
do_exploration = 1
exploration_jitter = 0.20    # How much the step will 'jitter' for exploration (fraction of the diameter)


# Starting bounds
trust_bounds = f.bounds_centred_on(np.zeros(2), diameter = 20)

# Initial values
x_k = x_opt = np.array([0,-10])#f.random_bounded_sample(trust_bounds)

# Plotting parameters
scale = 5
col = 3
fig = plt.figure(figsize=(scale*col, scale*math.ceil(n_i/col),))
contour_res = 250  # Number of points per axis plotted in the contour map
margin = 1.2       # How much bigger the contour plot is than the diameter of the trust region
color = {'grey' : '#888888',
         'red'  : '#cc2222'} 

# Pre-initialise
samples  = np.empty((0,2))
samples_eval = np.array([])
samples_k = np.empty((0,2))
old_samples_k = np.empty((0,2))
a = np.zeros(6)

evaluation_count = 0

def black_box(x):
    global evaluation_count
    evaluation_count += 1
    return f.black_box((x))

sample_scale = 0.5
x_all = np.array([[*x_k]])
current_evaluation = black_box(x_k)
best_evaluation = math.inf

# =============================================================================

for i in range(n_i):
    print("\nIteration {}".format(i+1))
    
    selection_status = ""

    x_k_old = x_k
    x_k = x_opt
    old_radius = radius
    trust_bounds = f.bounds_centred_on(x_k, diameter = 2*radius)
    previous_evaluation = current_evaluation
    
    # =============================================================================
    
    # Select up to n_s_old samples from previous iterations as long as they fit within
    # the trust region. Add them to samples_k, and generate new samples until there are
    # n_s samples in samples_k
    
    n_s_old = math.floor(n_s * (1-frac_new_s))
    old_samples_k = samples_k
    samples_k = np.empty((0,2))
    samples_eval_k = np.empty(0)
    updated_samples = np.empty((0,2))
    updated_samples_eval = np.empty(0)
    
    for s_ in range(len(samples)):
        if len(samples_k) >= n_s_old: break
        
        s = len(samples) - s_ - 1
        if f.inside_bounds(samples[s], trust_bounds) >= 0: 
            samples_k = np.vstack((samples_k, samples[s]))  
            samples_eval_k = np.append(samples_eval_k, samples_eval[s])
        else:
            
            updated_samples = np.vstack((updated_samples, samples[s]))
            updated_samples_eval = np.append(updated_samples_eval, samples_eval[s])
    
          
    # Generate new samples
    while len(samples_k) < n_s:
        # new_sample = ods.generate_new(x_k, radius, n_s, samples_k).reshape(1,2)
        new_sample      = ods.generate_new(trust_bounds, samples_k).reshape(1,2)
        samples_k       = np.vstack((samples_k, new_sample))
        samples_eval_k  = np.append(samples_eval_k, black_box(new_sample[0]))
    
    # Finally update total sample and sample_eval lists with data from current iteration 
    samples = np.vstack((updated_samples, samples_k))
    samples_eval = np.append(updated_samples_eval, samples_eval_k)
    
    """
    Why remove duplicate samples only to add them back? Why not just add the 
    new samples and leave the duplicates in the samples list?
    Well because the duplicates are used in the current iteration, ideally,
    we want them to be at the end of the list of samples for the next iteration.
    Notice how we search for potential repeated samples end-first, so the closer 
    the samples used in the previous iteration are to the end of the list, the 
    more priority is given to them.
    
    """
    
    # =============================================================================
    
    # Regression and optimisation of quadratic approximation
    
    # The solver keeps warning me about using a zero Hessian so here we go
    warnings.filterwarnings("ignore")
    
    def regression_func(a): return f.least_squares(a, f.quad_approx, samples_k, samples_eval_k) 
    nlc= NonlinearConstraint(f.convexity_constraint, 0, math.inf)
    sol = differential_evolution(regression_func, 
                                 [(-1e6,1e6) for i in range(len(a))],
                                 constraints=nlc)
    a = sol.x
    
    # Turn warnings back on
    warnings.filterwarnings("once")
    
    if sol.success:
        print("Regression successful")
    else:
        print("Regression failed. {}".format(sol.message))
        
    
    def quad_approx_k(x): return f.quad_approx(a, x)
    def trust_region(x): return f.distance_constraint(x, x_k, radius)
    cons = {'type': 'ineq', 'fun': trust_region}
    x_k1_quad = minimize(quad_approx_k, x_k, constraints=cons).x
    quad_evaluation = black_box(x_k1_quad)
    
    P = np.array([[a[3],a[4]],
                  [a[4],a[5]]])
    print("Eigenvalues: {}".format(np.linalg.eigvals(P)))
    
    # =============================================================================

    # Compares the performance of the linear and quadratic approximation, as well as the evaluations of the samples
    evaluations = [quad_evaluation, *samples_eval_k]
    best_evaluation = min(evaluations)
    if best_evaluation == quad_evaluation:
        x_k1 = x_k1_quad
        selection_status = "quadratic"
    else:
        x_k1 = samples_k[evaluations.index(min(evaluations))-1]
        selection_status = "sample selected"
    distance = np.linalg.norm((x_k1-x_k))
    
    # x_ds = x_k + direction_scaling * (x_k1 - x_opt)
    # ds_evaluation = black_box(x_ds)
    # print("Current evaluation: {}".format(current_evaluation))
    # print("Best evaluation: {}".format(best_evaluation))
    
    # =============================================================================
    
    # Chooses whether to accept x_k1 as the next point
    # and adjusts radius accordingly
    if best_evaluation < current_evaluation:
        
        distance = np.linalg.norm(x_k-x_k1)
        if distance > radius * (1-radius_tol):
            radius = beta_inc * radius
    
        # # If direction scaling provides any benefit, apply it:
        # if ds_evaluation < best_evaluation:
        #     x_opt = x_ds
        #     selection_status += ", direction scaled"
        #     # If x_ds is outside the region, expand the region to capture it:
        #     radius = max(radius, np.linalg.norm(x_ds - x_k))
        #     current_evaluation = ds_evaluation
        # else:
        x_opt = x_k1
        current_evaluation = best_evaluation
            
    else:
        radius = max(min_radius, beta_red * radius)
        x_opt = x_k
        best_evaluation = current_evaluation
        selection_status = "step rejected"
    
    
    # =============================================================================
    
    # Apply exploration jitter
    
    if(do_exploration):
        jitter_bounds = f.bounds_centred_on(x_opt, diameter = 2*exploration_jitter*old_radius)
        old_opt = x_opt
        x_opt = f.random_bounded_sample(jitter_bounds)
        best_evaluation = black_box(x_opt)
        current_evaluation = best_evaluation

# =============================================================================
#     Plotting
# =============================================================================

    x1_bounds = margin * np.array([-old_radius, old_radius]) + x_k[0]
    x2_bounds = margin * np.array([-old_radius, old_radius]) + x_k[1]
    
    x1 = np.linspace(*x1_bounds, contour_res)
    x2 = np.linspace(*x2_bounds, contour_res)
    X1, X2 = np.meshgrid(x1, x2)

    def black_box_log(x):
        output = f.black_box(x)
        if output > 0:
            return math.log10(output)
        else:
            return -1e06

    sub = fig.add_subplot(math.ceil(n_i/col), col, i+1)
    
    black_box_Z = f.mat_func(X1, X2, black_box_log)
    fun_approx_Z = f.mat_func(X1, X2, quad_approx_k)
    black_box_contour = sub.contour(X1, X2, black_box_Z, 10, colors=color['red'], linewidths=[0.5])
    approx_contour = sub.contour(X1, X2, fun_approx_Z, 10, cmap='summer', linewidths=[0.5])
    
    sub.clabel(black_box_contour, black_box_contour.levels, inline=True, fontsize=5)
    
    # Plot trust region
    plt.plot(*f.circle(x_k, old_radius, resolution = 40), 'k--', linewidth=1)
    # Plot trust bounds
    plt.plot(*f.rect(trust_bounds), '--', color=color['grey'], linewidth=1)
    

    # Plot sample points
    for s in range(len(samples_k)):
        sub.plot(*samples_k[s], 'bo', markersize=3.0)    
    for s in range(len(old_samples_k)):
        sub.plot(*old_samples_k[s], marker='o', color=color['grey'], markersize=6.0, markerfacecolor='none')

    # Plot all previous points in grey
    for j in range(len(x_all)):
        sub.plot(*x_all[j], marker='x', color=color['grey'])
    sub.plot(*f.path(*x_all,x_k), '--', color=color['grey'], linewidth=1)
    

    # If x_k is repeated, do not add it to the list of x_all
    x_k_repeated = False
    for j in range(len(x_all)):
        if (x_k == x_all[j]).all():
            x_k_repeated = True
    if not x_k_repeated:
        x_all = np.append(x_all, np.array([[*x_k]]), axis=0)

    # Plot all x's
    stepRejected = selection_status == "step rejected"
    sub.plot(*x_k, 'kx')
    sub.plot(*x_k1_quad, 'gs', markerfacecolor='none')

    if stepRejected:
        sub.plot(*x_k1, 'rx')
    else:
        sub.plot(*f.path(x_k, x_k1), linestyle='-', color='k', linewidth=1)
        sub.plot(*x_k1, 'gx')
    
        
    # Show jitter bounds
    if do_exploration:
        sub.plot(*f.rect(jitter_bounds), '--', color=color['grey'], linewidth = 0.2)
        if stepRejected:
            sub.plot(*f.path(x_k, x_opt), linestyle='-', color='k', linewidth=1)
        else:
            sub.plot(*f.path(x_k1,x_opt), linestyle='-', color='k', linewidth=1)
        sub.plot(*x_opt, 'gx')
    
    # Enforce axes' bounds
    sub.set_xlim(x1_bounds.tolist())
    sub.set_ylim(x2_bounds.tolist())

    sub.title.set_text('Iteration: {}'.format(i) +
                       "\n" + r"f($\bf{x}_{k}$)=" + "{:.4f} or 10^{:.2f}".format(previous_evaluation, math.log10(previous_evaluation)) +
                       "\n" + r"f($\bf{x}_{k+1}$)=" + "{:.4f} or 10^{:.2f}".format(current_evaluation, math.log10(current_evaluation)) +
                       "\n" + r"f($\bf{x}_{quad}$)=" + "{:.4f} or 10^{:.2f}".format(quad_evaluation, math.log10(quad_evaluation)) +
                       "\n" + "r: {:.6f}".format(old_radius) +
                       "\n" + "{}".format(selection_status)) 

fig.tight_layout()
plt.savefig(directory + "plot.pdf")
plt.close(fig)
file = open(directory + "done.txt") 
print(file.read()) ; file.close()

print("\nCompleted with {} evaluations".format(evaluation_count))
print("\nAveraged {} evaluations per iteration".format(evaluation_count/n_i))

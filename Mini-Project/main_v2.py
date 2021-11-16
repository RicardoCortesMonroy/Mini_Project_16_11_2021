from scipy.optimize import minimize
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import functions as f
import optimised_sampling as ods

"""
Changes:
    Optimised distance sampling:
        By creating a optimisation sub-problem wherein samples maximise their
        their distance from eachother, sampling now has better coverage of the
        trust region. More details in optimised_sampling.py
    Removed weighted average sample
    Removed LHS
"""


radius = 10             # Radius of Trust region
radius_tol = 0.1        # Distance from the boundary x_k1 must be less than to count as "on the boundary"
n_s = 6                 # Number of samples per iteration
n_i = 50                # Number of iterations
eta_1 = 0.0             # Minimum decrease ratio for accepting new point
eta_2 = 0.1             # Minimum decrease ratio for increasing trust-region radius
beta_red = 0.9          # Reduction multiplier for trust-region radius
beta_inc = 1.1          # Increase multiplier for trust-region radius
dif_step = 1e-3         # Step size for finding approximate derivative
wa_strictness = 1e-05   # Strictness of the weighted average for samples
ds_k = 0.5              # The exponent of this is the ratio by which the dot product of each direction decreases in direction scaling
ds_r0 = 3               # Initial ratio in direction scaling

# Starting bounds
blc = np.array([-10,-10]) # Bottom left corner
width = 20
height = 20

# Initial values
x_k = x_opt = np.array([random.random() * width  + blc[0],
                        random.random() * height + blc[1]])
sample_scale = 0.5
x_all = np.array([[*x_k]])
best_evaluation = math.inf

scale = 5
col = 3
fig = plt.figure(figsize=(scale*col, scale*math.ceil(n_i/col),))
contour_res = 250  # Number of points per axis plotted in the contour map
margin = 1.2       # How much bigger the contour plot is than the diameter of the trust region



for i in range(n_i):

    selection_status = ""

    x_k_old = x_k
    x_k = x_opt
    current_evaluation = best_evaluation
    old_radius = radius

    s = 0
    samples = np.array([[]])
    
    # =============================================================================
    
    # Optimised distance sampling
    
    samples = ods.optimised_distance_sampling(x_k, radius, n_s)
    samples_eval = np.zeros(n_s)
    for j in range(n_s): samples_eval[j] = f.black_box(samples[j])
    
    # =============================================================================
       
    # Scaling based on repeated direction
    direction_scaling = 1
    for j in range(min(len(x_all)-2,10)):
        index = len(x_all) - j - 1
        d1 = x_all[ index ] - x_all[index-1]
        d1 = d1 / np.linalg.norm(d1)
        d2 = x_all[index-1] - x_all[index-2]
        d2 = d2 / np.linalg.norm(d2)
        
        dot = np.dot(d1,d2)
        direction_scaling += max(0,ds_r0 * math.exp(-ds_k * j) * dot)
   
    # =============================================================================

    # Generates 2 different surrogate functions: quadratic approximation and linear
    # Evaluates the black box value of each, along with those of all the sampels
    # Then chooses the evaluation that gives the greatest decrease in f(x)

    a = np.zeros(6)
    cons = [{'type': 'ineq', 'fun': f.convexity_constraint}]
    def regression_func(a): return f.least_squares(
        a, f.quad_approx, samples, samples_eval)
    a = minimize(regression_func, a, constraints=cons).x

    def quad_approx_k(x): return f.quad_approx(a, x)

    b = np.zeros(3)
    def regression_func(b): return f.least_squares(
        b, f.lin_approx, samples)
    b = minimize(regression_func, a).x

    def lin_approx_k(x): return f.lin_approx(b, x)

    def trust_region(x): return f.distance_constraint(x, x_k, radius)
    cons = {'type': 'ineq', 'fun': trust_region}
    x_k1_quad = minimize(quad_approx_k, x_k, constraints=cons).x
    x_k1_lin = minimize(lin_approx_k, x_k, constraints=cons).x
    quad_evaluation = f.black_box(x_k1_quad)
    lin_evaluation = f.black_box(x_k1_lin)
    
    # =============================================================================

    # Compares the performance of the linear and quadratic approximation, as well as the evaluations of the samples
    evaluations = [quad_evaluation, lin_evaluation, *samples_eval]
    best_evaluation = min(evaluations)
    if best_evaluation == quad_evaluation:
        x_k1 = x_k1_quad
        approx_k = quad_approx_k
        selection_status = "quadratic"
    elif best_evaluation == lin_evaluation:
        x_k1 = x_k1_lin
        approx_k = lin_approx_k
        selection_status = "linear"
    else:
        x_k1 = samples[evaluations.index(min(evaluations))-2]
        approx_k = None
        selection_status = "sample selected"
    distance = np.linalg.norm((x_k1-x_k))
    
    x_ds = x_k + direction_scaling * (x_k1 - x_opt)
    ds_evaluation = f.black_box(x_ds)
    
    # =============================================================================
    
    # Chooses whether to accept x_k1 as the next point
    # and adjusts radius accordingly
    if best_evaluation < current_evaluation:
        
        if distance > radius - radius_tol:
            radius = beta_inc * radius
    
        # If direction scaling provides any benefit, apply it:
        if ds_evaluation < best_evaluation:
            x_opt = x_ds
            selection_status += ", direction scaled"
            # If x_ds is outside the region, expand the region to capture it:
            radius = max(radius, np.linalg.norm(x_ds - x_k))
        else:
            x_opt = x_k1
            
    else:
        radius = beta_red * radius
        x_opt = x_k
        best_evaluation = current_evaluation
        selection_status = "x_(k+1) rejected"


# =============================================================================
#     Plotting
# =============================================================================


    x1_bounds = margin * np.array([-old_radius, old_radius]) + x_k[0]
    x2_bounds = margin * np.array([-old_radius, old_radius]) + x_k[1]
    
    # Ensures x_opt is captured in the bounds should it be outside of the trust region
    x1_bounds[0] = min(x1_bounds[0], x_opt[0] - radius * margin)
    x2_bounds[0] = min(x2_bounds[0], x_opt[1] - radius * margin)
    x1_bounds[1] = max(x1_bounds[1], x_opt[0] + radius * margin)
    x2_bounds[1] = max(x2_bounds[1], x_opt[1] + radius * margin)


    x1 = np.linspace(*x1_bounds, contour_res)
    x2 = np.linspace(*x2_bounds, contour_res)
    X1, X2 = np.meshgrid(x1, x2)

    def black_box_log(x):
        output = f.black_box(x)
        if output > 0:
            return math.log(output)
        else:
            return -1e06

    black_box_Z = f.mat_func(X1, X2, black_box_log)
    if approx_k != None:
        fun_approx_Z = f.mat_func(X1, X2, approx_k)

    sub = fig.add_subplot(math.ceil(n_i/col), col, i+1)
    black_box_contour = sub.contour(
        X1, X2, black_box_Z, 10, colors='#cc2222', linewidths=[0.5])
    if approx_k != None:
        approx_contour = sub.contour(
            X1, X2, fun_approx_Z, 10, cmap='summer', linewidths=[0.5])

    # Plot trust region
    angle = np.linspace(0, 2*math.pi, 45)
    x = old_radius * np.cos(angle) + x_k[0]
    y = old_radius * np.sin(angle) + x_k[1]
    plt.plot(x, y, 'k--', linewidth=1)

    # Plot sample points
    for s in range(n_s):
        sub.plot(*samples[s], 'bo', markersize=0.5)

    # Plot all previous points in grey
    for j in range(len(x_all)):
        sub.plot(*x_all[j], marker='x', color='#888888')

    # If x_k is repeated, do not add it to the list of x_all
    x_k_repeated = False
    for j in range(len(x_all)):
        if (x_k == x_all[j]).all():
            x_k_repeated = True
    if not x_k_repeated:
        x_all = np.append(x_all, np.array([[*x_k]]), axis=0)

    # Plot all x's
    stepRejected = (x_k == x_opt).all()
    sub.plot(*x_k, 'kx')

    if stepRejected:
        sub.plot(*x_k1, 'rx')
    else:
        sub.plot(*x_opt, 'gx')
        sub.plot([x_k[0], x_opt[0]], [x_k[1], x_opt[1]],
                 linestyle='-', color='k', linewidth=1)

    # Enforce axes' bounds
    sub.set_xlim(x1_bounds.tolist())
    sub.set_ylim(x2_bounds.tolist())

    sub.title.set_text('Iteration: {}'.format(i) +
                       "\n" + r"f($\bf{x}$)=" + "{:.4f}".format(best_evaluation) +
                       "\n" + "r: {:.4f}".format(old_radius) +
                       "\n" + "{}".format(selection_status)) 

fig.tight_layout()
plt.savefig("plot.pdf")
plt.close(fig)
print("done")

from scipy.optimize import minimize
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functions as f


radius = 10         # Radius of Trust region
radius_tol = 0.1    # Distance from the boundary x_k1 must be less than to count as "on the boundary"
n_s = 10            # Number of samples per iteration
n_i = 30            # Number of iterations
eta_1 = 0.0         # Minimum decrease ratio for accepting new point
eta_2 = 0.1         # Minimum decrease ratio for increasing trust-region radius
beta_red = 0.2      # Reduction multiplier for trust-region radius
beta_inc = 1.1      # Increase multiplier for trust-region radius
dif_step = 1e-3     # Step size for finding approximate derivative

# Initial values
x_k = x_opt = np.array([10, 10])
sample_scale = 0.5
x_all = np.array([[*x_k]])
best_evaluation = math.inf

scale = 5
col = 3
fig = plt.figure(figsize=(scale*col, scale*math.ceil(n_i/col),))
contour_res = 250  # Number of points per axis plotted in the contour map
margin = 1.2       # How much bigger the contour plot is than the diameter of the trust region

for i in range(n_i):

    approx_type = ""

    x_k_old = x_k
    x_k = x_opt
    current_evaluation = best_evaluation
    old_radius = radius

    # Randomises selection of samples within a circle of radius delta
    # by randomising angle (0 to 2*pi) and distance (0 to delta)
    s = 0
    samples = np.array([[]])
    min_separation = radius/n_s
    for j in range(n_s):

        # Generates radial points using polar coordinates
        # angle = np.random.uniform(0, 2*math.pi)
        # distance = math.sqrt(np.random.uniform(0, radius**2)) # sqrt to ensure uniform distribution (we want the density to be the same for all distances)
        # sample = x_opt + np.array([distance*math.sin(angle),distance*math.cos(angle)]) # starts with random initial sample point

        # if j == 0:
        #   samples = np.array([[*sample]])
        # else:
        #   samples = np.append(samples, np.array([sample]), axis=0)

        # LHS implementation
        # Locates the centre of the bottom left cell of the latin hypercube (within bounds)
        lower_corner = (x_k - ((n_s-0.5)/n_s) * radius *
                        np.ones(2)).reshape((1, 2))
        # Matrix with each row as the lower_corner. Needed for correcting position of samples
        S_0 = lower_corner
        for j in range(n_s-1):
            S_0 = np.append(S_0, lower_corner, axis=0)
        samples = 2 * radius * f.latin_hypercube(n_s, 2) / n_s + S_0
        samples_evaluated = np.zeros(n_s)
        for j in range(n_s):
            samples_evaluated[j] = f.black_box(samples[j])

    # Generates 2 different surrogate functions: quadratic approximation and linear
    # Then chooses the point that gives the greatest decrease in f(x)

    a = np.zeros(6)
    cons = [{'type': 'ineq', 'fun': f.convexity_constraint}]
    def regression_func(a): return f.least_squares(
        a, f.quad_approx, samples)
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

    # Compares the performance of the linear and quadratic approximation, as well as the evaluations of the samples
    evaluations = [quad_evaluation, lin_evaluation, *samples_evaluated]
    best_evaluation = min(evaluations)
    if best_evaluation == quad_evaluation:
        x_k1 = x_k1_quad
        approx_k = quad_approx_k
        approx_type = "quadratic"
    elif best_evaluation == lin_evaluation:
        x_k1 = x_k1_lin
        approx_k = lin_approx_k
        approx_type = "linear"
    else:
        x_k1 = samples[evaluations.index(min(evaluations))-2]
        approx_k = None
        approx_type = "sample selected"
    distance = np.linalg.norm((x_k1-x_k))

    # Chooses whether to accept x_k1 as the next point
    # Adjusts radius accordingly
    if best_evaluation < current_evaluation:
        if distance > radius - radius_tol:
            radius = beta_inc * radius
        x_opt = x_k1
    else:
        radius = beta_red * radius
        x_opt = x_k

    #/------------------------------------------------------------------------------------------/#
    # Plotting

    x1_bounds = margin * np.array([-old_radius, old_radius]) + x_k[0]
    x2_bounds = margin * np.array([-old_radius, old_radius]) + x_k[1]

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
                       "\n" + r"f($\bf{x}$)=" + "{:.4f}".format(f.black_box(x_opt)) +
                       "\n" + "r: {:.4f}".format(old_radius) +
                       "\n" + "{}".format(approx_type))

fig.tight_layout()
plt.savefig("plot.pdf")
plt.close(fig)

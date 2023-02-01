import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.cost import QRCost, PathQRCost
from ilqr.dynamics import constrain
from ilqr.dynamics import tensor_constrain

from NavigationDynamics import NavigationDynamics

J_hist = []

# most_recent_is_complete = [converged, info, iteration_count]
most_recent_is_complete_packet = [None, None, None]

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")

    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)
    
    most_recent_is_complete = [converged, info, iteration_count]


def run_solver(exp):
    exp.reinit_file_id()

    x0_raw          = exp.get_start()    # initial state
    x_goal_raw      = exp.get_target_goal()

    # dynamics = AutoDiffDynamics(f, [x], [u], t)
    dynamics = NavigationDynamics(exp.get_dt())

    # Note that the augmented state is not all 0.
    x0      = dynamics.augment_state(np.array(x0_raw)).T
    x_goal  = dynamics.augment_state(np.array(x_goal_raw)).T

    N       = exp.get_N()
    dt      = exp.get_dt()

    x_T     = N
    Xrefline = np.tile(x_goal_raw, (N+1, 1))
    Xrefline = np.reshape(Xrefline, (-1, 2))

    u_blank = np.asarray([0.0, 0.0])
    Urefline = np.tile(u_blank, (N, 1))
    Urefline = np.reshape(Urefline, (-1, 2))

    state_size  = 2 #
    action_size = 2 # 

    ### EXP IS USED AFTER THIS POINT
    cost = exp.setup_cost(Xrefline, Urefline)

    # l = leg_cost.l
    # l_terminal = leg_cost.term_cost
    # cost = AutoDiffCost(l, l_terminal, x_inputs, u_inputs)

    FLAG_JUST_PATH = False
    if FLAG_JUST_PATH:
        traj        = Xrefline
        us_init     = Urefline
        cost        = PathQRCost(Q, R, traj, us_init)
        print("Set to old school pathing")
        exit()

    # default value from text
    max_reg = 1e-10 #None # default value is 1e-10
    ilqr = iLQR(dynamics, cost, N, max_reg=None)

    tol = 1e-6
    # tol = 1e-10

    num_iterations = 200

    start_time = time.time()
    xs, us = ilqr.fit(x0_raw, Urefline, tol=tol, n_iterations=num_iterations, on_iteration=on_iteration)
    end_time = time.time()

    t = np.arange(N) * dt
    theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
    theta_dot = xs[:, 1]

    # Plot of the path through space
    verts = xs
    xs, ys = zip(*verts)
    gx, gy = zip(*exp.get_goals())
    sx, sy = zip(*[x0_raw])

    is_completed = most_recent_is_complete_packet

    elapsed_time = end_time - start_time
    cost.graph_legibility_over_time(verts, us, elapsed_time=elapsed_time, status_packet=is_completed)

    return verts, us, cost, most_recent_is_complete_packet

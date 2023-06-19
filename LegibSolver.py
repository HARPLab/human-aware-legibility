import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.cost import QRCost, PathQRCost
from ilqr.dynamics import constrain
from ilqr.dynamics import tensor_constrain

from NavigationDynamics import NavigationDynamics
import LegibTestScenarios as test_scenarios


J_hist = []

# most_recent_is_complete = [converged, info, iteration_count]
most_recent_is_complete_packet = [None, None, None]

# class iLQR_plus(iLQR):
#     def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False):
#         iLQR.__init__(self, dynamics, cost, N, max_reg=1e10, hessians=False)
#         self.most_recent_is_complete_packet = [None, None, None]

def on_iteration_default(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")

    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)
    
    most_recent_is_complete_packet = [converged, info, iteration_count]

def run_solver(exp):
    STATIC_ANGLE_DEFAULT = 360
    dash_folder = exp.get_run_filters()[test_scenarios.DASHBOARD_FOLDER]
    exp.reinit_file_id()

    state_size  = 3 #
    action_size = 2 #

    exp.set_state_size(state_size)
    exp.set_action_size(action_size)

    start   = exp.get_start()
    goal    = exp.get_target_goal()

    x0_raw          = np.asarray([start[0],    start[1],   STATIC_ANGLE_DEFAULT]).T
    x_goal_raw      = np.asarray([goal[0],     goal[1],    STATIC_ANGLE_DEFAULT]).T

    if state_size == 2:
        x_goal_raw = x_goal_raw[:2]
        x0_raw = x0_raw[:2]

    # dynamics = AutoDiffDynamics(f, [x], [u], t)
    dynamics = NavigationDynamics(exp.get_dt(), exp)

    # Note that the augmented state is not all 0.
    # x0      = dynamics.augment_state(np.array(x0_raw)).T
    # x_goal  = dynamics.augment_state(np.array(x_goal_raw)).T

    N       = exp.get_N()
    dt      = exp.get_dt()

    x_T      = N

    ### EXP IS USED AFTER THIS POINT
    cost, Urefline = exp.setup_cost(state_size, action_size, x_goal_raw, N)

    FLAG_JUST_PATH = False
    if FLAG_JUST_PATH:
        traj        = Xrefline
        us_init     = Urefline
        cost        = PathQRCost(Q, R, traj, us_init)
        print("Set to old school pathing")

    # default value from text
    # If this is set to none, it will over-search to find stuff
    max_reg = None # default value is 1e-10
    ilqr = iLQR(dynamics, cost, N, max_reg=max_reg)

    cost.init_output_log(dash_folder)

    tol = 1e-4 #8 #1e-5
    num_iterations = 100

    if exp.get_run_filters()[test_scenarios.SCENARIO_FILTER_FAST_SOLVE] is True:
        num_iterations = 1

    on_iteration_exp = exp.on_iteration_exp

    start_time = time.time()
    xs, us = ilqr.fit(x0_raw, Urefline, tol=tol, n_iterations=num_iterations, on_iteration=on_iteration_exp)
    xs_best = exp.best_xs
    us_best = exp.best_us

    end_time = time.time()

    t = np.arange(N) * dt

    # Plot of the path through space
    verts = xs

    if state_size == 3:
        xs, ys, thetas = zip(*verts)
        sx, sy, stheta = zip(*[x0_raw])
    else:
        xs, ys = zip(*verts)
        sx, sy = zip(*[x0_raw])
  
    gx, gy = zip(*exp.get_goals())


    elapsed_time = end_time - start_time

    suptitle = exp.get_suptitle()

    most_recent_is_complete_packet = exp.get_solver_status()
    cost.graph_legibility_over_time(verts, us, elapsed_time=elapsed_time, status_packet=most_recent_is_complete_packet, dash_folder=dash_folder, suptitle=suptitle)

    # fn = cost.get_export_label() + "path_for_unity"
    # exp.get_restaurant().export_paths_csv(verts, fn)

    return verts, us, cost, most_recent_is_complete_packet

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
    STATIC_ANGLE_DEFAULT = 0
    dash_folder = exp.get_run_filters()[test_scenarios.DASHBOARD_FOLDER]
    exp.reinit_file_id()

    state_size  = 3 #
    action_size = 2 #

    start   = exp.get_start()
    goal    = exp.get_target_goal()

    x0_raw          = np.asarray([start[0],    start[1],   STATIC_ANGLE_DEFAULT]).T
    x_goal_raw      = np.asarray([goal[0],     goal[1],    STATIC_ANGLE_DEFAULT]).T

    # dynamics = AutoDiffDynamics(f, [x], [u], t)
    dynamics = NavigationDynamics(exp.get_dt(), exp)

    # Note that the augmented state is not all 0.
    x0      = dynamics.augment_state(np.array(x0_raw)).T
    x_goal  = dynamics.augment_state(np.array(x_goal_raw)).T

    N       = exp.get_N()
    dt      = exp.get_dt()

    x_T      = N
    Xrefline = np.tile(x_goal_raw, (N+1, 1))
    Xrefline = np.reshape(Xrefline, (-1, 3))

    u_blank  = np.asarray([0.0, 0.0])
    Urefline = np.tile(u_blank, (N, 1))
    Urefline = np.reshape(Urefline, (-1, 2))

    # print(Xrefline)
    # print(Urefline)

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

    cost.init_output_log(dash_folder)

    tol = 1e-5
    # tol = 1e-10

    num_iterations = 100

    if exp.get_run_filters()[test_scenarios.SCENARIO_FILTER_FAST_SOLVE] is True:
        num_iterations = 1

    on_iteration_exp = exp.on_iteration_exp

    start_time = time.time()
    xs, us = ilqr.fit(x0_raw, Urefline, tol=tol, n_iterations=num_iterations, on_iteration=on_iteration_exp)

    end_time = time.time()

    t = np.arange(N) * dt
    theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
    theta_dot = xs[:, 1]

    # Plot of the path through space
    verts = xs
    xs, ys, thetas = zip(*verts)
    gx, gy = zip(*exp.get_goals())
    sx, sy, stheta = zip(*[x0_raw])

    elapsed_time = end_time - start_time

    suptitle = exp.get_suptitle()

    most_recent_is_complete_packet = exp.get_solver_status()
    cost.graph_legibility_over_time(verts, us, elapsed_time=elapsed_time, status_packet=most_recent_is_complete_packet, dash_folder=dash_folder, suptitle=suptitle)

    # fn = cost.get_export_label() + "path_for_unity"
    # exp.get_restaurant().export_paths_csv(verts, fn)

    return verts, us, cost, most_recent_is_complete_packet

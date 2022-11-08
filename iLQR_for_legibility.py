#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import os
import sys
import copy
import time

module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

import theano.tensor as T

from ilqr import iLQR
from ilqr.cost import QRCost, PathQRCost
from ilqr.dynamics import constrain
from ilqr.dynamics import tensor_constrain

import PathingExperiment as ex
from LegiblePathQRCost import LegiblePathQRCost
from DirectPathQRCost import DirectPathQRCost
from ObstaclePathQRCost import ObstaclePathQRCost
from LegibilityOGPathQRCost import LegibilityOGPathQRCost
from OALegiblePathQRCost import OALegiblePathQRCost
from NavigationDynamics import NavigationDynamics

from sklearn.preprocessing import normalize

import utility_environ_descrip as resto

J_hist = []
dynamics = None

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")

    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)


def get_window_dimensions_for_envir(self, start, goals, pts):
    xmin, xmax, ymin, ymax = 0.0, 0.0, 0.0, 0.0

    all_points = copy.copy(goals)
    all_points.append(start)
    all_points.append(pts)
    for pt in all_points:
        x, y = pt[0], pt[1]

        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y

    xwidth      = xmax - xmin
    yheight     = ymax - ymin

    xbuffer     = .1 * xwidth
    ybuffer     = .1 * yheight

    return xmin - xbuffer, xmax + xbuffer, ymin - ybuffer, ymax + ybuffer

def scenario_0():
    start           = [1.0, 0.01]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    # goal1           = [0.0, 6.0]
    # goal3           = [2.0, 6.0]
    goal1 = [0.0, 18.0]
    goal3 = [2.0, 18.0]

    goal1 = [0.0, 6.0]
    goal3 = [2.0, 6.0]
    goal2 = [4.0, 4.0]

    goal1 = [0.0, 6.0]
    goal3 = [2.0, 9.0]

    target_goal = goal3
    all_goals   = [goal1, goal3]

    exp = ex.PathingExperiment(start, target_goal, all_goals)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    return exp

def scenario_1():
    restaurant      = None
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal1

    all_goals   = [goal1, goal3]

    exp = ex.PathingExperiment(start, target_goal, all_goals)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    return exp


def scenario_2():
    restaurant      = None
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal4

    all_goals   = [goal1, goal4, goal2]

    exp = ex.PathingExperiment(start, target_goal, all_goals)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 11 #42 #21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    return exp

def scenario_3():
    restaurant      = None
    start           = [8.0, 2.0]

    true_goal       = [0.0, 0.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal3
    # start = goal3

    all_goals   = [goal1, goal3, goal2]

    exp = ex.PathingExperiment(start, target_goal, all_goals)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    return exp

def scenario_5_large_scale():
    restaurant      = None
    start           = [800.0, 200.0]

    true_goal       = [0.0, 0.0]
    goal2           = [200.0, 100.0]
    goal3           = [400.0, 100.0]

    goal1           = [400.0, 200.0]
    goal3           = [100.0, 300.0]

    target_goal = goal3
    # start = goal3

    all_goals   = [goal1, goal3, goal2]
    
    exp = ex.PathingExperiment(start, target_goal, all_goals)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    return exp

def scenario_4_has_obstacles():
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal4
    all_goals   = [goal1, goal4, goal2]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    table_pts.append([3.0, 0.5])

    exp = ex.PathingExperiment(start, target_goal, all_goals, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = int(21 * 1.5)
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size()) * 4
    Qf = np.identity(2) * 400.0

    # print(Q, R, Qf)

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 10000000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return exp

def scenario_4_has_obstacles_and_observer():
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal4
    all_goals   = [goal1, goal4, goal2]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    table_pts.append([1.0, 1.0])
    table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 0.0
    exp.set_solver_scale_obstacle(obs_scale)

    return exp

def scenario_7_observer():
    start           = [0.0, 2.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 1.0]
    goal3           = [4.0, 3.0]

    target_goal = goal1
    all_goals   = [goal1, goal3]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([3.5, 1.0, 0])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = int(21 * 2)
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 0.0
    exp.set_solver_scale_obstacle(obs_scale)

    return exp

def scenario_7_observer():
    start           = [100.0, 102.0]

    true_goal       = [108.0, 102.0]
    goal2           = [102.0, 101.0]
    goal4           = [104.0, 101.0]

    goal1           = [104.0, 101.0]
    goal3           = [104.0, 103.0]

    target_goal = goal3
    all_goals   = [goal1, goal3]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([3.5, 1.0, 0])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = int(21 * 2)
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 0.0
    exp.set_solver_scale_obstacle(obs_scale)

    return exp

def scenario_6():
    restaurant      = None
    start           = [0, 0]

    true_goal       = [0.0, 0.0]
    goal2           = [2.5, 1.0]
    goal4           = [4.0, 1.5]

    goal1           = [5.0, 2.0]
    goal3           = [1.0, 3.5]

    target_goal = goal4
    # start = goal3

    all_goals   = [goal1, goal3, goal2, goal4]

    exp = ex.PathingExperiment(start, target_goal, all_goals)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 21
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    return exp


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

    ilqr = iLQR(dynamics, cost, N)

    start_time = time.time()
    xs, us = ilqr.fit(x0_raw, Urefline, tol=1e-6, n_iterations=N, on_iteration=on_iteration)
    end_time = time.time()

    t = np.arange(N) * dt
    theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
    theta_dot = xs[:, 1]

    # Plot of the path through space
    verts = xs
    xs, ys = zip(*verts)
    gx, gy = zip(*exp.get_goals())
    sx, sy = zip(*[x0_raw])

    elapsed_time = end_time - start_time
    cost.graph_legibility_over_time(verts, us, elapsed_time=elapsed_time)


def exp_observers():
    return

def exp_obstacles():
    ####################################
    ### SET UP EXPERIMENT

    # exp = scenario_7_observer()
    exp = scenario_4_has_obstacles()
    # exp = scenario_4_has_obstacles_and_observer()
    # exp = scenario_2()
    # exp = scenario_5_large_scale()

    ### STANDARD SOLVE DEFAULTS
    print("Running solver")

    ###### COST/SOLVER OPTIONS
    # exp.set_cost_label(ex.COST_LEGIB)
    exp.set_cost_label(ex.COST_OBS)
    # exp.set_cost_label(ex.COST_OA)

    ###### WEIGHTING FUNCTION 
    ###    (DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
    # exp.set_f_label(ex.F_VIS)
    exp.set_f_label(ex.F_NONE)
    exp.set_solver_scale_obstacle(10000000.0)

    run_solver(exp)

    ###### RUN ADDITIONAL REMIXES
    # exp.set_f_label(ex.F_NONE)
    # run_solver(exp)

    # exp.set_f_label(ex.F_ANCA_LINEAR)
    # run_solver(exp)

def main():
    print("Setting up experiment")
    exp_obstacles()

    print("Done")

if __name__ == "__main__":
    main()
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
import LegibTestScenarios as test_scenarios
import LegibTestExperiments as test_exp
import LegibSolver as solver

from sklearn.preprocessing import normalize

import utility_environ_descrip as resto

dynamics = None

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

def exp_plain_symmetrical(goal_index=None, obs_angle=0):
    ####################################
    ### SET UP EXPERIMENT

    exp = test_scenarios.scenario_7_observer_on_zero(goal_index=goal_index, obs_angle=obs_angle)
    # exp = scenario_7_mirrored(goal_index=goal_index, obs_angle=obs_angle)

    ### STANDARD SOLVE DEFAULTS
    print("Running solver")

    ###### COST/SOLVER OPTIONS
    # exp.set_cost_label(ex.COST_LEGIB)
    exp.set_cost_label(ex.COST_OA_AND_OBS)

    N = 31
    dt = 0.1 #025

    exp.set_N(N)
    exp.set_dt(dt)

    ###### WEIGHTING FUNCTION 
    ###    (DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
    # exp.set_f_label(ex.F_VIS_LIN)
    exp.set_f_label(ex.F_VIS_BIN)

    solver.run_solver(exp)

def exp_observers(goal_index=None, obs_angle=None):
    ####################################
    ### SET UP EXPERIMENT

    exp = test_scenarios.scenario_7_observer_on_zero(goal_index=goal_index, obs_angle=obs_angle)
    # exp = scenario_7_mirrored(goal_index=goal_index, obs_angle=obs_angle)

    ### STANDARD SOLVE DEFAULTS
    print("Running solver")

    ###### COST/SOLVER OPTIONS
    exp.set_cost_label(ex.COST_OA)

    N = 31
    dt = 0.1 #025

    exp.set_N(N)
    exp.set_dt(dt)

    exp.set_solver_scale_term(10000.0)

    # exp.set_R(200.0 * np.eye(2))

    ###### WEIGHTING FUNCTION 
    ###    (DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
    # exp.set_f_label(ex.F_VIS_LIN)
    exp.set_f_label(ex.F_VIS_BIN)
    # exp.set_f_label(ex.F_NONE)

    solver.run_solver(exp)

def exp_obstacles(goal_index=None):
    ####################################
    ### SET UP EXPERIMENT

    # exp = test_scenarios.scenario_7_observer()
    exp = test_scenarios.scenario_4_has_obstacles(goal_index=goal_index)
    # exp = test_scenarios.scenario_4_has_obstacles_and_observer()
    # exp = test_scenarios.scenario_2()
    # exp = test_scenarios.scenario_5_large_scale()

    ### STANDARD SOLVE DEFAULTS
    print("Running solver")

    ###### COST/SOLVER OPTIONS
    # exp.set_cost_label(ex.COST_LEGIB)
    # exp.set_cost_label(ex.COST_OBS)
    exp.set_cost_label(ex.COST_OA_AND_OBS)

    ###### WEIGHTING FUNCTION 
    ###    (DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
    exp.set_f_label(ex.F_VIS_LIN)
    exp.set_f_label(ex.F_VIS_BIN)
    # exp.set_f_label(ex.F_NONE)

    exp.set_solver_scale_obstacle(1000.0)
    exp.set_solver_scale_term(1000.0)

    solver.run_solver(exp)

    ###### RUN ADDITIONAL REMIXES
    # exp.set_f_label(ex.F_NONE)
    # run_solver(exp)

    # exp.set_f_label(ex.F_ANCA_LINEAR)
    # run_solver(exp)

def exp_obstacles_and_observers(goal_index=None, obs_angle=0):
    ####################################
    ### SET UP EXPERIMENT
    exp = test_scenarios.scenario_4_has_obstacles_and_observer(goal_index=goal_index)

    ### STANDARD SOLVE DEFAULTS
    print("Running solver")

    ###### COST/SOLVER OPTIONS
    # exp.set_cost_label(ex.COST_LEGIB)
    # exp.set_cost_label(ex.COST_OBS)
    # exp.set_cost_label(ex.COST_OA)
    exp.set_cost_label(ex.COST_OA_AND_OBS)

    ###### WEIGHTING FUNCTION 
    ###    (DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
    exp.set_f_label(ex.F_VIS_BIN)
    # exp.set_f_label(ex.F_VIS_LIN)
    # exp.set_f_label(ex.F_NONE)

    # exp.set_solver_scale_obstacle(100000000.0)
    exp.set_solver_scale_obstacle(100000.0)
    # exp.set_solver_scale_obstacle(0.0)
    # exp.set_solver_scale_obstacle(100000.0)
    # exp.set_solver_scale_term(10000000000000.0)
    exp.set_solver_scale_term(1000.0)
 
    exp.set_N(30)

    solver.run_solver(exp)

    ###### RUN ADDITIONAL REMIXES
    # exp.set_f_label(ex.F_NONE)
    # run_solver(exp)

    # exp.set_f_label(ex.F_ANCA_LINEAR)
    # run_solver(exp)

def main():
    print("Setting up experiment")

    # # exp_plain_symmetrical(goal_index=1)
    # # exp_obstacles(goal_index=1)
    # exp_obstacles_and_observers(goal_index=1, obs_angle=90)
    # # exp_obstacles(goal_index=1)
    # # exp_observers(goal_index=1, obs_angle=180)

    print("Done")

if __name__ == "__main__":
    main()
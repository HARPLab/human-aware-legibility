# from __future__ import print_function

import os
import sys
import copy
import time

module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
# import matplotlib.pyplot as plt
# from datetime import timedelta, datetime

# import theano.tensor as T

# from ilqr import iLQR
# from ilqr.cost import QRCost, PathQRCost
# from ilqr.dynamics import constrain
# from ilqr.dynamics import tensor_constrain

import PathingExperiment as ex
# from LegiblePathQRCost import LegiblePathQRCost
# from DirectPathQRCost import DirectPathQRCost
# from ObstaclePathQRCost import ObstaclePathQRCost
# from LegibilityOGPathQRCost import LegibilityOGPathQRCost
# from OALegiblePathQRCost import OALegiblePathQRCost
# from NavigationDynamics import NavigationDynamics

import utility_environ_descrip as resto


def scenario_1():
	label = "TEST1"
	full_label = (label)


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

	return full_label, ex.PathingExperiment(start, target_goal, all_goals)

def scenario_0(goal_index=None):
    label = "scenario_0"
    
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

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

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

    return label, exp

def scenario_1(goal_index=None):
    label = "scenario_0"

    restaurant      = None
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal1

    all_goals   = [goal1, goal3]

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

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

    return label, exp


def scenario_2(goal_index=None):
    label = "scenario_2"

    restaurant      = None
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal4

    all_goals   = [goal1, goal4, goal2]

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

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

    return label, exp

def scenario_3(goal_index=None):
    label = "scenario_3"

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

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

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

    return label, exp

def scenario_5_large_scale(goal_index=None):
    label = "scenario_5_large_scale"

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
    
    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

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

    return label, exp

def scenario_4_has_obstacles(goal_index=None):
    label = "scenario_4_has_obs"

    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    all_goals   = [goal1, goal4, goal2]
    if goal_index != None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = goal4

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])
    table_pts.append([1.0, 1.0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = int(41)
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 1.0 * np.eye(exp.get_action_size())
    Qf = 1.0 * np.identity(2)

    # print(Q, R, Qf)

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 100000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_4_has_obstacles_and_observer(goal_index=None):
    label = "scenario_4_has_obs"

    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    all_goals   = [goal1, goal4, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = 31
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_7_observer(goal_index=None, obs_angle=0):
    label = "scenario_7_obs"

    start           = [0.0, 2.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 1.0]
    goal3           = [4.0, 3.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal3
    else:
        target_goal = all_goals[goal_index]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([3.5, 1.0, obs_angle])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

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

    return label, exp

def scenario_7_observer_offset(goal_index=None):
    label = "scenario_7_offset"

    start           = [100.0, 102.0]

    true_goal       = [108.0, 102.0]
    goal2           = [102.0, 101.0]
    goal4           = [104.0, 101.0]

    goal1           = [104.0, 101.0]
    goal3           = [104.0, 103.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal3
    else:
        target_goal = all_goals[goal_index]
    
    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([3.5, 1.0, 0])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

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

    return label, exp

def scenario_7_observer_on_zero(goal_index=None, obs_angle=0):
    label = "scenario_7_obs_on_zero"

    start           = [0.0, 0.0]

    goal1           = [2.0, -1.0]
    goal3           = [2.0, 1.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal3
    else:
        target_goal = all_goals[goal_index]
    
    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([.5, -1.0, obs_angle])
    obs_pts.append([1.75, -1.0, obs_angle])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    exp.set_state_size(2)
    exp.set_action_size(2)

    dt = .025
    N = int(21)
    Q = 1.0 * np.eye(exp.get_state_size())
    R = 200.0 * np.eye(exp.get_action_size())
    Qf = np.identity(2) * 400.0

    exp.set_QR_weights(Q, R, Qf)
    exp.set_N(N)
    exp.set_dt(dt)

    obs_scale = 0.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp


def scenario_7_observer_rot90(goal_index=None):
    label = "scenario_7_rot_90"

    start           = [0.0, 2.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 1.0]
    goal3           = [4.0, 3.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal3
    else:
        target_goal = all_goals[goal_index]
    
    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([3.5, 1.0, 0])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

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

    return label, exp

def scenario_7_mirrored(goal_index=None, obs_angle=0):
    label = "scenario_7_mirrored"

    start           = [-0.0, -2.0]

    true_goal       = [-8.0, -2.0]
    goal2           = [-2.0, -1.0]
    goal4           = [-4.0, -1.0]

    goal1           = [-4.0, -1.0]
    goal3           = [-4.0, -3.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal3
    else:
        target_goal = all_goals[goal_index]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([3.5, 1.0, obs_angle])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

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

    return label, exp

def scenario_6():
    label = "scenario_6"

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

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

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

    return label, exp

def get_scenario_set():
	scenarios = {}

	# TEST SCENARIO
	label, exp = scenario_0()
	scenarios[label] = exp

	# TEST SCENARIO
	label, exp = scenario_1()
	scenarios[label] = exp

	# TEST SCENARIO
	label, exp = scenario_2()
	scenarios[label] = exp

	# TEST SCENARIO
	label, exp = scenario_3()
	scenarios[label] = exp

	return scenarios


def get_scenarios():
	return get_scenario_set()

def get_scenarios_heading():
	return get_scenario_set()

def get_scenarios_obstacles():
	all_scenarios = get_scenarios()
	new_list = {}

	for key in all_scenarios.keys():
		scenario = all_scenarios[key]

		if len(scenario.get_tables()) > 0:
			new_list[key] = scenario

	return new_list

def get_scenarios_observers():
	all_scenarios = get_scenarios()
	new_list = {}

	for key in all_scenarios.keys():
		scenario = all_scenarios[key]

		if len(scenario.get_observers()) > 0:
			new_list[key] = scenario

	return new_list











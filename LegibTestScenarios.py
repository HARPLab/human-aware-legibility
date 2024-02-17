# from __future__ import print_function

import os
import sys
import copy
import time

module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

import PathingExperiment as ex
import utility_environ_descrip as resto

SCENARIO_FILTER_MINI = 'mini'
SCENARIO_FILTER_FAST_SOLVE = 'fastsolve'
DASHBOARD_FOLDER = None


def scenario_aimakerspace_no_obs(goal_index=None):
    label = "pilot_no_obs" # _g" + str(goal_index)


    start = [0.0, 10.0]
    goal1 = [10.0, 8.0]
    goal3 = [5.0, 12.0]

    obs_pts = []

    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    
    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp


def scenario_triangle_equid(goal_index=None):
    label = "tri_equid" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal2 = [-5.0, 0.0]

    goal1 = [5.0, 0.0]
    goal3 = [0.0, 8.66]

    obs2 = [goal2[0] - 1.5, goal2[1], 30]

    obs1 = [goal3[0], goal3[1] + 1.5, 270]
    obs3 = [goal1[0] + 1.5, goal1[1], 150]

    obs_pts = []
    obs_pts.append(obs1)
    obs_pts.append(obs2)
    obs_pts.append(obs3)

    start       = goal2
    target_goal = goal3
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal3])

    N = 20 #26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_triangle_thin(goal_index=None):
    label = "tri_isos" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal2 = [-3.0, 0.0]
    goal1 = [3.0, 0.0]

    goal3 = [0.0, -8.66]

    obs3 = [goal3[0], goal3[1] + 1.5, 270]

    obs2 = [goal2[0] - 1.5, goal2[1], 0]
    obs1 = [goal1[0] + 1.5, goal1[1], 180]

    obs_pts = []
    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append()

    start       = goal3
    target_goal = goal2
    all_goals   = [goal1, goal2]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2])

    N = 12 #26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_triangle_thin_alt(goal_index=None):
    label = "tri_isos" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal2 = [-3.0, 0.0]
    goal1 = [3.0, 0.0]

    goal3 = [0.0, -8.66]

    obs3 = [goal3[0], goal3[1] + 1.5, 270]

    obs2 = [goal2[0] - 1.5, goal2[1], 0]
    obs1 = [goal1[0] + 1.5, goal1[1], 180]

    obs_pts = []
    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append()

    start       = goal3
    target_goal = goal2
    all_goals   = [goal1, goal2, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2])

    N = 20 #26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp


def scenario_triangle2(goal_index=None):
    label = "tri" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    start = [0.0, 8.66]
    goal1 = [5.0, 0.0]
    goal2 = [-5.0, 0.0]

    obs_pts = []
    obs_pts.append([start[0], start[1] + 1.5, 270])
    obs_pts.append([goal1[0] + 1.5, goal1[1], 150])
    obs_pts.append([goal2[0] - 1.5, goal2[1], 30])

    target_goal = goal1
    all_goals   = [goal1, goal2]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), [start, goal1, goal2])

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_diamond(goal_index=None):
    label = "diam" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [0.0, -5.0]
    goal3 = [10.0, 0.0]
    goal2 = [0.0, 5.0]
    goal1 = [-10.0, 0.0]

    start       = goal1

    obs_pts = []
    obs1 = [goal1[0] - 1.5, goal1[1], 0]
    obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs_pts.append(obs1)
    obs_pts.append(obs2)
    obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)

    target_goal = goal1
    all_goals   = [goal2, goal3, goal4]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal3, goal4])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent_asymm0(goal_index=None):
    label = "pent2_0" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [1.0, -5.0]
    goal2 = [-1.0, 5.0]

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -5.0]
    goal32 = [10.0, 5.0]

    start       = goal1

    obs_pts = []
    obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    target_goal = goal1
    all_goals   = [goal2, goal4, goal34, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    # exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent_asymm1(goal_index=None):
    label = "pent2_1" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [1.0, -5.0]
    goal2 = [-1.0, 5.0]

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -5.0]
    goal32 = [10.0, 5.0]

    obs_pts = []
    obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)

    start       = goal2
    target_goal = goal2
    all_goals   = [goal1, goal4, goal34, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent_asymm2(goal_index=None):
    label = "pent2_2" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [1.0, -5.0]
    goal2 = [-1.0, 5.0]

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -5.0]
    goal32 = [10.0, 5.0]

    start       = goal4

    obs_pts = []
    obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    target_goal = goal4
    all_goals   = [goal1, goal2, goal34, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    # exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent_asymm3(goal_index=None):
    label = "pent2_3" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [1.0, -5.0]
    goal2 = [-1.0, 5.0]

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -5.0]
    goal32 = [10.0, 5.0]

    start       = goal34

    obs_pts = []
    obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    # target_goal = goal34
    all_goals   = [goal1, goal2, goal4, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent_asymm4(goal_index=None):
    label = "pent2_4" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [1.0, -5.0]
    goal2 = [-1.0, 5.0]

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -5.0]
    goal32 = [10.0, 5.0]

    start       = goal32

    obs_pts = []
    obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    # target_goal = goal32
    all_goals   = [goal1, goal2, goal4, goal34]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    # exp.set_observer_goal_pairs(exp.get_observers(), [goal1, goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent_wide(goal_index=None):
    label = "pent_w" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal4 = [0.0, -5.0]
    goal2 = [0.0, 5.0]

    goal34 = [10.0, -3.0]
    goal32 = [10.0, 3.0]

    start       = goal1

    obs_pts = []
    # obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    # obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    target_goal = goal1
    all_goals   = [goal2, goal4, goal34, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    exp.set_observer_goal_pairs(exp.get_observers(), [goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_pent(goal_index=None):
    label = "pent" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [0.0, -5.0]
    goal2 = [0.0, 5.0]

    goal1 = [-10.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -5.0]
    goal32 = [10.0, 5.0]

    start       = goal1

    obs_pts = []
    # obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0], goal4[1] - 1.5, 90]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]

    obs34 = [goal34[0], goal34[1] - 1.5, 90]
    obs32 = [goal32[0], goal32[1] + 1.5, 270]

    # obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    target_goal = goal1
    all_goals   = [goal2, goal4, goal34, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    exp.set_observer_goal_pairs(exp.get_observers(), [goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_fan(goal_index=None):
    label = "fan" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [10.0, -5.0]
    goal2 = [10.0, 5.0]

    goal1 = [0.0, 0.0]

    # goal3 = [0.0, -5.0]

    goal34 = [10.0, -10.0]
    goal32 = [10.0, 10.0]

    start       = goal1

    obs_pts = []
    # obs1 = [goal1[0] - 1.5, goal1[1], 0]
    # obs3 = [goal3[0] + 1.5, goal3[1], 180]

    obs4 = [goal4[0] + 1.5, goal4[1], 180]
    obs2 = [goal2[0] + 1.5, goal2[1], 180]

    obs34 = [goal34[0] + 1.5, goal34[1], 180]
    obs32 = [goal32[0] + 1.5, goal32[1], 180]

    # obs_pts.append(obs1)
    obs_pts.append(obs2)
    # obs_pts.append(obs3)
    # obs_pts.append(obs3)
    obs_pts.append(obs4)
    obs_pts.append(obs34)
    obs_pts.append(obs32)


    target_goal = goal1
    all_goals   = [goal2, goal4, goal34, goal32]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order

    exp.set_observer_goal_pairs(exp.get_observers(), [goal2, goal4, goal34, goal32])
    # exp.set_observer_goal_pairs([resto.Observer(obs2[0], obs2[1]), resto.Observer(obs4[0], obs4[1])], [goal2, goal4])

    N = 15
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_parallelogram(goal_index=None):
    label = "para" # _g" + str(goal_index)

    # (2, 4), (4,2), (3,6), (5,4))
    goal4 = [2.0, 4.0]
    goal3 = [4.0, 2.0]
    goal2 = [3.0, 6.0]
    goal1 = [5.0, 4.0]

    start       = goal4

    obs_pts = []
    obs3 = [goal3[0], goal3[1] - 1.5, 90]
    obs1 = [goal1[0] + 1.5, goal1[1], 180]
    obs2 = [goal2[0], goal2[1] + 1.5, 270]
    obs4 = [start[0] - 1.5, start[1], 0]


    obs_pts.append(obs1)
    obs_pts.append(obs2)
    obs_pts.append(obs3)
    obs_pts.append(obs4)

    target_goal = goal1
    all_goals   = [goal1, goal2, goal3, goal4]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), [start, goal1, goal2, goal3])

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_aimakerspace(goal_index=None):
    label = "pilot" # _g" + str(goal_index)


    start = [0.0, 10.0]
    goal3 = [5.0, 12.0]
    goal1 = [10.0, 8.0]

    obs_pts = []
    obs_pts.append([goal3[0], goal3[1] + 1.5, 270])
    obs_pts.append([goal1[0] + 1.5, goal1[1], 180])

    target_goal = goal1
    all_goals   = [goal3, goal1]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), all_goals[::-1])

    N = 16
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_equidist3(goal_index=None):
    label = "pilot_equidist3" # _g" + str(goal_index)


    start = [0.0, 0.0]
    goal3 = [5.0, 2.5]
    goal1 = [5.0, -2.5]

    obs_pts = []
    obs_pts.append([goal3[0], goal3[1] + 1.5, 270])
    obs_pts.append([goal1[0] + 1.5, goal1[1], 180])


    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), all_goals[::-1])

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_equidist2(goal_index=None):
    label = "pilot_equidist2" # _g" + str(goal_index)


    start = [0.0, 0.0]
    goal3 = [5.0, 5.0]
    goal1 = [5.0, -5.0]

    obs_pts = []
    obs_pts.append([goal3[0], goal3[1] + 1.5, 270])
    obs_pts.append([goal1[0] + 1.5, goal1[1], 180])


    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), all_goals[::-1])

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp


def scenario_equidist(goal_index=None):
    label = "pilot_equidist" # _g" + str(goal_index)

    start = [0.0, 0.0]
    goal3 = [0.0, 10.0]
    goal1 = [10.0, 0.0]

    obs_pts = []
    obs_pts.append([goal3[0], goal3[1] + 1.5, 270])
    obs_pts.append([goal1[0] + 1.5, goal1[1], 180])


    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), all_goals[::-1])

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp


def scenario_aimakerspace_long(goal_index=None):
    label = "pilot_long" # _g" + str(goal_index)


    start = [0.0, 10.0]
    goal3 = [55.0, 12.0]
    goal1 = [60.0, 8.0]

    obs_pts = []
    obs_pts.append([goal3[0], goal3[1] + 1.5, 270])
    obs_pts.append([goal1[0] + 1.5, goal1[1], 180])


    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    table_pts = []

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)
    # Make sure these have the same order
    exp.set_observer_goal_pairs(exp.get_observers(), all_goals[::-1])

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_a(goal_index=None):
    label = "testa" # _g" + str(goal_index)

    # start           = [1.0, 3.0]

    # goal1 = [3.0, 4.0]
    # goal3 = [3.0, 2.0]

    start = [0.0, 0.0]
    goal3 = [10.0, -5.0]
    goal1 = [10.0, 5.0]

    # start           = [0.0, 0.0]
    # goal1 = [-1.0, 2.0]
    # goal3 = [1.0, 2.0]

    # start = [0.0, 0.0]
    # goal1 = [-2, 1.0]
    # goal3 = [-2, -1.0]

    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_c(goal_index=None):
    label = "testc" # _g" + str(goal_index)

    # start           = [1.0, 3.0]

    # goal1 = [3.0, 4.0]
    # goal3 = [3.0, 2.0]

    start = [0.0, 0.0]
    goal3 = [1.00, .50]
    goal1 = [1.00, 1.50]

    # start           = [0.0, 0.0]
    # goal1 = [-1.0, 2.0]
    # goal3 = [1.0, 2.0]

    # start = [0.0, 0.0]
    # goal1 = [-2, 1.0]
    # goal3 = [-2, -1.0]

    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_b(goal_index=None):
    label = "testb" # _g" + str(goal_index)

    start           = [0.0, 0.0]

    goal3 = [-5.0, 10.0]
    goal1 = [5.0, 10.0]

    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 26
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_0(goal_index=None):
    label = "test0" # _g" + str(goal_index)

    start           = [1.0, 0.01]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    # goal1           = [0.0, 6.0]
    # goal3           = [2.0, 6.0]
    # goal1 = [0.0, 18.0]
    # goal3 = [2.0, 18.0]

    # goal1 = [0.0, 6.0]
    # goal3 = [2.0, 6.0]
    # goal2 = [4.0, 4.0]

    goal1 = [0.0, 6.0]
    goal3 = [2.0, 9.0]

    target_goal = goal1
    all_goals   = [goal1, goal3]

    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 31

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_1(goal_index=None):
    label = "test1_asym" # _g" + str(goal_index)

    start           = [0.0, 3.0]

    goal1           = [3.5, 4.0]
    goal2           = [3.0, 2.0]

    all_goals   = [goal2, goal1]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    # dt = .025
    N = 40

    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_2(goal_index=None):
    label = "test2_colin" # _g" + str(goal_index)

    start           = [0.0, 0.0]

    goal1           = [1.0, 0] #-0.001]
    goal2           = [1.5, 0] #-0.003]
    # goal3           = [2.0, 0] #-0.002]
    # goal4           = [2.5, 0] #-0.002]

    # goal1           = [0.0, 0] #-0.001]
    # goal2           = [0.0, 0] #-0.003]
    # goal3           = [0.0, 0] #-0.002]

    all_goals   = [goal1, goal2] #, goal3, goal4]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    # dt = .025
    N = 36
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_3(goal_index=None):
    label = "test3_colin" # _g" + str(goal_index)

    start           = [0.0, 0.0]

    goal1           = [1.0, 1.0]
    goal2           = [2.0, 2.0]
    # goal3           = [4.0, 0.0]

    all_goals   = [goal1, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    # dt = .0125
    N = 31
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_4(goal_index=None):
    label = "test4_needle" # _g" + str(goal_index)

    start           = [0.0, 2.0]

    goal1           = [2.0, 1.0]
    goal2           = [3.0, 4.0]
    # goal3           = [4.0, 0.0]

    all_goals   = [goal1, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    table_pts.append([1.0, 3.0])
    table_pts.append([2.5, 2.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 40
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_4_flip(goal_index=None):
    label = "test4_needle_flip" # 2_g" + str(goal_index)

    start           = [0.0, 2.0]

    goal1           = [2.0, 4.0]
    goal2           = [3.0, 1.0]
    # goal3           = [4.0, 0.0]

    all_goals   = [goal1, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    table_pts.append([1.0, 2.5])
    table_pts.append([2.5, 3.0])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 40
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp


def scenario_test_4_flip_inverse(goal_index=None):
    label = "test4_needle3_inv" # _g" + str(goal_index)

    start           = [-0.0, 2.0]

    goal1           = [-2.0, 4.0]
    goal2           = [-3.0, 1.0]
    # goal3           = [4.0, 0.0]

    all_goals   = [goal1, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    table_pts.append([-1.0, 2.5])
    table_pts.append([-2.5, 3.0])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 40
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_5(goal_index=None):
    label = "test5_blocked" # _g" + str(goal_index)

    start           = [0.0, 2.5]

    goal1           = [3.0, 1.0]
    goal2           = [3.0, 4.0]
    # goal3           = [4.0, 0.0]

    all_goals   = [goal1, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    table_pts.append([1.5, 3.0])
    # table_pts.append([2.5, 2.5])

    obs_pts = []
    # obs_pts.append([1.0, 0.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 31
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_test_6(goal_index=None):
    label = "test6_obs" # _g" + str(goal_index)

    start           = [0.0, 2.5]

    goal1           = [3.0, 1.0]
    goal2           = [3.0, 4.0]
    # goal3           = [4.0, 0.0]

    all_goals   = [goal1, goal2]
    if goal_index is not None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = all_goals[0]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    table_pts.append([0.0, 1.5])
    # table_pts.append([2.5, 2.5])

    obs_pts = []
    obs_pts.append([1.0, 1.5, 0])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = 31
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

##########################################

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
    label = "test0"
    
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

    if goal_index != None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = goal1

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

    N = 31
    exp.set_N(N)

    return label, exp

# def scenario_1(goal_index=None):
#     label = "scenario_0"

#     restaurant      = None
#     start           = [0.0, 0.0]

#     true_goal       = [8.0, 2.0]
#     goal2           = [2.0, 1.0]
#     goal3           = [4.0, 1.0]

#     goal1           = [4.0, 2.0]
#     goal3           = [1.0, 3.0]

#     target_goal = goal1

#     all_goals   = [goal1, goal3]

#     if goal_index != None:
#         target_goal = all_goals[goal_index]
#     else:
#         target_goal = goal1


#     exp = ex.PathingExperiment(label, start, target_goal, all_goals)

#     exp.set_state_size(2)
#     exp.set_action_size(2)

#     dt = .025
#     N = 21
#     Q = 1.0 * np.eye(exp.get_state_size())
#     R = 200.0 * np.eye(exp.get_action_size())
#     Qf = np.identity(2) * 400.0

#     exp.set_QR_weights(Q, R, Qf)
#     exp.set_N(N)
#     exp.set_dt(dt)

#     return label, exp


def scenario_test_8(goal_index=None):
    label = "test_8" # _g" + str(goal_index)

    restaurant      = None
    start           = [0.0, 0.0]

    # true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    # goal3           = [1.0, 3.0]

    target_goal = goal4
    all_goals   = [goal4, goal1, goal2]

    if goal_index != None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = goal4

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

    N = 21 #42 #21
    exp.set_N(N)

    return label, exp

def scenario_test_9(goal_index=None):
    label = "test_9" # _g" + str(goal_index)

    restaurant      = None
    start           = [8.0, 2.0]

    # true_goal       = [0.0, 0.0]
    goal2           = [2.0, 1.0]
    # goal3           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal3
    # start = goal3

    all_goals   = [goal1, goal3, goal2]

    if goal_index != None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = goal3

    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

    N = 31
    exp.set_N(N)

    return label, exp

def scenario_10_large_scale(goal_index=None):
    label = "test_10_large_scale" # _g" + str(goal_index)

    restaurant      = None
    start           = [800.0, 200.0]

    true_goal       = [0.0, 0.0]
    goal2           = [200.0, 100.0]
    goal3           = [400.0, 100.0]

    goal1           = [400.0, 200.0]
    goal3           = [100.0, 300.0]

    target_goal = goal3
    all_goals   = [goal1, goal3, goal2]

    if goal_index != None:
        target_goal = all_goals[goal_index]
    else:
        target_goal = goal3
    
    exp = ex.PathingExperiment(label, start, target_goal, all_goals)

    N = 21
    exp.set_N(N)

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

    N = int(41)
    exp.set_N(N)

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

    N = 31
    exp.set_N(N)

    obs_scale = 10000.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_7_observer(goal_index=None, obs_angle=0):
    label = "test_7_obs" # _g" + str(goal_index)

    start           = [0.0, 2.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 1.0]
    goal3           = [4.0, 3.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal1
    else:
        target_goal = all_goals[goal_index]

    # center points of tables, circular in iLQR world
    # radius needs to be agreed upon between this definition and the Obstacle class
    table_pts = []
    # table_pts.append([1.0, 1.0])
    # table_pts.append([3.0, 0.5])

    obs_pts = []
    obs_pts.append([2.5, 1.0, obs_angle])
    # obs_pts.append([4.5, 1.0, 180])
    # obs_pts.append([4.0, 0.5, 90])

    exp = ex.PathingExperiment(label, start, target_goal, all_goals, observers=obs_pts, table_pts=table_pts)

    N = int(21 * 2)
    exp.set_N(N)

    obs_scale = 0.0
    exp.set_solver_scale_obstacle(obs_scale)

    return label, exp

def scenario_7_observer_offset(goal_index=None):
    label = "scenario_7_offset" # _g" + str(goal_index)

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

    N = int(21 * 2)
    exp.set_N(N)

    return label, exp

def scenario_7_observer_on_zero(goal_index=None, obs_angle=0):
    label = "test_7_obs_on_zero"

    start           = [0.0, 0.0]

    goal1           = [2.0, -1.0]
    goal3           = [2.0, 1.0]

    all_goals   = [goal1, goal3]

    if goal_index is None or goal_index > len(all_goals):
        target_goal = goal1
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

    N = int(31)
    exp.set_N(N)

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

    N = int(21 * 2)
    exp.set_N(N)

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

    N = int(21 * 2)
    exp.set_N(N)

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

    N = 21
    exp.set_N(N)

    return label, exp

def get_scenario_set(scenario_filters=[]):
    scenarios = {}

    # TEST SCENARIO
    label, exp = scenario_pent(goal_index=0)
    scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_fan(goal_index=0)
    scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_pent_wide(goal_index=0)
    scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_pent_asymm0(goal_index=0)
    # scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_pent_asymm1(goal_index=0)
    # scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_pent_asymm2(goal_index=0)
    # scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_pent_asymm3(goal_index=0)
    # scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_pent_asymm4(goal_index=0)
    # scenarios[label] = exp

    ############
    # TEST SCENARIO
    label, exp = scenario_triangle_thin(goal_index=0)
    scenarios[label] = exp


    # # TEST SCENARIO
    # label, exp = scenario_diamond(goal_index=0)
    # scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_triangle_equid(goal_index=0)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_aimakerspace(goal_index=0)
    scenarios[label] = exp

    return scenarios

    # TEST SCENARIO
    label, exp = scenario_parallelogram(goal_index=0)
    scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_equidist(goal_index=0)
    scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_equidist2(goal_index=0)
    # scenarios[label] = exp
    return scenarios

    # TEST SCENARIO
    label, exp = scenario_equidist3(goal_index=0)
    scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_aimakerspace_long(goal_index=0)
    scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_aimakerspace_no_obs(goal_index=0)
    # scenarios[label] = exp

    return scenarios

    # TEST SCENARIO
    label, exp = scenario_test_a(goal_index=1)
    scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_test_b(goal_index=0)
    scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_test_c(goal_index=0)
    scenarios[label] = exp

    return scenarios
    
    # # TEST SCENARIO
    # label, exp = scenario_test_b(goal_index=1)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_test_0(goal_index=0)
    scenarios[label] = exp

    # # TEST SCENARIO
    # label, exp = scenario_test_0(goal_index=1)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_test_1(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_1(goal_index=1)
    # scenarios[label] = exp


    # NOTE this doesn't guarantee that scenario 0 has obstacles, heading and all else
    # May want to verify there's an option on the list
    if scenario_filters[SCENARIO_FILTER_MINI]:
        return scenarios

    # TEST SCENARIO
    label, exp = scenario_test_4_flip(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_4_flip(goal_index=1)
    # scenarios[label] = exp

    # # TEST SCENARIO
    label, exp = scenario_test_4_flip_inverse(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_4_flip_inverse(goal_index=1)
    # scenarios[label] = exp

    # NOTE this doesn't guarantee that scenario 0 has obstacles, heading and all else
    # May want to verify there's an option on the list
    if scenario_filters[SCENARIO_FILTER_MINI]:
        return scenarios

    # TEST SCENARIO
    label, exp = scenario_test_2(goal_index=0)
    scenarios[label] = exp

    # return scenarios

    # label, exp = scenario_test_2(goal_index=1)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_test_3(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_3(goal_index=1)
    scenarios[label] = exp

    ####### NEEDLE
    # # TEST SCENARIO
    label, exp = scenario_test_4(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_4(goal_index=1)
    # scenarios[label] = exp

    # # TEST SCENARIO
    label, exp = scenario_test_4_flip(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_4_flip(goal_index=1)
    # scenarios[label] = exp

    # return scenarios

    # # TEST SCENARIO
    label, exp = scenario_test_5(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_5(goal_index=1)
    # scenarios[label] = exp

    # # TEST SCENARIO
    label, exp = scenario_test_6(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_6(goal_index=1)
    # scenarios[label] = exp

    # label, exp = scenario_4_has_obstacles_and_observer(goal_index=None)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_7_observer_on_zero(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_7_observer_on_zero(goal_index=1)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_7_observer_rot90(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_7_observer_rot90(goal_index=1)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_7_observer_offset(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_7_observer_offset(goal_index=1)

    label, exp = scenario_test_8(goal_index=0)
    scenarios[label] = exp
    
    # label, exp = scenario_test_8(goal_index=1)
    # scenarios[label] = exp

    # label, exp = scenario_test_8(goal_index=2)
    # scenarios[label] = exp

    # TEST SCENARIO
    label, exp = scenario_test_9(goal_index=0)
    scenarios[label] = exp

    # label, exp = scenario_test_9(goal_index=1)
    # scenarios[label] = exp

    # label, exp = scenario_test_9(goal_index=2)
    # scenarios[label] = exp

    return scenarios


def get_scenarios(scenario_filters=[]):
    options = get_scenario_set(scenario_filters)

    return options


# All scenarios can care about heading
def get_scenarios_heading(scenario_filters=[]):
    scenarios = get_scenario_set(scenario_filters)

    return scenarios

# only a subset care about obstacles
def get_scenarios_obstacles(scenario_filters=[]):
    all_scenarios = get_scenarios(scenario_filters)
    new_list = {}

    for key in all_scenarios.keys():
        scenario = all_scenarios[key]

        if len(scenario.get_tables()) > 0:
            new_list[key] = scenario

        if scenario_filters[SCENARIO_FILTER_MINI] is True and len(new_list.keys()) > 0:
            return new_list

    return new_list

def get_scenarios_observers(scenario_filters=[]):
    all_scenarios = get_scenarios(scenario_filters)
    new_list = {}

    for key in all_scenarios.keys():
        scenario = all_scenarios[key]

        if len(scenario.get_observers()) > 0:
            new_list[key] = scenario

        if scenario_filters[SCENARIO_FILTER_MINI] is True  and len(new_list.keys()) > 0:
            return new_list

    return new_list












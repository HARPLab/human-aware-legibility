import sys
import os
import numpy as np

from datetime import timedelta, datetime

import utility_environ_descrip as resto

from LegiblePathQRCost import LegiblePathQRCost
from DirectPathQRCost import DirectPathQRCost
from ObstaclePathQRCost import ObstaclePathQRCost
from LegibilityOGPathQRCost import LegibilityOGPathQRCost
from OALegiblePathQRCost import OALegiblePathQRCost

PREFIX_EXPORT = 'experiment_outputs/'

# OPTIONS OF F FUNCTION
F_NONE          = 'f_none'
F_ANCA_LINEAR   = 'f_anca_linear'
F_VIS           = 'f_vis'

# OPTIONS FOR COST/SOLVER TYPE
COST_DIRECT     = 'cost_direct'
COST_LEGIB      = 'cost_legible'
COST_OALEGIB    = 'cost_oalegib'
COST_OBS        = 'cost_obstacles'

# # SOLVER TYPE
# SOLVER_LEGIB        = 'Solver=Legible'
# SOLVER_OBSTACLES    = 'Solver=Obstacles'
# SOLVER_OA           = 'Solver=OALegible'
# SOLVER_DIRECT       = 'Solver=Direct'

class PathingExperiment():

    # default values for solver
    solver_coeff_terminal   = 1000000.0
    solver_scale_term       = 0.01
    solver_scale_stage      = 2
    solver_scale_obstacle   = 0

    # DEFAULT COST TYPE AND F TYPE
    cost_label  = COST_LEGIB
    f_label     = F_NONE

    def __init__(self, restaurant, f_label=None, cost_label=None):
        self.restaurant = restaurant

        self.start          = restaurant.get_start()
        self.target_goal    = restaurant.get_goals_all()
        self.goals          = restaurant.get_goals()

        self.observers      = restaurant.get_observers()
        self.tables         = restaurant.get_tables()

        self.f_label    = cost_label
        self.cost_label = f_label

        self.setup_file_id()


    def __init__(self, start, target_goal, all_goals, restaurant=None, observers=[], table_pts=[], f_label=None, cost_label=None):
        self.start = start
        self.target_goal = target_goal
        self.goals = all_goals

        if restaurant is None:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=table_pts, goals=all_goals, start=start, obs_pts=observers, dim=None)
        self.restaurant = restaurant

        self.f_label    = cost_label
        self.cost_label = f_label

        self.observers = observers
        self.table_pts = table_pts

        self.setup_file_id()

    def setup_cost(self, Xrefline, Urefline):
        solver_label = self.cost_label

        if solver_label is COST_LEGIB:
            return LegiblePathQRCost(self, Xrefline, Urefline)
        elif solver_label is COST_OBS:
            return ObstaclePathQRCost(self, Xrefline, Urefline)
        elif solver_label is COST_OA:
            return OALegiblePathQRCost(self, Xrefline, Urefline)
        elif solver_label is COST_DIRECT:
            return DirectPathQRCost(self, Xrefline, Urefline)

        print("ERROR, NO KNOWN SOLVER, PLEASE ADD A VALID SOLVER TO EXP")

    def setup_file_id(self):
        # Create a new folder for this experiment, along with sending debug output there
        self.file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        os.mkdir(PREFIX_EXPORT + self.file_id)
        sys.stdout = open(PREFIX_EXPORT + self.file_id + '/output.txt','wt')

    def set_f_label(self, label):
        self.f_label = label

    def get_f_label(self):
        return self.f_label

    def set_cost_label(self, label):
        self.cost_label = label

    def get_cost_label(self):
        return self.cost_label

    def set_state_size(self, label):
        self.state_size = label

    def get_state_size(self):
        return self.state_size

    def set_action_size(self, label):
        self.action_size = label

    def get_action_size(self):
        return self.action_size

    def set_observers(self, x):
        self.observers = x

    def get_observers(self):
        return self.observers

    def get_restaurant(self):
        return self.restaurant

    def get_start(self):
        return self.start

    def get_goals(self):
        return self.goals

    def get_target_goal(self):
        return self.target_goal

    def set_QR_weights(self, Q, R, Qf):
        self.Q  = Q
        self.R  = R
        self.Qf = Qf

    def get_Q(self):
        return self.Q

    def get_R(self):
        return self.R

    def get_Qf(self):
        return self.Qf

    def set_N(self, N):
        self.N = N

    def get_N(self):
        return self.N

    def set_dt(self, dt):
        self.dt = dt

    def get_dt(self):
        return self.dt

    def get_file_id(self):
        return self.file_id

    def set_solver_scale_obstacle(self, scale):
        self.solver_scale_obstacle = scale

    def get_solver_scale_obstacle(self):
        return self.solver_scale_obstacle

    def get_experiment_dict(self):
        return {}



import sys
import os
import numpy as np

from datetime import timedelta, datetime

import utility_environ_descrip as resto

PREFIX_EXPORT = 'experiment_outputs/'

# OPTIONS OF F FUNCTION
F_NONE          = 'f_none'
F_ANCA_LINEAR   = 'f_anca_linear'
F_VIS           = 'f_vis'

# OPTIONS FOR COST/SOLVER TYPE
COST_DIRECT     = 'cost_direct'
COST_LEGIB      = 'cost_legible'
COST_OALEGIB    = 'cost_oalegib'
COST_DIRECT     = 'cost_obstacles'

class PathingExperiment():

    # default values for solver
    solver_coeff_terminal   = 1000000.0
    solver_scale_term       = 0.01
    solver_scale_stage      = 2
    solver_scale_obstacle   = 0

    def __init__(self, restaurant, f_label=None, cost_label=None):
        self.restaurant = restaurant

        self.start          = restaurant.get_start()
        self.target_goal    = restaurant.get_goals_all()
        self.goals      = restaurant.get_goals()

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
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=table_pts, goals=all_goals, start=start, observers=observers, dim=None)
        self.restaurant = restaurant

        self.f_label    = cost_label
        self.cost_label = f_label

        self.observers = observers
        self.table_pts = table_pts

        self.setup_file_id()

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

    def set_observers(self, x):
        self.observers = x

    def get_observers(self):
        return self.observers

    def get_restaurant(self):
        return self.restaurant

    def get_start(self):
        return self.start

    def get_all_goals(self):
        return self.goals

    def get_target_goal(self):
        return self.target_goal

    def get_file_id(self):
        return self.file_id

    def get_experiment_dict(self):
        return {}



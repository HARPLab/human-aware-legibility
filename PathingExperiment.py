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
from OAObsPathQRCost import OAObsPathQRCost
from SocLegPathQRCost import SocLegPathQRCost

PREFIX_EXPORT = 'experiment_outputs/'

# OPTIONS OF F FUNCTION
F_NONE          = 'f_none'
F_ANCA_LINEAR   = 'f_anca_linear'
F_VIS_LIN           = 'f_vis_lin'
F_VIS_BIN           = 'f_vis_bin'

# OPTIONS FOR COST/SOLVER TYPE
COST_DIRECT     = 'cost_direct'
COST_LEGIB      = 'cost_legible'
COST_OA         = 'cost_oalegib'
COST_OBS        = 'cost_obstacles'
COST_OA_AND_OBS = 'cost_oa_and_obs'

# # SOLVER TYPE
# SOLVER_LEGIB        = 'Solver=Legible'
# SOLVER_OBSTACLES    = 'Solver=Obstacles'
# SOLVER_OA           = 'Solver=OALegible'
# SOLVER_DIRECT       = 'Solver=Direct'

class PathingExperiment():
    label = "needslabel"
    ax = None

    # default values for solver
    solver_coeff_terminal   = 1000000.0
    solver_scale_term       = 1 #.01
    solver_scale_stage      = 2
    solver_scale_obstacle   = 0

    lambda_cost_path_coeff  = 1.0

    TABLE_RADIUS            = .25
    OBSERVER_RADIUS         = .1
    GOAL_RADIUS             = .15
    OBSTACLE_BUFFER         = .25 #05


    # DEFAULT COST TYPE AND F TYPE
    cost_label  = COST_OA_AND_OBS
    f_label     = F_VIS_BIN

    state_size  = 2
    action_size = 2

    dt  = .025
    N   = int(21 * 2)
    Qf  = np.identity(2) * 400.0
    Q   = 1.0 * np.eye(state_size)
    R   = 200.0 * np.eye(action_size)

    oa_on                   = True
    heading_on              = True
    norm_on                 = False
    weighted_close_on       = False
    mode_pure_heading       = False
    mode_heading_err_sqr    = False
    mode_dist_type          = 'exp'

    J_hist = []
    best_xs = None
    best_us = None
    solve_status = None

    fn_note = ""
    run_filters = []
    mode_dist_legib_on = True

    def __init__(self, label, restaurant, f_label=None, cost_label=None):
        self.exp_label = label
        self.restaurant = restaurant

        self.start          = restaurant.get_start()
        self.target_goal    = restaurant.get_goals_all()
        self.goals          = restaurant.get_goals()

        self.observers      = restaurant.get_observers()
        self.tables         = restaurant.get_tables()

        if f_label is not None:
            self.f_label    = f_label
    
        if cost_label is not None:
            self.cost_label = cost_label

        self.setup_file_id()


    def __init__(self, label, start, target_goal, all_goals, restaurant=None, observers=[], table_pts=[], f_label=None, cost_label=None):
        self.exp_label = label

        self.start = start
        self.target_goal = target_goal
        self.goals = all_goals

        if restaurant is None:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=table_pts, goals=all_goals, start=start, obs_pts=observers, dim=None)
        self.restaurant = restaurant

        if f_label is not None:
            self.f_label    = f_label
    
        if cost_label is not None:
            self.cost_label = cost_label

        self.observers      = restaurant.get_observers()
        self.table_pts = table_pts

        self.setup_file_id()

    def on_iteration_exp(self, iteration_count, xs, us, J_opt, accepted, converged):
        self.J_hist.append(J_opt)
        info = "converged" if converged else ("accepted" if accepted else "failed")

        final_state = xs[-1]
        print("iteration", iteration_count, info, J_opt, final_state)
        
        if converged:
            self.best_xs = xs
            self.best_us = us

        most_recent_is_complete_packet = [converged, info, iteration_count]

        print("FINAL PATH IS " + str(xs))
        self.solve_status = most_recent_is_complete_packet

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
        elif solver_label is COST_OA_AND_OBS:
            return SocLegPathQRCost(self, Xrefline, Urefline)


        print("ERROR, NO KNOWN SOLVER, PLEASE ADD A VALID SOLVER TO EXP")
        print("''''''" + str(solver_label) + "''''''")

    def setup_file_id(self):
        # Create a new folder for this experiment, along with sending debug output there
        self.file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + self.exp_label

        # try:
        #     os.mkdir(PREFIX_EXPORT + self.file_id)
        # except:
        #     print("FILE ALREADY EXISTS " + self.file_id)
        
        # sys.stdout = open(PREFIX_EXPORT + self.file_id + '/output.txt','wt')

    def reinit_file_id(self):
        self.setup_file_id()

    def set_f_label(self, label):
        self.f_label = label

    def get_f_label(self):
        return self.f_label

    def get_exp_label(self):
        return self.exp_label

    def set_exp_label(self, label):
        goal_index = str(self.goals.index(self.target_goal))

        self.exp_label = label + "-g" + goal_index

    def get_goal_label(self):
        return "g" + str(self.goals.index(self.target_goal))

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
        if type(x[0]) is resto.Observer:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=self.table_pts, goals=self.goals, start=self.start, observers=x, dim=None)
        else:
            restaurant = resto.Restaurant(resto.TYPE_CUSTOM, table_pts=self.table_pts, goals=self.goals, start=self.start, obs_pts=x, dim=None)
        
        self.observers      = restaurant.get_observers()
        self.restaurant = restaurant

    def get_observers(self):
        return self.observers

    def set_tables(self, x):
        self.tables = x

    def get_tables(self):
        return self.restaurant.get_tables()

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

    def set_solver_scale_stage(self, scale):
        self.solver_scale_stage = scale

    def get_solver_scale_stage(self):
        return self.solver_scale_stage

    def set_solver_scale_term(self, scale):
        self.solver_scale_term = scale

    def get_solver_scale_term(self):
        return self.solver_scale_term

    def set_solver_coeff_terminal(self, scale):
        self.solver_coeff_terminal = scale

    def get_solver_coeff_terminal(self):
        return self.solver_coeff_terminal

    def get_experiment_dict(self):
        return {}

    def get_table_radius(self):
        return self.TABLE_RADIUS

    def set_table_radius(self, trad):
        self.obstacle_table_radius = trad

    def get_observer_radius(self):
        return self.OBSERVER_RADIUS

    def get_goal_radius(self):
        return self.GOAL_RADIUS

    def get_obstacle_buffer(self):
        return self.OBSTACLE_BUFFER

    def get_lambda_cost_path_coeff(self):
        return self.lambda_cost_path_coeff

    def set_lambda_cost_path_coeff(self, l):
        self.lambda_cost_path_coeff = l

    def set_heading_on(self, v):
        self.heading_on = v

    def get_is_heading_on(self):
        return self.heading_on

    def set_oa_on(self, v):
        self.oa_on = v

    def get_is_oa_on(self):
        return self.oa_on

    def set_ax(self, v):
        self.ax = v

    def get_ax(self):
        return self.ax

    def set_norm_on(self, v):
        self.norm_on = v

    def get_norm_on(self):
        return self.norm_on

    def set_mode_dist_legib_on(self, v):
        self.mode_dist_legib_on = v

    def get_mode_dist_legib_on(self):
        return self.mode_dist_legib_on

    def set_weighted_close_on(self, v):
        self.weighted_close_on = v

    def get_weighted_close_on(self):
        return self.weighted_close_on

    def set_mode_pure_heading(self, v):
        self.mode_pure_heading = v

    def get_mode_pure_heading(self):
        return self.mode_pure_heading

    def set_mode_heading_err_sqr(self, v):
        self.mode_heading_err_sqr = v

    def get_mode_heading_err_sqr(self):
        return self.mode_heading_err_sqr

    def set_mode_dist_type(self, v):
        self.mode_dist_type = v

    def get_mode_dist_type(self):
        return self.mode_dist_type

    def set_fn_note(self, label):
        self.fn_note = "-" + label

    def get_fn_note(self):
        return self.fn_note

    def set_run_filters(self, v):
        self.run_filters = v

    def get_run_filters(self):
        return self.run_filters

    def get_solver_status(self):
        return self.solve_status

    def get_solver_status_blurb(self, info_packet=None):
        if info_packet is None:
            info_packet = self.get_solver_status()

        # info_packet = [converged, info, iteration_count]
        if info_packet is not None:
            converged_text = ""
            converged, info, iteration_count = info_packet
            if converged is False:
                converged_text = "INCOMPLETE after " + str(iteration_count)
            else:
                converged_text = "CONVERGED in " + str(iteration_count)
            
            blurb = converged_text
        else:
            blurb = ""

        return blurb

    def get_solve_quality_status(self, test_group, info_packet=None):
        if info_packet is None:
            info_packet = self.get_solver_status()

        # info_packet = [converged, info, iteration_count]
        if info_packet is not None:
            converged_text = ""
            converged, info, iteration_count = info_packet
            if converged is False:
                converged_text = "INC in " + str(iteration_count)
            else:
                converged_text = "OK in " + str(iteration_count)
            
            blurb = converged_text
        else:
            blurb = ""

        scenario    = self.get_exp_label()
        purpose     = self.get_fn_note()[1:]

        return (scenario, self.get_goal_label(), test_group, purpose, converged_text, converged, iteration_count)

    def get_solve_quality_columns(self):
        return ('scenario', 'goal', 'test', 'condition', 'status_summary', 'converged', 'num_iterations')

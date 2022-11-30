import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import decimal
import copy

from ilqr import iLQR
from ilqr.cost import Cost
from ilqr.cost import QRCost
from ilqr.cost import PathQRCost, AutoDiffCost, FiniteDiffCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics
from ilqr.dynamics import BatchAutoDiffDynamics, tensor_constrain
from LegiblePathQRCost import LegiblePathQRCost

import PathingExperiment as ex

from scipy.optimize import approx_fprime

import utility_legibility as legib
import utility_environ_descrip as resto
import pipeline_generate_paths as pipeline



class ObstaclePathQRCost(LegiblePathQRCost):
    FLAG_DEBUG_J = True

    """Quadratic Regulator Instantaneous Cost for trajectory following."""
    def __init__(self, exp, x_path, u_path):
        self.make_self(
            exp,
            exp.get_Q(),
            exp.get_R(),
            exp.get_Qf(),
            x_path,
            u_path,
            exp.get_start(),
            exp.get_target_goal(),
            exp.get_goals(),
            exp.get_N(),
            exp.get_dt(),
            restaurant=exp.get_restaurant(),
            file_id=exp.get_file_id()
            )

    def make_self(self, exp, Q, R, Qf, x_path, u_path, start, target_goal, goals, N, dt, restaurant=None, file_id=None, Q_terminal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """

        LegiblePathQRCost.__init__(
            self, exp, x_path, u_path
        )

    # original version for plain path following
    def l(self, x, u, i, terminal=False, just_term=False, just_stage=False):
        """Instantaneous cost function.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Qf if terminal else self.Q
        R = self.R
        start = self.start
        goal = self.target_goal
        thresh = .0001

        scale_term  = self.exp.get_solver_scale_term() #0.01 # 1/100
        scale_stage = self.exp.get_solver_scale_stage() #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0


        term_cost = 0 #self.term_cost(x, i)
        x_diff = x - self.x_path[i]

        u_diff = 0
        if i < len(self.u_path):
            u_diff = u - self.u_path[i]


        if terminal or just_term: #abs(i - self.N) < thresh or
            # TODO verify not a magic number
            return scale_term * self.term_cost(x, i) # * 1000
        else:
            if self.FLAG_DEBUG_STAGE_AND_TERM:
                # difference between this step and the end
                print("x, N, x_end_of_path -> inputs and then term cost")
                print(x, self.N, self.x_path[self.N])
                # term_cost = self.term_cost(x, i)
                print(term_cost)

        term_cost = 0 #self.term_cost(x, i)

        f_func     = self.get_f()
        f_value    = f_func(i)
        J = self.michelle_stage_cost(start, goal, x, u, i, terminal) * f_value

        wt_legib     = -1.0 * 5
        wt_lam       = .00001
        wt_control   = 3.0 * 200.0
        wt_obstacle = self.exp.get_solver_scale_obstacle()

        tables = self.restaurant.get_tables()
        obstacle_penalty = 0

        TABLE_RADIUS = self.exp.get_table_radius()
        self.scale_obstacle = self.exp.get_solver_scale_obstacle()
        for table in tables:
            obstacle = table.get_center()
            obs_dist = obstacle - x
            obs_dist = np.linalg.norm(obs_dist)
            # Flip so edges lower cost than center
            obs_dist = TABLE_RADIUS - obs_dist

            if obs_dist < TABLE_RADIUS:
                print("obstacle dist for " + str(x) + " " + str(obs_dist))
                # obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                # OBSTACLE PENALTY NOW ALWAYS SCALED TO RANGE 0 -> 1
                obstacle_penalty += (obs_dist / TABLE_RADIUS)**2

                # np.inf #

        if obstacle_penalty > 0:
            print("total obstacle penalty " + str(obstacle_penalty))
    
        J =  (wt_legib      * J)
        J += (wt_control    * u_diff.T.dot(R).dot(u_diff))
        J += (wt_lam        * x_diff.T.dot(Q).dot(x_diff))
        J += (wt_obstacle)  * obstacle_penalty
    
        stage_costs = J

        # term_cost += obstacle_penalty
        # stage_costs += obstacle_penalty
        # term_cost += obstacle_penalty

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

        total = (scale_term * term_cost) + (scale_stage * stage_costs)

        return float(total)





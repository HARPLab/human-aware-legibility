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
    scale_obstacle = 100000.0

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
        thresh = 1


        if terminal or abs(i - self.N) < thresh:
            return self.term_cost(x, i) #*1000
        else:
            if self.FLAG_DEBUG_STAGE_AND_TERM:
                # difference between this step and the end
                print("x, N, x_end_of_path -> inputs and then term cost")
                print(x, self.N, self.x_path[self.N])
                # term_cost = self.term_cost(x, i)
                print(term_cost)

        term_cost = 0
        
        f_func     = self.get_f()
        f_value    = f_func(i)
        stage_costs = self.michelle_stage_cost(start, goal, x, u, i, terminal) * f_value
    
        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

        # Log of remixes of term and stage cost weightings
        # term_cost      = decimal.Decimal.ln(decimal.Decimal(term_cost)) 
        # stage_costs    = decimal.Decimal.ln(stage_costs)
        # if i < 30:
        #     stage_scale = 200
        #     term_scale = 0.1
        # else:
        #     stage_scale = 10
        #     term_scale = 1
        # stage_scale = max([(self.N - i), 20])
        # term_scale = 100
        # stage_scale = 10
        # term_scale = 1
        # stage_scale = max([self.N-i, 10])
        # stage_scale = abs(self.N-i)
        # term_scale = i/self.N
        # term_scale = 1
        # stage_scale = 50

        scale_term = self.scale_term #0.01 # 1/100
        scale_stage = self.scale_stage #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0

        tables = self.restaurant.get_tables()
        obstacle_penalty = 0

        TABLE_RADIUS = .5
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

                obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                # np.inf #


        if obstacle_penalty > 0:
            print("total obstacle penalty " + str(obstacle_penalty))
    
        term_cost += obstacle_penalty

        total = (scale_term * term_cost) + (scale_stage * stage_costs)

        return float(total)





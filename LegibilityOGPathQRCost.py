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

from scipy.optimize import approx_fprime

import utility_legibility as legib
import utility_environ_descrip as resto
import pipeline_generate_paths as pipeline



class LegibilityOGPathQRCost(LegiblePathQRCost):
    FLAG_DEBUG_J = True

    """Quadratic Regulator Instantaneous Cost for trajectory following."""
    def __init__(self, Q, R, x_path, u_path, start, target_goal, goals, N, dt, Q_terminal=None):
        """Constructs a QRCost.
        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """

        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x_path = np.array(x_path)

        self.start = np.array(start)
        self.goals = goals
        self.target_goal = target_goal
        self.N = N
        self.dt = dt

        # Create a restaurant object for using those utilities, functions, and print functions
        # dim gives the dimensions of the restaurant
        self.restaurant = resto.Restaurant(resto.TYPE_CUSTOM, tables=[], goals=goals, start=start, observers=[], dim=None)

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

        x_eps = .01 #05
        u_eps = .01 #05

        # self._x_eps_hess = np.sqrt(self._x_eps)
        # self._u_eps_hess = np.sqrt(self._u_eps)

        self._state_size = state_size
        self._action_size = action_size

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if u_path is None:
            self.u_path = np.zeros(path_length - 1, action_size)
        else:
            self.u_path = np.array(u_path)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + 1, \
                "x_path must be 1 longer than u_path"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        LegiblePathQRCost.__init__(
            self,
            Q, R, x_path, u_path, start, target_goal, goals, N, dt, Q_terminal=Q_terminal
        )



    # original version for plain path following
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        resto_envir = self.restaurant
        goals = self.goals
        exp_settings = pipeline.get_default_exp_settings(unique_key="ilqr_verif")

        # This is a value from 0 to 1
        l = legib.f_legibility(resto_envir, self.target_goal, self.goals, self.x_path[:i], [])
        legibility_scalar = -1000.0
        l = float(l) * legibility_scalar

        if terminal:
            return squared_x_cost + l

        u_diff = u - self.u_path[i]
        u_diff_cost = u_diff.T.dot(R).dot(u_diff)

        J = squared_x_cost + u_diff_cost
    
        total_J = J + l

        print("parts of J: J = l + x + u")
        print(total_J, l, squared_x_cost, u_diff_cost)

        return total_J

    # def get_total_stage_cost(self, start, goal, x, u, i, terminal):
    #     N = self.N
    #     R = self.R

    #     stage_costs = 0.0
        
    #     # print("u = " + str(u))
    #     # print("Getting stage cost")
    #     for j in range(i):
    #         u_diff = u - self.u_path[j]
    #         x_diff = x - self.x_path[j]

    #         # print("at " + str(j) + "u_diff = " + str(u_diff))
    #         # print(u_diff.T.dot(R).dot(u_diff))

    #         # stage_costs += self.michelle_stage_cost(start, goal, x, u, j, terminal)

    #         stage_costs += u_diff.T.dot(R).dot(u_diff)
    #         stage_costs += x_diff.T.dot(R).dot(x_diff)


    #         # stage_cost(x, u, j, terminal) #
    #         # stage_costs = stage_costs + self.goal_efficiency_through_point_relative(start, goal, x, terminal)

    #     print("total stage cost " + str(stage_costs))
    #     return stage_costs


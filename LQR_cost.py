import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.cost import Cost
from ilqr.cost import QRCost
from ilqr.cost import PathQRCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics
from ilqr.dynamics import BatchAutoDiffDynamics, tensor_constrain


class LegiblePathQRCost(PathQRCost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Q, R, x_path, u_path, target_goal, goals, Q_terminal=None):
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

        self.goals = goals
        self.target_goal = target_goal

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

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

        super(PathQRCost, self).__init__()

    # TODO ada why does this always come out to be 0???
    def term_cost(self, x, i):
        start = self.x_path[0]
        goal1 = self.target_goal


        # print("start, goal1")
        # print(start)
        # print(goal1)

        Qf = np.identity(2) # * 10
        R = self.R
        all_goals = self.goals

        if i == -1:
            xref = goal1
        else:
            xref = self.x_path[i]

        # TODO Check whether this was evil or not
        x_diff = (x - xref)
        terminal_cost = 0.5*((x_diff).dot(Qf).dot((x_diff).T))

        # print("x_diff")
        # print(x_diff)
        terminal_coeff = 1.0
        terminal_cost = terminal_cost * terminal_coeff

        # Once we're at the goal, the terminal cost is 0
        return terminal_cost

    # original version for plain path following
    def l_og(self, x, u, i, terminal=False):
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

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    # stage cost
    def l(self, x, u, i, terminal=False):
        # return self.l_og(x, u, i, terminal)

        # def trajectory_cost(X, U, Xref, Uref, start, goal1, all_goals, nongoal_scale):
        # calculate the cost of a given trajectory 
        # N_len = len(Xref)
        Xref = self.x_path
        Uref = self.u_path

        end_of_path = self.x_path[-1]
        # end_goal    = Xref[N_len]

        # start with the term cost
        term_cost = self.term_cost(end_of_path, -1)



        # J = term_cost(X[N_len],Xref[N_len])
        N = Uref.shape[0]

        print("Uref shape")
        print(N)
        stage_costs = 0

        # currently has a value of about 10 * N steps
        for i in range(N):
            stage_costs = stage_costs + self.stage_cost(x, u, i, terminal=terminal)

        J = term_cost + stage_costs

        print("Total J for stage: term, stage, J")
        print(term_cost, stage_costs, J)

        return J


    def stage_cost(self, x, u, i, terminal=False):
        """Instantaneous cost function.
        #     Args:
        #         x: Current state [state_size].
        #         u: Current control [action_size]. None if terminal.
        #         i: Current time step.
        #         terminal: Compute terminal cost. Default: False.
        #     Returns:
        #         Instantaneous cost (scalar).
        #     """

         # NOTE: The terminal cost needs to at most be a function of x and i, whereas
         #  the non-terminal cost can be a function of x, u and i.

        nongoal_scale = 1 #50

        start = self.x_path[0]
        goal1 = self.target_goal

        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        all_goals = self.goals

        if terminal: #ie, u is not None:
        # TERMINAL COST FUNCTION
            return self.term_cost(x, -1)

        xref = self.x_path[i]
        uref = self.u_path[i]
        
        x_diff = x - xref
        u_diff = u - uref
        
        # Find the cost of going to the targeted goal
        # STAGE COST FUNCTION
        goal_diff   = start - goal1
        start_diff  = (start - x)
        togoal_diff = (x - goal1)

        # goal_diff   = np.reshape(goal_diff.T, (-1, 2))
        # start_diff  = np.reshape(start_diff.T, (-1, 2))
        # togoal_diff = np.reshape(togoal_diff.T, (-1, 2))

        a = (goal_diff).dot(Q).dot((goal_diff).T)
        b = (start_diff).dot(Q).dot((start_diff).T)
        c = (togoal_diff).dot(Q).dot((togoal_diff).T)
    


        J_g1 = a - b + - c
        print("Value for goal before others: " + str(J_g1))

        # J_g1 = (np.exp(a - b)/np.exp(c))

        # J_g1 *= 0.5

        # print("J_g1 = legibility for goal 1")
        # print(J_g1)

        # and then also find this ratio for all of the other goals and combine
        print("Finding ratio")
        all_components = []
        goal_component = 0

        log_sum = 0
        for i in range(len(all_goals)):
            goal = all_goals[i]
            # print("goal = " + str(goal))

            scale = 1
            if goal != goal1:
                scale = nongoal_scale

            alt_goal_diff               = (x - goal)
            alt_goal_from_start_diff    = (start - goal)

            n0 = (start_diff).dot(Q).dot((start_diff).T)
            n1 = (alt_goal_diff).dot(Q).dot((alt_goal_diff).T)


            # weight_before = 5
            # weight_after = 10
            weight_before = 0.0
            weight_after = 0.0

            n = - (n0 + weight_before) - (n1 + weight_after)
            d = (alt_goal_from_start_diff).dot(Q).dot((alt_goal_from_start_diff).T)
            d = d
            component = (np.exp(n)/np.exp(d))
            
            all_components.append(component)
            if goal == goal1:
                goal_component = component

            # add weighted value for this component
            log_sum += component * scale
        

        print("RATIO")
        print(all_components)
        # print(J_g1)
        ratio = goal_component / sum(all_components)
        print(ratio)
        if ratio > .5:
            print("Doing the thing!")

        J = np.log(goal_component) - np.log(log_sum)
        # J *= -1
        # J_addition = 0.5 * (u_diff.T.dot(R).dot(u_diff))
        # J += J_addition

        print("J components")
        print((goal_component), (log_sum))
        print(np.log(goal_component), -np.log(log_sum))
        print("stage cost~~~")
        print(J)

        return J


    # def l(self, x, u, i, terminal=False):
    #     """Instantaneous cost function.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         Instantaneous cost (scalar).
    #     """
    #     Q = self.Q_terminal if terminal else self.Q
    #     R = self.R
    #     x_diff = x - self.x_path[i]
    #     squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

    #     if terminal:
    #         return squared_x_cost

    #     u_diff = u - self.u_path[i]
    #     return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_path[i]

        val = x_diff.T.dot(Q_plus_Q_T)

        print("J_x")
        print(val)
        return val

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        val = u_diff.T.dot(self._R_plus_R_T)

        print("J_u")
        print(val)

        return val

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        val = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        print("J_xx")
        print(val)

        return val

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/dudx [action_size, state_size].
        """
        val = np.zeros((self.R.shape[0], self.Q.shape[0]))
        print("J_ux")
        print(val)
        return val

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        val = self._R_plus_R_T
        print("J_uu")
        print(val)

        return val

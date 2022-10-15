#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import os
import sys
import copy
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt

import theano.tensor as T

from ilqr import iLQR
from ilqr.cost import QRCost, PathQRCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics
from ilqr.dynamics import FiniteDiffDynamics, BatchAutoDiffDynamics, tensor_constrain

from LegiblePathQRCost import LegiblePathQRCost
from DirectPathQRCost import DirectPathQRCost
from ObstaclePathQRCost import ObstaclePathQRCost
from LegibilityOGPathQRCost import LegibilityOGPathQRCost
from OALegiblePathQRCost import OALegiblePathQRCost

from sklearn.preprocessing import normalize

# from contextlib import redirect_stdout

# with open('out.txt', 'w') as f:
#     with redirect_stdout(f):
#         print('data')


import sys
sys.stdout = open('output.txt','wt')

class NavigationDynamics(FiniteDiffDynamics):

    _state_size  = 2
    _action_size = 2

    x_eps = .1
    u_eps = .1

    def f(self, x, u, i):
        return self.dynamics(x, u)

    # hardcoded test function for being within an obstacle
    def in_object(self, x):

        obstacle = .5, 1.0
        xx, xy = x
        if xx > .5 and xx < 1.0 and xy > .5 and xy < 1.0:
            return True


        return False

    # Combine the existing state with 
    def dynamics(self, x, u, max_u=5.0):
        # # Constrain action space.
        if constrain:
            min_bounds, max_bounds = -1.0 * max_u, max_u
            
            # If we want to constrain movements to manhattan 
            # (straight lines and diagonals)
            # if False:
            #     diff = (max_bounds - min_bounds) / 2.0
            #     mean = (max_bounds + min_bounds) / 2.0
            #     ux = diff * np.tanh(u[0]) + mean
            #     uy = diff * np.tanh(u[1]) + mean
            #     u = ux, uy

            # norm1 = u / np.linalg.norm(u)
            # norm2 = normalize(u[:,np.newaxis], axis=0).ravel()
            # u = norm2 * max_u

            # downscaling the u
            # if np.linalg.norm(x) > max_u:
            #     scalar = max_u / np.linalg.norm(x)
            #     print("SCALING U")
            #     print(scalar)
            #     u = u * scalar
            #     print(u)

            # u = tensor_constrain(u, min_bounds, max_bounds)
            
        dt = self.dt

        # Apply a constraint that limits how much the robot can move per-timestep
        # TODO: apply to overall vector magnitude rather than x and y components alone

        # Moving a square
        A = np.eye(self._state_size)
        B = np.eye(self._action_size)
        v0 = A.dot(x)
        v1 = B.dot(u)
        xnext = v0 + v1     # A*x + B*u

        # Sample code that shows what would happen if we attempted raw obstacle avoidance within dynamics
        if False and self.in_object(xnext):
            # Bounce
            xnext = x

        print("xnext")
        print(str(x) + " -> " + str(xnext))

        return xnext

    """ Original based on inverted pendulum auto-differentiated dynamics model."""
    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 **kwargs):
        """Constructs an InvertedPendulumDynamics model.
        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N m].
            max_bounds: Maximum bounds for action [N m].
            m: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                BatchAutoDiffDynamics constructor.
        Note:
            state: [x, y]
            action: [torque]
            theta: 0 is pointing up and increasing counter-clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds


        self.dt = dt
        # sin_theta = x[..., 0]
        # cos_theta = x[..., 1]
        # theta_dot = x[..., 2]
        # torque = u[..., 0]

        # # Deal with angle wrap-around.
        # theta = T.arctan2(sin_theta, cos_theta)

        # # Define acceleration.
        # theta_dot_dot = -3.0 * g / (2 * l) * T.sin(theta + np.pi)
        # theta_dot_dot += 3.0 / (m * l**2) * torque

        # next_theta = theta + theta_dot * dt

        # return T.stack([
        #     T.sin(next_theta),
        #     T.cos(next_theta),
        #     theta_dot + theta_dot_dot * dt,
        # ]).T

        super(NavigationDynamics, self).__init__(self.f, 2, 2)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).
        In this case, it converts:
            [theta, theta'] -> [sin(theta), cos(theta), theta']
        Args:
            state: State vector [reducted_state_size].
        Returns:
            Augmented state size [state_size].
        """
        return state


    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.
        In this case, it converts:
            [sin(theta), cos(theta), theta'] -> [theta, theta']
        Args:
            state: Augmented state vector [state_size].
        Returns:
            Reduced state size [reducted_state_size].
        """
        return state

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    # print(J_opt)
    # print(xs)

    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


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

def scenario_0():
    restaurant      = None
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
    return start, target_goal, all_goals, restaurant

def scenario_1():
    restaurant      = None
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal1

    all_goals   = [goal1, goal3]
    return start, target_goal, all_goals, restaurant


def scenario_2():
    restaurant      = None
    start           = [0.0, 0.0]

    true_goal       = [8.0, 2.0]
    goal2           = [2.0, 1.0]
    goal4           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal4

    all_goals   = [goal1, goal4, goal2]
    return start, target_goal, all_goals, restaurant

def scenario_3():
    start           = [8.0, 2.0]

    true_goal       = [0.0, 0.0]
    goal2           = [2.0, 1.0]
    goal3           = [4.0, 1.0]

    goal1           = [4.0, 2.0]
    goal3           = [1.0, 3.0]

    target_goal = goal2
    # start = goal3

    all_goals   = [goal1, goal3, goal2]
    return start, target_goal, all_goals, restaurant


# In[1]:

# dt = .025
# N = 61

dt = .025
N = 21

x = T.dscalar("x")
u = T.dscalar("u")
t = T.dscalar("t")

start, target_goal, all_goals, restaurant = scenario_0()

x0_raw          = start    # initial state
x_goal_raw      = target_goal

# dynamics = AutoDiffDynamics(f, [x], [u], t)
dynamics = NavigationDynamics(dt)

# Note that the augmented state is not all 0.
x0      = dynamics.augment_state(np.array(x0_raw)).T
x_goal  = dynamics.augment_state(np.array(x_goal_raw)).T

x_T = N
Xrefline = np.tile(x_goal_raw, (N+1, 1))
Xrefline = np.reshape(Xrefline, (-1, 2))

u_blank = np.asarray([0.0, 0.0])
Urefline = np.tile(u_blank, (N, 1))
Urefline = np.reshape(Urefline, (-1, 2))

state_size = 2
action_size = 2
# The coefficients weigh how much your state error is worth to you vs
# the size of your controls. You can favor a solution that uses smaller
# controls by increasing R's coefficient.
Q = 1.0 * np.eye(state_size)
# R = 100.0 * np.eye(action_size)
# R = 100.0 * np.eye(action_size)
# Qf = np.identity(2) * 40.0
# R = 100.0 * np.eye(action_size)
R = 200.0 * np.eye(action_size)
Qf = np.identity(2) * 400.0

# cost = LegiblePathQRCost(Q, R, Qf, Xrefline, Urefline, start, target_goal, all_goals, N, dt, restaurant=restaurant)
cost = OALegiblePathQRCost(Q, R, Qf, Xrefline, Urefline, start, target_goal, all_goals, N, dt, restaurant=restaurant)
# cost = DirectPathQRCost(Q, R, Xrefline, Urefline, start, target_goal, all_goals, N, dt, restaurant=restaurant)
# cost = ObstaclePathQRCost(Q, R, Xrefline, Urefline, start, target_goal, all_goals, N, dt, restaurant=restaurant)
# cost = LegibilityOGPathQRCost(Q, R, Xrefline, Urefline, start, target_goal, all_goals, N, dt, restaurant=restaurant)

# l = leg_cost.l
# l_terminal = leg_cost.term_cost
# cost = AutoDiffCost(l, l_terminal, x_inputs, u_inputs)



FLAG_JUST_PATH = False
if FLAG_JUST_PATH:
    traj        = Xrefline
    us_init     = Urefline
    cost        = PathQRCost(Q, R, traj, us_init)
    print("Set to old school pathing")
    exit()

# x_dot = (dt * t - u) * x**2
# f = T.stack([x + x_dot * dt])

ilqr = iLQR(dynamics, cost, N)
J_hist = []
xs, us = ilqr.fit(x0_raw, Urefline, tol=1e-6, n_iterations=N, on_iteration=on_iteration)

# # Reduce the state to something more reasonable.
# xs = dynamics.reduce_state(xs)

# # Constrain the actions to see what's actually applied to the system.
# us = constrain(us, dynamics.min_bounds, dynamics.max_bounds)

# # Constrain the inputs.
# min_bounds = np.array([0.0, 0.0, 0.0])
# max_bounds = np.array([61.0, 10.0, 10.0])
# u_constrained_inputs = tensor_constrain(us, min_bounds, max_bounds)

# # Constrain the solution.
# xs, us_unconstrained = ilqr.fit(x0, us_init)
# us = constrain(us_unconstrained, min_bounds, max_bounds)

t = np.arange(N) * dt
theta = np.unwrap(xs[:, 0])  # Makes for smoother plots.
theta_dot = xs[:, 1]


# _ = plt.plot(theta, theta_dot)
# _ = plt.xlabel("theta (rad)")
# _ = plt.ylabel("theta_dot (rad/s)")
# _ = plt.title("Phase Plot")
# plt.show()
# plt.clf()

# Plot of the path through space
verts = xs
xs, ys = zip(*verts)
gx, gy = zip(*all_goals)
sx, sy = zip(*[x0_raw])


cost.graph_legibility_over_time(verts, us)


# n_spline = fn_pathpickle_from_exp_settings(exp_settings) + 'sample-cubic_spline_imposed_tangent_direction.png'
# plt.savefig(fn_spline)

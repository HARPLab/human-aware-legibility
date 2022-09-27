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

from LQR_cost import LegiblePathQRCost

class NavigationDynamics(FiniteDiffDynamics):

    _state_size  = 2
    _action_size = 2

    def f(self, x, u, i):
        return self.dynamics(x, u)

    def dynamics(self, x, u):
        # # Constrain action space.
        # if constrain:
        #     u = tensor_constrain(u, min_bounds, max_bounds)

        dt = self.dt

        # Moving a square
        A = np.eye(self._state_size)
        B = np.eye(self._action_size) * dt

        v0 = A.dot(x)
        v1 = B.dot(u)

        xnext = v0 + v1     # A * x + B*u
        return xnext

    # these are functions from michelle's implementation
    # def rk4(self, x, u, dt):
    #     # rk4 for integration
    #     k1 = dt * self.dynamics(x, u)
    #     k2 = dt * self.dynamics(x + k1/2, u)
    #     k3 = dt * self.dynamics(x + k2/2, u)
    #     k4 = dt * self.dynamics(x + k3, u)
    #     # print("rk4")
    #     # print(x)
    #     # print(u)
    #     # print(dt)
    #     return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)


    # def dynamics_jacobians(self, x, u, dt):
    #     # returns the discrete time dynamics jacobians
    #     A = self.rk4(0, u, dt) # FD.jacobian(_x -> rk4(_x,u,dt),x)
    #     B = self.rk4(x, 0, dt) #FD.jacobian(_u -> rk4(x,_u,dt),u)
        
    #     return A,B

    """ Original based on inverted pendulum auto-differentiated dynamics model."""
    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 m=1.0,
                 l=1.0,
                 g=9.80665,
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
    print(J_opt)
    print(xs)

    final_state = dynamics.reduce_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)


def get_window_dimensions_for_envir(start, goals):
    xmin, xmax, ymin, ymax = 0, 0, 0, 0

    all_points = copy.copy(goals)
    all_points.append(start)
    for pt in all_points:
        x, y = pt

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



# In[1]:

dt = .1 #.025
N = 61

start           = [0.0, 0.0]

true_goal       = [8.0, 2.0]
goal2           = [2.0, 1.0]
goal3           = [4.0, 1.0]

goal1           = [4.0, 2.0]
goal3           = [1.0, 3.0]

true_goal = goal2
start = goal3

all_goals   = [goal1, goal3, goal2]
# all_goals   = [[0.0, 0.0], goal2]
bounds0     = [-2.0,    -2.0]
bounds1     = [10.0,    10.0]

x = T.dscalar("x")
u = T.dscalar("u")
t = T.dscalar("t")

target_goal = true_goal

x0_raw          = [0.0, 0.0]    # initial state
x_goal_raw      = true_goal

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

Q = np.identity(2)
R = np.identity(2)
Qf = np.identity(2) * 10


cost = LegiblePathQRCost(Q, R, Xrefline, Urefline, target_goal, all_goals)

FLAG_JUST_PATH = True
if FLAG_JUST_PATH:
    traj        = Xrefline
    us_init     = Urefline
    cost        = PathQRCost(Q, R, traj, us_init)
    print("Set to old school pathing")
    # exit()

# x_dot = (dt * t - u) * x**2
# f = T.stack([x + x_dot * dt])

ilqr = iLQR(dynamics, cost, N)
J_hist = []
xs, us = ilqr.fit(x0_raw, Urefline, n_iterations=N, on_iteration=on_iteration)

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

print("verts")
print(verts)
print("Attempt to display this path")

xmin, xmax, ymin, ymax = get_window_dimensions_for_envir(start, all_goals)

plt.plot(xs, ys, 'o--', lw=2, color='black', label="path", markersize=3)
plt.plot(gx, gy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="green", lw=0, label="goals")
plt.plot(sx, sy, marker="o", markersize=10, markeredgecolor="black", markerfacecolor="grey", lw=0, label="start")
_ = plt.xlabel("X")
_ = plt.ylabel("Y")
_ = plt.title("Path through space")
plt.legend(loc="upper left")
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()
plt.clf()

_ = plt.plot(t, us)
_ = plt.xlabel("time (s)")
_ = plt.ylabel("Force (N)")
_ = plt.title("Action path")
# plt.show()
# plt.clf()

_ = plt.plot(J_hist)
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Total cost")
_ = plt.title("Total cost-to-go")
# plt.show()
# plt.clf()

# n_spline = fn_pathpickle_from_exp_settings(exp_settings) + 'sample-cubic_spline_imposed_tangent_direction.png'
# plt.savefig(fn_spline)

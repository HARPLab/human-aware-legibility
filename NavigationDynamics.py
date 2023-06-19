import numpy as np
from ilqr.dynamics import FiniteDiffDynamics, tensor_constrain, constrain

from shapely.geometry import LineString
from shapely.geometry import Point
import decimal

import math
import copy

### CLASS DESCRIBING THE DYNAMICS OF A SIMPLE ROBOT
### MOVING THOUGH SPACE
class NavigationDynamics(FiniteDiffDynamics):

    _state_size  = 3    # state is x_x, x_y, x_theta
    _action_size = 2

    x_eps = .05
    u_eps = .05

    def f(self, x, u, i):
        return self.dynamics(x, u)

    def get_angle_between_triplet(self, a_in, b_in, c_in):
        a = copy.copy(a_in)
        b = copy.copy(b_in)
        c = copy.copy(c_in)

        # A debugging check for shape across these objects
        # format_check = str((a.shape, b.shape, c.shape))
        # if format_check == "((2,), (2,), (2,))":
        #     pass
        # else:
        #     print("abt check")
        #     print("!" + format_check + "!")
        #     # print(type(a), type(b), type(c))
        #     print(a.shape, b.shape, c.shape)
        #     print(a, b, c)
        #     exit()

        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        if ang < 0:
            return ang + 360
        else:
            return ang
        
    # Returns the degrees clockwise from the 0 of east
    def get_heading_of_pt_diff_p2_p1(self, p2_in, p1_in):
        # print("heading looking from ")
        p1 = copy.copy(p1_in)[:2]
        p2 = copy.copy(p2_in)[:2]
        unit_vec    = np.asarray([p1[0] + 1.0, p1[1]])

        heading     = self.get_angle_between_triplet(p2, p1, unit_vec)
        return heading

    ##### METHODS FOR ANGLE MATH - Should match SocLegPathQRCost
    def get_heading_moving_between(self, p2, p1):
        # print("Get heading moving in nav from " + str(p1) + " to " + str(p2))
        # print(p2)
        # print(p1)

        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        ang1    = np.arctan2(*p1[::-1])
        ang2    = np.arctan2(*p2[::-1])
        heading = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        
        # heading = self.get_minimum_rotation_to(heading)
        # Heading is in degrees

        return heading

    # Combine the existing state with 
    def dynamics(self, x_triplet, u_input, max_u=8.0):
        # TODO raise max_u to experimental settings
        # print("IN DYNAMICS")
        if u_input is None:
            # print("if u is none, don't go anywhere")
            # if none, don't move anywhere
            return x_triplet

        # This is the distance we go in dt time
        dt          = self.dt # seconds
        u           = copy.copy(u_input)
        xy          = np.asarray([x_triplet[0], x_triplet[1]])
        max_speed   = max_u / self.dt

        if np.isnan(u[0]) or np.isnan(u[1]):
            print("ALERT U IS NAN")
            pass

        if True: #constrain:
            min_bounds, max_bounds = -1.0 * max_u, max_u
            if np.linalg.norm(u) > max_u:
                print("Speed limit applied")
                scalar = max_u / np.linalg.norm(u)
                u = u * scalar
                if np.isnan(u[0]) or np.isnan(u[1]):
                    print("ALERT SPEED LIMIT ADDED A NAN")

        u = u * dt
        xnext_wout_theta   = xy + (u)

        if np.isnan(u[0]) or np.isnan(u[1]):
            # print("ALERT NAN after math")
            xnext_wout_theta = xy

        # # Moving a square
        # # We only apply this to the x y parts of the matrix 
        # A = np.eye(self._action_size)       #(self._state_size)
        # B = np.eye(self._action_size).dot(dt)
        # v0 = A.dot(xy)
        # v1 = B.dot(u)

        # print("dynams")
        # print(x_triplet.shape, u_input.shape)

        # print(x_triplet.shape, x_triplet, xnext_wout_theta)
        # print(xy, v0)
        # print(u, dt, v1)

        # print(xy, u*dt)
        # print("then")
        if self._state_size == 3:
            xtheta_old = x_triplet[2]
        else:
            xtheta_old = 0
        
        # Heading is clockwise degrees from EAST
        xtheta_new = 19.3 #self.get_heading_of_pt_diff_p2_p1(xnext_wout_theta, xy)

        # if xtheta_old == xtheta_old:
        #     print("Robot maintained the same heading, but that's fine")

        # print("u in dynamics model")
        # print(u)

        if self._state_size == 3:
            xnext = np.asarray([xnext_wout_theta[0], xnext_wout_theta[1], xtheta_new]).T
        else:
            xnext = xnext_wout_theta


        # print("xnext heading notes")
        # print("from " + str(xy) + " to " + str(xnext_wout_theta) + " is a heading of " + str(xtheta_new))
        # print(str(xy) + " -> " + str(xnext) + " step of magnitude " + str(np.linalg.norm(u)))

        # print("After step, location is:")
        # print(xnext)

        print("dynams")
        print("x_triplet, u_input, ->, xnext")
        print(x_triplet, u_input, "->", xnext)
        print("xy, xtheta_old, xtheta_new")
        print(xy, xtheta_old, xtheta_new)


        return xnext

    # # Combine the existing state with 
    # def dynamics_v1(self, x_triplet, u, max_u=10.0):
    #     # TODO raise max_u to experimental settings
    #     print("IN DYNAMICS")

    #     # This is the distance we go in dt time
    #     dt          = self.dt # seconds
    #     # if the max speed is .5 m/s
    #     max_speed   = 20.0 #15 #m
    #     max_u       = dt * max_speed

    #     # # Constrain action space.
    #     # Apply a constraint that limits how much the robot can move per-timestep
    #     if False: #constrain:
    #         min_bounds, max_bounds = -1.0 * max_u, max_u
            
    #         # If we want to constrain movements to manhattan 
    #         # (straight lines and diagonals)
    #         # if False:
    #         #     diff = (max_bounds - min_bounds) / 2.0
    #         #     mean = (max_bounds + min_bounds) / 2.0
    #         #     ux = diff * np.tanh(u[0]) + mean
    #         #     uy = diff * np.tanh(u[1]) + mean
    #         #     u = ux, uy

    #         # norm1 = u / np.linalg.norm(u)
    #         # norm2 = normalize(u[:,np.newaxis], axis=0).ravel()
    #         # u = norm2 * max_u

    #         # downscaling the u
    #         if np.linalg.norm(u) > max_u:
    #             scalar = max_u / np.linalg.norm(u)
    #             u = u * scalar
    #             # print(u)

    #         # u = tensor_constrain(u, min_bounds, max_bounds)

    #     if x_triplet.shape < 3:
    #         print("Eeeek! shape is wrong!")
    #         # exit()

    #     xy       = [x_triplet[0], x_triplet[1]]

    #     # Moving a square
    #     # We only apply this to the x y parts of the matrix 
    #     A = np.eye(2)       #(self._state_size)
    #     B = np.eye(self._action_size)
    #     v0 = A.dot(xy)
    #     v1 = B.dot(u) * dt



    #     xtheta_old  = x_triplet[2]
    #     xtheta_new  = self.get_heading_of_pt_diff_p2_p1(xnext_wout_theta, xy)


    #     if xnext_wout_theta[0] == np.nan or xnext_wout_theta[1] == np.nan:
    #         xnext_wout_theta     = xy
    #         xtheta_new           = xtheta_old

    #         u = np.asarray([0, 0])
    
    #     if xtheta_old == xtheta_old:
    #         print("Robot maintained the same heading, but that's fine")

    #     print("u in dynamics model")
    #     print(u)

    #     xnext = [xnext_wout_theta[0], xnext_wout_theta[1], xtheta_new].T
    #     print("xnext heading notes")
    #     print("from " + str(xy) + " to " + str(xnext_wout_theta) + " is a heading of " + str(xtheta_new))
    #     print(str(xy) + " -> " + str(xnext) + " step of magnitude " + str(np.linalg.norm(u)))

    #     # is_in_obstacle = self.is_in_obstacle(v0, v1)
    #     # across_obstacle = self.across_obstacle(v0, v1)

    #     # if np.isnan(np.linalg.norm(u)):
    #     #     print("caught a nan")
    #     #     xnext = v0
    #     # elif across_obstacle:
    #     #     print("teleporting across an obstacle!")
    #     #     xnext = v0
    #     # elif is_in_obstacle:
    #     #     print("don't enter obstacle")
    #     #     xnext = v0
    #     # else:
    #     #     xnext = v0 + v1     # A*x + B*u

    #     return xnext

    """ Original based on inverted pendulum auto-differentiated dynamics model."""
    def __init__(self,
                 dt, 
                 exp,
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

        self.exp = exp
        self._state_size    = exp.get_state_size()
        self._action_size   = exp.get_action_size()

        super(NavigationDynamics, self).__init__(self.f, self._state_size, self._action_size)

    def get_obstacle_penalty_given_obj(self, x, x1, obst_center, threshold):
        obst_dist = obst_center - x
        obst_dist = np.abs(np.linalg.norm(obst_dist))

        obst_dist = obst_center - x
        obst_dist = np.abs(np.linalg.norm(obst_dist))

        # rho is the distance the closest point is from the center
        rho             = obst_dist - self.exp.get_obstacle_buffer()
        eta             = 1.0

        # if rho > threshold:
        #     return 0

        if obst_dist > threshold:
            return False

        return True

    # Citation for future paper
    # https://studywolf.wordpress.com/2016/11/24/full-body-obstacle-collision-avoidance/
    def is_in_obstacle(self, x, x1):
        TABLE_RADIUS    = self.exp.get_table_radius()
        OBS_RADIUS      = self.exp.get_observer_radius()
        GOAL_RADIUS     = self.exp.get_goal_radius()

        tables      = self.exp.get_tables()
        goals       = self.exp.get_goals()
        observers   = self.exp.get_observers()

        obstacle_penalty = False
        for table in tables:
            new_obstacle_penalty = self.get_obstacle_penalty_given_obj(x, x1, table.get_center(), TABLE_RADIUS)
            if new_obstacle_penalty:
                return True

        for obs in observers:
            obstacle = obs.get_center()
            new_obstacle_penalty = self.get_obstacle_penalty_given_obj(x, x1, obs.get_center(), OBS_RADIUS)
            if new_obstacle_penalty:
                return True

        for g in goals:
            if g is not self.exp.get_target_goal():
                obstacle = g
                new_obstacle_penalty = self.get_obstacle_penalty_given_obj(x, x1, g, GOAL_RADIUS)
                if new_obstacle_penalty:
                    return True

        # x1 = x
        # if i > 0:
        #     x0 = self.x_path[i - 1]
        # else:
        #     x0 = x

        # if self.across_obstacle(x0, x1):
        #     obstacle_penalty += 1.0
        #     print("TELEPORT PENALTY APPLIED")

        return False

    def across_obstacle(self, x0, x1):
        TABLE_RADIUS    = self.exp.get_table_radius()
        OBS_RADIUS      = .1
        GOAL_RADIUS     = .15 #.05

        tables      = self.exp.get_tables()
        goals       = self.exp.get_goals()
        observers   = self.exp.get_observers()

        l = LineString([x0, x1])

        for t in tables:
            ct = t.get_center()
            p = Point(ct[0],ct[1])
            c = p.buffer(TABLE_RADIUS).boundary
            i = c.intersection(l)
            if i is True:
                return True
        
        for o in observers:
            ct = o.get_center()
            p = Point(ct[0],ct[1])
            c = p.buffer(OBS_RADIUS).boundary
            i = c.intersection(l)
            if i is True:
                return True
        
        for g in goals:
            ct = g
            p = Point(ct[0],ct[1])
            c = p.buffer(GOAL_RADIUS).boundary
            i = c.intersection(l)
            if i is True:
                return True

        return False

    # hardcoded test function for being within an obstacle
    def in_object(self, x_double):

        obstacle = .5, 1.0
        xx, xy = x
        if xx > .5 and xx < 1.0 and xy > .5 and xy < 1.0:
            return True


        return False

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

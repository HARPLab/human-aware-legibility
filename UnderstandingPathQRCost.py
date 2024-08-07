import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import autograd.numpy as np
import matplotlib.pyplot as plt
import decimal
import copy
import math
import torch

from ilqr import iLQR
from ilqr.cost import Cost
from ilqr.cost import QRCost
from ilqr.cost import PathQRCost, AutoDiffCost, FiniteDiffCost
from ilqr.dynamics import constrain
from ilqr.examples.pendulum import InvertedPendulumDynamics
from ilqr.dynamics import BatchAutoDiffDynamics, tensor_constrain

from scipy.optimize import approx_fprime
from sklearn import preprocessing

import utility_legibility as legib
import utility_environ_descrip as resto
import pipeline_generate_paths as pipeline
import pdb

from LegiblePathQRCost import LegiblePathQRCost
import PathingExperiment as ex

from shapely.geometry import LineString
from shapely.geometry import Point

np.set_printoptions(suppress=True)
np.seterr(divide='raise')
MATH_EPSILON = 0 #.0000001

FLAG_SECONDARY_CONSIDERED   = False
FLAG_PD_WITH_ALTS           = False

class UnderstandingPathQRCost(LegiblePathQRCost):
    FLAG_DEBUG_J = False
    FLAG_DEBUG_STAGE_AND_TERM = True

    FLAG_COST_PATH_OVERALL  = True
    FLAG_OBS_FLAT_PENALTY   = True

    best_xs = None
    best_us = None

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


    ##### METHODS FOR ANGLE MATH - Should match SocLegPathQRCost
    def get_heading_moving_between(self, p2, p1):
        print("Get heading moving from " + str(p1) + " to " + str(p2))
        print(p2)
        print(p1)

        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        ang1    = np.arctan2(*p1[::-1])
        ang2    = np.arctan2(*p2[::-1])
        heading = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        print(heading)

        # heading = self.get_minimum_rotation_to(heading)
        # Heading is in degrees
        return heading

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
            if i == True:
                print("TELEPORTED ACROSS: table " + str(ct))
                return True
        
        for o in observers:
            ct = o.get_center()
            p = Point(ct[0],ct[1])
            c = p.buffer(OBS_RADIUS).boundary
            i = c.intersection(l)
            if i == True:
                print("TELEPORTED ACROSS: o " + str(ct))
                return True
        
        for g in goals:
            ct = g
            p = Point(ct[0],ct[1])
            c = p.buffer(GOAL_RADIUS).boundary
            i = c.intersection(l)
            if i == True:
                print("TELEPORTED ACROSS: goal " + str(ct))
                return True

        return False

    def get_closest_point_on_line(self, x, i, obst_center):
        x1 = x
        if i > 0:
            x0 = self.x_path[i - 1]
        else:
            x0 = x


        vec_line = x1 - x0
        # the vector from the obstacle to the first line point
        vec_ob_line = obst_center - x0
        # calculate the projection normalized by length of arm segment
        projection = (np.dot(vec_ob_line, vec_line) /
                      np.sum((vec_line)**2))

        if projection < 0:         
            # then closest point is the start of the segment
            closest = x0
        elif projection > 1:
            # then closest point is the end of the segment
            closest = x1
        else:
            closest = x0 + projection * vec_line

        print("ptclosest")
        print(closest)
        # calculate distance from obstacle vertex to the closest point
        dist = np.abs(np.linalg.norm(obst_center - closest)) #np.sqrt(np.sum((obst_center - closest)**2))

        print("Closest point dist")
        print(dist)
        return dist

    # https://studywolf.wordpress.com/2016/11/24/full-body-obstacle-collision-avoidance/
    def get_obstacle_penalty_given_obj(self, x_triplet, i, obst_center, obstacle_radius):
        obstacle_radius = obstacle_radius # the physical obj size
        obstacle_buffer = self.exp.get_obstacle_buffer() # the area in which the force will apply
        rho_o       = obstacle_radius + obstacle_buffer
        x           = x_triplet[:2]

        # obst_dist is the distance between the point and the center of the obj
        obst_dist = obst_center - x
        obst_dist = np.abs(np.linalg.norm(obst_dist))
        # obst_dist = self.get_closest_point_on_line(x, i, obst_center)

       # rho is the distance between the object's edge and the pt
        rho             = obst_dist - obstacle_buffer #rho_o
        # if rho is negative, we are inside the sphere
        eta     = 1.0
        v       = obst_center
        closest = x
        threshold = obstacle_radius + obstacle_buffer

        # print("Are we in it?")
        # print("dist, threshold, rho")
        # print(obst_dist, threshold, rho)

        Fpsp = 0
        if rho < obstacle_radius:
            # rho = rho
            # vector component
            # drhodx = (v - closest) / rho
            d_rho_top = obst_center - x
            d_rho_dx = d_rho_top / rho


            # print("Rho says yes")
            eta = 1.0
            drhodx = (v - closest) / rho
            Fpsp = (eta * (1.0/rho - 1.0/threshold) *
                    1.0/rho**2 * d_rho_dx)

            # print("Fpsp")
            # print(Fpsp)

            # print(1.0/rho)
            # print(1.0/threshold)
            # print(1.0 / rho**2)
            # print(drhodx)
            # print("~~~~~")
        else:
            # print("rho says no")
            pass

        Fpsp = np.linalg.norm(Fpsp)
        # print("obspenalty is: ")
        # print(Fpsp)

        if Fpsp < 0:
            print("ALERT: REWARD FOR ENTERING OBSTACLE!")

        if obst_dist < obstacle_radius:
            print("Inside the actual obj")
            print(x, obst_center)
            print("Distance")
            print(obst_dist, self.dist_between(obst_center, x))
            if Fpsp == 0:
                print("ALERT: no penalty")
        elif obst_dist < obstacle_buffer + obstacle_radius:
            print("Inside the overall force diagram")
            print(x, obst_center)
            print("Distance")
            print(obst_dist, self.dist_between(obst_center, x))
            if Fpsp == 0:
                print("ALERT: no penalty")
        else:
            print("Not in an obstacle")
            if Fpsp > 0:
                print("ALERT: obstacle penalty when not in object")

        if Fpsp > 0:
            print("Penalty: " + str(Fpsp))

        return 0.0
        return Fpsp

    # Citation for future paper
    # https://studywolf.wordpress.com/2016/11/24/full-body-obstacle-collision-avoidance/
    def get_obstacle_penalty(self, x, i, goal):
        TABLE_RADIUS    = self.exp.get_table_radius()
        OBS_RADIUS      = self.exp.get_observer_radius() #.2
        GOAL_RADIUS     = self.exp.get_goal_radius() #.3 #.05

        tables      = self.exp.get_tables()
        goals       = self.goals
        observers   = self.exp.get_observers()

        obstacle_penalty = 0
        for table in tables:
            obstacle_penalty += self.get_obstacle_penalty_given_obj(x, i, table.get_center(), TABLE_RADIUS)

        for obs in observers:
            obstacle = obs.get_center()
            obstacle_penalty += self.get_obstacle_penalty_given_obj(x, i, obs.get_center(), OBS_RADIUS)

        for g in goals:
            if not np.array_equal(g, goal):
                obstacle = g
                obstacle_penalty += self.get_obstacle_penalty_given_obj(x, i, g, GOAL_RADIUS)

        # x1 = x
        # if i > 0:
        #     x0 = self.x_path[i - 1]
        # else:
        #     x0 = x

        # if self.across_obstacle(x0, x1):
        #     obstacle_penalty += 1.0
        #     print("TELEPORT PENALTY APPLIED")

        return obstacle_penalty


    # TODO ADD TEST SUITE FOR THIS
    def get_obstacle_penalty_v1(self, x, i, goal):
        TABLE_RADIUS    = self.exp.get_table_radius()
        OBS_RADIUS      = .2
        GOAL_RADIUS     = .3 #.05

        tables      = self.exp.get_tables()
        goals       = self.goals
        observers   = self.exp.get_observers()

        x1 = x
        if i > 0:
            x0 = self.x_path[i - 1]
        else:
            x0 = x

        obstacle_penalty = 0
        for table in tables:
            obstacle = table.get_center()
            obs_dist = obstacle - x
            obs_dist = np.abs(np.linalg.norm(obs_dist))
            # Flip so edges lower cost than center

            if obs_dist < TABLE_RADIUS:
                obs_dist = TABLE_RADIUS - obs_dist
                print("PENALTY: table obstacle dist for " + str(x) + " " + str(obs_dist))
                print(str(table.get_center()))
                # obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                # OBSTACLE PENALTY NOW ALWAYS SCALED TO RANGE 0 -> 1
                if self.FLAG_OBS_FLAT_PENALTY:
                    obstacle_penalty += 1.0
                obstacle_penalty += np.abs(obs_dist / TABLE_RADIUS)

                # np.inf #

        for obs in observers:
            obstacle = obs.get_center()
            obs_dist = obstacle - x
            obs_dist = np.abs(np.linalg.norm(obs_dist))
            # Flip so edges lower cost than center

            if obs_dist < OBS_RADIUS:
                obs_dist = OBS_RADIUS - obs_dist
                print("PENALTY: obs obstacle dist for " + str(x) + " " + str(obs_dist))
                print(str(obs.get_center()))
                # obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                # OBSTACLE PENALTY NOW ALWAYS SCALED TO RANGE 0 -> 1
                if self.FLAG_OBS_FLAT_PENALTY:
                    obstacle_penalty += 1.0

                obstacle_penalty += np.abs(obs_dist / OBS_RADIUS) #**2

        for g in goals:
            if not np.array_equal(g, self.exp.get_target_goal()):
                obstacle = g
                obs_dist = obstacle - x
                obs_dist = np.abs(np.linalg.norm(obs_dist))
                # Flip so edges lower cost than center

                if obs_dist < GOAL_RADIUS:
                    obs_dist = GOAL_RADIUS - obs_dist
                    print("PENALTY: goal obstacle dist for " + str(x) + " " + str(obs_dist))
                    print(str(g))
                    # obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                    # OBSTACLE PENALTY NOW ALWAYS SCALED TO RANGE 0 -> 1
                    if self.FLAG_OBS_FLAT_PENALTY:
                        obstacle_penalty += 1.0
                    obstacle_penalty += np.abs(obs_dist / GOAL_RADIUS) #**2
            # else:
            #     print("inside final goal " + str(g))

        if self.across_obstacle(x0, x1):
            obstacle_penalty += 1.0
            print("TELEPORT PENALTY APPLIED")

        obstacle_penalty = obstacle_penalty * np.Inf
        return obstacle_penalty

    def get_angle_between_pts(self, p2, p1):
        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        ang1    = np.arctan2(*p1[::-1])
        ang2    = np.arctan2(*p2[::-1])
        heading = np.rad2deg((ang1 - ang2) % (2 * np.pi))

        # Heading is in degrees
        return heading


    ##### METHODS FOR ANGLE MATH
    def get_angle_between(self, p2, p1):
        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        ang1    = np.arctan2(*p1[::-1])
        ang2    = np.arctan2(*p2[::-1])
        heading = np.rad2deg((ang1 - ang2) % (2 * np.pi))

        heading = self.get_minimum_rotation_to(heading)

        # Heading is in degrees
        return heading

    def get_min_rotate_angle_diff(self, x, y):
        if np.abs(x - y) > 180:
            return (360) - abs(x - y)

        a = min((360) - abs(x - y), abs(x - y))
        return abs(x - y)

    def get_heading_look_from_x0_to_x1(self, x0, x1):
        unit_vec    = [x0[0] + 1.0, x0[1]]
        heading     = self.get_angle_between_triplet(x1, x0, unit_vec)
        heading     = heading % 360

        print("Look from " + str(x0) + " to " + str(x1) + " == " + str(heading))
        return heading

    # ADA TODO
    def angle_x1_x0(self, x_cur, x_prev):
        x_cur_x, x_cur_y = x_cur
        x_prev_x, x_prev_y = x_prev

        # atan2 = -180 to 180, in radians
        heading = np.arctan2(x_cur_y-x_prev_y, x_cur_x-x_prev_x)
        heading = heading * (180.0/np.pi)

        heading = (heading + 360) % 360

        # This angle is counter clockwise from 0 at EAST
        # 90 = NORTH
        # 270 = SOUTH

        return heading

    def get_angle_to_look_at_point(self, robot_x1, robot_x0, look_at_pt):
        # robot_x1 = robot_x[:2]
        # robot_x0 = robot_x[2:]

        prev_angle      = self.angle_x1_x0(robot_x1, robot_x0)
        prev_angle_rev  = self.angle_x1_x0(robot_x0, robot_x1)
        # prev_angle = self.get_heading_look_from_x0_to_x1(robot_x1, robot_x0)

        goal_angle = self.angle_x1_x0(look_at_pt, robot_x1)

        # target_shifted = look_at_pt - robot_x1

        # angle = math.atan2(target_shifted[1], target_shifted[0])
        # angle = angle * (180.0/np.pi)
        # angle = (angle + 360) % 360

        # # atan2 gives -180 to 180
        # if angle < 0:
        #     angle = 360 - angle

        print("Robot looking at " + str(prev_angle))
        print("Opp: " + str(prev_angle_rev))
        print("Angle to goal is " + str(goal_angle))

        offset = np.abs(prev_angle - goal_angle)
        print("OG offset " + str(offset))

        if offset > 180:
            offset = 360 - offset

        print("final offset " + str(offset))


        return offset

    def inversely_proportional_to_distance(self, x):
        if x == 0:
            return np.Inf
        return 1.0 / (x)

    def get_relative_distance_k(self, x, goal=None, goals=None):
        total_distance = 0.0
        if goals is None:
            goals = self.goals

        if goal is None:
            goal = self.target_goal

        for g in goals:
            dist = g - x
            dist = np.abs(np.linalg.norm(dist))

            total_distance += self.inversely_proportional_to_distance(dist)

        target_goal_dist = np.abs(np.linalg.norm(goal - x))
        tg_dist = self.inversely_proportional_to_distance(target_goal_dist)

        # rel_dist = 1.0 - (tg_dist / total_distance)

        rel_dist = (total_distance - tg_dist) / total_distance

        return rel_dist

    def get_relative_distance_k_sqr(self, x, goal=None, goals=None):
        total_distance = 0.0
        if goals is None:
            goals = self.goals

        if goal is None:
            goal = self.target_goal

        for g in goals:
            dist = g - x
            dist = np.abs(np.linalg.norm(dist))

            total_distance += (self.inversely_proportional_to_distance(dist) ** 2)

        target_goal_dist = np.abs(np.linalg.norm(goal - x))
        tg_dist = self.inversely_proportional_to_distance(target_goal_dist)**2

        # rel_dist = 1.0 - (tg_dist / total_distance)

        rel_dist = (total_distance - tg_dist) / total_distance

        return rel_dist


    def get_minimum_rotation_to(self, angle):
        if angle < 0:
            angle = abs(angle)

        if angle < 180:
            return angle
        else:
            angle = 360 - angle

        return angle

    def get_relative_distance_k_v1(self, x, goal, goals):
        max_distance = 0.0
        for g in goals:
            dist = g - x
            dist = np.abs(np.linalg.norm(dist))

            if dist > max_distance:
                max_distance = dist

        target_goal_dist = goal - x
        tg_dist = np.abs(np.linalg.norm(target_goal_dist))

        return 1 - (tg_dist / max_distance)

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        
        print("unit vecs")
        print(v1_u)
        print(v2_u)
        ang = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return ang

    def inversely_proportional_to_angle(self, angle):
        if angle == 0:
            print("There's an infinity involved here")
            return np.Inf

        return 1.0 / (angle)

    def get_angle_between_triplet(self, a_in, b_in, c_in):
        # print(type(a), type(b), type(c))

        a = copy.copy(a_in)
        b = copy.copy(b_in)
        c = copy.copy(c_in)

        # Reminder atan2 does y then x, versus x y
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        ang = ang + 360 if ang < 0 else ang

        # Alternate implementation for comparison
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        alt_check = np.degrees(angle)

        if ang > 180:
            ang = 360.0 - ang

        print("\tsoc math check")
        print("\t", a, b, c, str(ang) + " degrees", alt_check)

        if np.abs(ang - alt_check) > 1.0:
            print("ALERT: HEADING ANGLE MATH MAY BE INCONSISTENT")

        return ang


    def get_heading_of_pt_diff_p2_p1(self, p2, p1):
        unit_vec    = [p1[0] + 1.0, p1[1]]
        heading     = self.get_angle_between_triplet(p2, p1, unit_vec)
        return heading

    def get_legibility_heading_stage_cost_3d(self, x, u, i, goal, visibility_coeff, override=None):
        # if len(x) > 2 and x[2] is None:
        #     return 1.0

        P_oa = self.prob_heading(x, u, i, goal, visibility_coeff, override=override)

        # cost = ((np.exp(1.0)) - (np.exp(P_oa)))
        cost = ((1.0) - (P_oa))
        return cost

    def get_legibility_heading_stage_cost_from_pt_seq(self, x_cur, x_prev, i, goal, visibility_coeff, u=None, override=None):
        # start = self.exp.get_start()
        # k = self.exp.get_dist_scalar_k()
        # print("mergh")
        # print(k)

        # print(np.abs((x_cur[0] - x_prev[0])), np.abs((x_cur[1] - x_prev[1])))
        # # If the heading is not going anywhere, put max penalty
        # if np.abs((x_cur[0] - x_prev[0])) < k and np.abs((x_cur[1] - x_prev[1])) < k:
        #     return 2.0

        # print(np.abs((x_cur[0] - goal[0])), np.abs((x_cur[1] - goal[1])))
        # # if we are on the goal, return max penalty
        # if np.abs((x_cur[0] - goal[0])) < k and np.abs((x_cur[1] - goal[1])) < k:
        #     return 2.0

        # Protection against the weird case of both ydiff and xdiff = 0
        # which would be a discontinuity
        # https://github.com/google/jax/discussions/15865
        if np.array_equal(x_cur, x_prev):
            return 2.0

        # How do we handle getting to the goal early?
        if np.array_equal(x_cur, goal):
            return 2.0


        P_oa = self.prob_heading_from_pt_seq(x_cur, x_prev, i, goal, visibility_coeff, u=u, override=override)

        if (P_oa) < 0:
            print("ALERT: P_oa too small")
        elif (P_oa) > 1.0:
            print("ALERT: P_oa too large")

        # k = 5.0
        # scalar = np.abs((P_oa) - 0.5) * k

        cost = (1.0) - (P_oa)
        cost = (P_oa)

        # return (scalar) * ((cost))
        return ((cost)) #+ (10.0)

    def prob_heading_from_pt_seq(self, x_cur_in, x_prev_in, i_in, goal_in, visibility_coeff, u=None, override=None):
        x_cur = x_cur_in
        x_prev = x_prev_in

        if len(x_cur) == 4:
            x_cur = x_cur[2:]
        if len(x_prev) == 4:
            x_prev = x_prev[2:]


        # 4.117876  -0.2    -6.848375

        # g_away          = np.asarray([7.41, -6.99]) # G_AWAY   0 angle
        # g_me            = np.asarray([3.79, -6.99]) # G_ME   180 angle
        if np.array_equal(goal_in, [3.79, -6.99]) and self.dist_between(x_cur_in, goal_in) < .36:
            return 1.0

        # if np.array_equal(goal_in, [7.41, -6.99]) and goal_in dist_between(x_cur_in, goal_in) < .36:
        #     return 1.0


        # if not np.array_equal(x_cur_in, x_prev):
        #     x_theta = self.get_heading_of_pt_diff_p2_p1(x_cur[:2], x_prev[:2])
        # else:
        #     x_theta = None
        #     return (1.0 / len(self.goals))

        # print("PRECALC heading angle")
        # print(x_cur, x_prev, x_theta)

        # x_triplet = [x_cur[0], x_cur[1], x_theta]

        u_output = u
        # heading_P_oa = self.prob_heading_3d(x_triplet, u_output, i_in, goal_in, visibility_coeff, override=override)

        # Version that uses pt differences
        heading_P_oa_4d = self.prob_heading_from_pt_seq_alt_4d(x_cur, x_prev, u_output, i_in, goal_in, visibility_coeff, override=override)

        print("HEADCOMP: " + str(heading_P_oa_4d))
        # print("HEADCOMP: " + str(heading_P_oa) + " vs " + str(heading_P_oa_4d))

        # If you did nothing and stood in the same spot, 
        # then min probability, maximum penalty

        return heading_P_oa_4d

    # This function only takes size 3 vectors
    def prob_heading_from_pt_seq_alt_4d(self, x_cur, x_prev, u_in, i_in, goal_in, visibility_coeff, override=None):
        all_goals   = self.goals
        goal        = goal_in[:2]

        mode_dist    = self.exp.get_mode_type_dist()
        mode_heading = self.exp.get_mode_type_heading()
        mode_blend   = self.exp.get_mode_type_blend()

        if override != None:
            if 'mode_heading' in override.keys():
                mode_heading = override['mode_heading']
                print("OO: override to do heading mode")

        if u_in is not None:
            u = copy.copy(u_in)
        else:
            u = None
        i = copy.copy(i_in)

        if override is not None:
            if 'mode_heading' in override:
                mode_dist = override['mode_heading']

        if not np.array_equal(goal[:2], self.exp.get_target_goal()[:2]):
            print("WARNING: Goal and exp goal not the same in prob heading")
            print(goal[:2], self.exp.get_target_goal()[:2])
            # exit()

        x_total = np.concatenate((x_cur, x_prev))
        debug_dict = {'x': x_total, 'u': u, 'i':i, 'goal': goal, 'start': self.exp.get_start(), 'all_goals':self.exp.get_goals(), 'visibility_coeff': visibility_coeff, 'N': self.exp.get_N(), 'override': override, 'mode_heading': mode_heading}
        
        print("HEADING EFFORT COST INPUTS")
        print(debug_dict)

        target_vector           = None
        all_goal_headings       = []

        target_index = -1
        x       = x_cur[:2]
        x_prev  = x_prev

        all_effort_measures         = []
        for j in range(len(self.goals)):
            # print("Goal angle diff for " + str(robot_theta) + " -> " + str(ghead))
            # goal_angle_diff  = self.get_min_rotate_angle_diff(robot_theta, ghead)
            # print("goal angle diff " + str(goal_angle_diff))
            # effort_made = (180.0 - goal_angle_diff)
            # print("effort made " + str(effort_made))

            alt_goal = all_goals[j]
            # goal_vector = alt_goal - x_current
            # all_goal_vectors.append(goal_vector)

            if np.array_equal(alt_goal[:2], goal[:2]):
                print("Yes, is target")
                target_index = j

            # effort_made = self.get_angle_between_triplet(x_prev, x, alt_goal)

            angle_to_look = self.get_angle_to_look_at_point(x_cur, x_prev, alt_goal)
            effort_made = (180.0 - angle_to_look)

            if angle_to_look < 0:
                print(goal_angle_diff)
                print("eeek negative angle diff")

            print("Angle to look at goal " + str(alt_goal) + " is " + str(angle_to_look))
            print("Effort is " + str(effort_made))

            # if (effort_made) > 180 or (effort_made) < 0:
            #     print("ALERT: Get angle between needs some work")
            #     # exit()

            if mode_heading == 'sqr':
                effort_made = effort_made**2
            elif mode_heading == 'exp':
                effort_made = np.exp(effort_made)

            all_effort_measures.append(effort_made)

        print("All effort measures")
        print(all_effort_measures)

        target_val  = all_effort_measures[target_index]
        total       = sum(all_effort_measures)
        num_goals   = len(all_goals)

        try:
            all_probs = [x/total for x in all_effort_measures]
            print(all_probs)

            P_heading   = ((target_val) / total)
        except (ValueError, decimal.InvalidOperation, ZeroDivisionError):
            print("Alert: Heading has divide by 0")
            P_heading = (1.0 / num_goals)
            P_heading = 0.0


        P_oa        = ((1.0/num_goals)*(1.0 - visibility_coeff)) + (((visibility_coeff) * P_heading))
        return P_oa

    # This function only takes size 3 vectors
    def prob_heading_3d(self, x_triplet, u_in, i_in, goal_in, visibility_coeff, override=None):
        all_goals   = self.goals
        goal        = goal_in[:2]

        mode_dist    = self.exp.get_mode_type_dist()
        mode_heading = self.exp.get_mode_type_heading()
        mode_blend   = self.exp.get_mode_type_blend()

        if override != None:
            if 'mode_heading' in override.keys():
                mode_heading = override['mode_heading']
                print("OO: override to do heading mode")

        if u_in is not None:
            u = copy.copy(u_in)
        else:
            u = None
        i = copy.copy(i_in)

        if override is not None:
            if 'mode_heading' in override:
                mode_dist = override['mode_heading']

        x_current = x_triplet[:2]

        debug_dict = {'x': x_triplet, 'u': u, 'i':i, 'goal': goal, 'start': self.exp.get_start(), 'all_goals':self.exp.get_goals(), 'visibility_coeff': visibility_coeff, 'N': self.exp.get_N(), 'override': override, 'mode_heading': mode_heading}
        print("HEADING COST INPUTS")
        print(debug_dict)

        if x_triplet[2] == None:
            print("Robot theta not yet set in theta solve mode, is that a problem?")
            return (0.0)
        else:
            robot_theta = x_triplet[2]

        if not np.array_equal(goal[:2], self.exp.get_target_goal()[:2]):
            print("Goal and exp goal not the same in prob heading")
            print(goal, self.exp.get_target_goal())
            # exit()


        # # if we are at the goal, we by definition are arriving correctly
        # if self.dist_between(x_current, goal) < 1.1:
        #     P_heading   = (1.0)

        #     num_goals   = len(all_goals)
        #     P_oa        = ((1.0/num_goals)*(1.0 - visibility_coeff)) + (((visibility_coeff) * P_heading))
        #     return P_oa

        target_vector           = None
        all_goal_headings       = []

        target_index = -1

        for j in range(len(all_goals)):
            alt_goal = all_goals[j]
            # goal_vector = alt_goal - x_current
            # all_goal_vectors.append(goal_vector)

            if np.array_equal(alt_goal[:2], goal[:2]):
                print("Yes, is target")
                target_index = j
            else:
                print("no, mismatch of " + str(alt_goal) + " != " + str(goal))

            goal_heading = self.get_heading_of_pt_diff_p2_p1(alt_goal, x_current)
            all_goal_headings.append(goal_heading)

        print("All goal headings")
        print(all_goal_headings)

        all_effort_measures         = []
        all_offset_angles           = []
        for ghead in all_goal_headings:
            print("Goal angle diff for " + str(robot_theta) + " -> " + str(ghead))
            goal_angle_diff  = self.get_min_rotate_angle_diff(robot_theta, ghead)
            print("goal angle diff " + str(goal_angle_diff))
            effort_made = (180.0 - goal_angle_diff)
            print("effort made " + str(effort_made))

            # effort_two = self.get_angle_between_triplet(a, b, c)

            # if (effort_made) > 180 or (effort_made) < 0:
            #     print("ALERT: Get angle between needs some work")
            #     # exit()
            if goal_angle_diff < 0:
                print(goal_angle_diff)
                print("eeek negative angle diff")
                # exit()

            if mode_heading == 'sqr':
                effort_made = effort_made**2

            all_offset_angles.append(goal_angle_diff)
            all_effort_measures.append(effort_made)

        print("All angle vectors to goals")
        print(all_offset_angles)
        print("All effort measures")
        print(all_effort_measures)

        target_val  = all_effort_measures[target_index]
        total       = sum(all_effort_measures)

        try:
            all_probs = [x/total for x in all_effort_measures]
            print(all_probs)

            P_heading   = (target_val) / total
        except (ValueError, decimal.InvalidOperation):
            print("Error! ...")
            P_heading = (1.0 / len(all_probs))


        num_goals   = len(all_goals)
        P_oa        = ((1.0/num_goals)*(1.0 - visibility_coeff)) + (((visibility_coeff) * P_heading))

        return P_oa

    def get_robot_vector(self, x, i):
        x1 = x
        if i > 0:
            x0 = self.x_path[i - 1]
        else:
            x0 = x

        print("Points in a row")
        print(x0, x1)

        if x1 == x0:
            print("Robot has no diff, so no heading")
            print(i)
            while j > 0 and x1 == x0:
                j = j - 1
                x0 = self.x_path[j - 1]

        if x1 == x0:
            print("Still a problem here with heading diff")

        return x1 - x0


    def get_heading_cost_v2_wraps(self, x, u, i, goal):
        if i == 0:
            return 0

        goals       = self.goals

        robot_vector    = self.get_robot_vector(x, i)
        target_vector   = None
        all_goal_vectors    = []

        for alt_goal in goals:
            goal_vector = alt_goal - x #[x1, alt_goal]

            if alt_goal is goal:
                target_vector = goal_vector
            all_goal_vectors.append(goal_vector)

        print("robot vector")
        print(robot_vector)

        print("all goal vectors")
        print(all_goal_vectors)

        all_goal_angles   = []
        for gvec in all_goal_vectors:
            goal_angle = self.get_angle_between(robot_vector, gvec)
            all_goal_angles.append(goal_angle)

        target_angle = self.get_angle_between(robot_vector, target_vector)

        print("all target angles")
        print(all_goal_angles)

        angles_squared = []
        for i in range(len(all_goal_angles)):
            gangle = all_goal_angles[i]
            gang_sqr = gangle * gangle

            if self.exp.get_weighted_close_on() is True:
                k = self.get_relative_distance_k(x, goals[i], goals)
                # k = k*k
            else:
                k = 1.0

            angles_squared.append(k * gang_sqr)

        target_angle_sqr = target_angle * target_angle

        total = sum(angles_squared)

        print("total")
        print(total)
        print("target_angle")
        print(target_angle)
        print("target angle sqr")
        print(target_angle_sqr)

        denominator = (180*180) * len(all_goal_angles)

        heading_clarity_cost = (total - target_angle_sqr) / (total)
        # heading_clarity_cost = (total - target_angle_sqr) / (denominator)
        # alt_goal_part_log = alt_goal_part_log / (total)

        print("Heading component of pathing ")
        print("Given x of " + str(x) + " and robot vector of " + str(robot_vector))
        print("for goals " + str(goals))
        # print(alt_goal_part_log)
        print(heading_clarity_cost)
        # print("good parts, bad parts")
        # print(good_part, bad_parts)

        return heading_clarity_cost


    # How far away is the final step in the path from the goal?
    # Note we only constrain the xy, not the theta
    def term_cost(self, x_triplet, i):
        start = self.start
        goal1 = self.target_goal
        print("TERM GOAL IS " + str(self.target_goal))

        x               = x_triplet[:2]
        squared_x_cost  = self.get_x_diff(x, i)

        print("IS TERM EVIL?")
        print(x_triplet, self.x_path[i], squared_x_cost)

        terminal_cost = squared_x_cost

        if terminal_cost is np.nan:
            print("Alert: Terminal cost of nan, but how?")
            print(x)
            print(x_diff)
            print(self.x_path)
            terminal_cost = np.inf

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("term cost squared x cost")
            print(squared_x_cost)
            pass

        FLAG_CUTOFF_AFTER_AT_GOAL = False
        if FLAG_CUTOFF_AFTER_AT_GOAL:
            # 1/k of the min step size is the catch distance of the goal
            k = 10.0
            actual_goal_dist = np.linalg.norm(x - goal1)
            sufficient_goal_dist = (np.linalg.norm(start - goal1)) / (self.exp.get_N() * k)
            print("sufficient_goal_dist = " + str(sufficient_goal_dist))
            print("actual_goal_dist = " + str(actual_goal_dist))

            if actual_goal_dist < sufficient_goal_dist:
                terminal_cost = 0.0
                print("POP: WITHIN SUFFICIENT GOAL DIST")


        # We want to value this highly enough that we don't not end at the goal
        # terminal_coeff = 100.0
        coeff_terminal = self.exp.get_solver_scale_term()
        terminal_cost = terminal_cost * coeff_terminal

        print("actual terminal cost")
        print(terminal_cost)

        # Once we're at the goal, the terminal cost is 0
        
        # Attempted fix for paths which do not hit the final mark
        # if squared_x_cost > .001:
        #     terminal_cost *= 1000.0

        return terminal_cost

    def get_x_diff_between_pts(self, x, i):
        u_calc = x[:2] - x[2:]
        R = np.eye(2)
        val_u_diff  = u_calc.T.dot(R).dot(u_calc)

        return val_u_diff


    def get_u_diff(self, x, u, i):
        u_calc = x[:2] - x[2:]
        # u = u_calc

        # if np.array_equal(u_calc, u):
        #     print("NA: U calc as promised!")
        # else:
        #     print("ALERT: U is wrong?")
        #     print(x, u, u_calc)
        #     print(x[:2], x[2:])
           
        try: 
            if u.any() == None:
                return 0.0
        except:
            return 0.0

        # print("incoming " + str(u))
        R = np.eye(2)
        
        u_diff      = np.abs(u - self.u_path[i])
        # u_diff      = np.abs(u - np.asarray([0, 0]))
        val_u_diff  = np.dot(u_diff.T, R)
        val_u_diff  = np.dot(val_u_diff, u_diff)

        if u[0] is np.nan or u[1] is np.nan:
            print("FLAT PENALTY FOR NANS")
            return 0.0 #val_u_diff * 10000.0

        print("udiff calc")
        print(u, "-", self.u_path[i], u_diff, val_u_diff)
        print(u_calc)

        return val_u_diff

    # def get_u_diff_heading(self, x, u, i):
    #     u_calc = x[:2] - x[2:]
    #     # u = u_calc

    #     # if np.array_equal(u_calc, u):
    #     #     print("NA: U calc as promised!")
    #     # else:
    #     #     print("ALERT: U is wrong?")
    #     #     print(x, u, u_calc)
    #     #     print(x[:2], x[2:])
            
    #     if u.any() == None:
    #         return 0.0

    #     # print("incoming " + str(u))
    #     R = np.eye(2)
    #     u_diff      = np.abs(u - self.u_path[i])
    #     # u_diff      = np.abs(u - np.asarray([0, 0]))

    #     val_u_diff  = u_diff.T.dot(R).dot(u_diff)

    #     if u[0] is np.nan or u[1] is np.nan:
    #         print("FLAT PENALTY FOR NANS")
    #         return 0.0 #val_u_diff * 10000.0

    #     print("udiff calc")
    #     print(u, "-", self.u_path[i], u_diff, val_u_diff)

    #     return val_u_diff

    def get_x_diff(self, x_input, i):
        x_ref   = np.asarray([self.x_path[i][0], self.x_path[i][1]])
        x_ref   = self.exp.get_target_goal()[:2]
        x_cur   = np.asarray([x_input[0], x_input[1]])
        x_diff  = x_cur - x_ref
        print("xdiff detail")
        print("x_input, x_ref, x_cur, x_diff")
        print(x_input, x_ref, x_cur, x_diff)

        Q = np.eye(2)
        squared_x_cost = .5 * x_diff.T.dot(Q).dot(x_diff)

        print(squared_x_cost)
        # print("xdiff")
        # print(x_cur, x_ref, x_diff, squared_x_cost)

        return squared_x_cost

    def get_stage_components_per_goal(x):
        return


    # original version for plain path following
    def l(self, input_x, input_u, input_i, terminal=False, just_term=False, just_stage=False, test_component=None):
        """Instantaneous cost function.
        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.
        Returns:
            Instantaneous cost (scalar).
        """
        print("77777777777777")
        print("L: ")
        print("x, u, i")
        print(input_x, input_u, input_i)

        Q = self.Q #self.Qf if terminal else self.Q
        R = self.R
        start   = self.start
        goal    = self.target_goal
        thresh  = .0001
        target_goal    = self.target_goal

        u = copy.copy(input_u)
        x = copy.copy(input_x)
        i = copy.copy(input_i)
       
        # print("INPUT U IS " + str(u))
        # if u is None:
        #     u = np.asarray([np.inf, np.inf])

        # if not isinstance(u, (list, np.ndarray)): # and np.isnan(u):
        #     print("Alert: u is not a list")
        #     return np.inf
        # if not isinstance(x, (list, np.ndarray)): # and np.isnan(x):
        #     print("Alert: x is not a list")
        #     return np.inf

        try:
            # if u[0] == np.nan:
            #     print("Alert: u has a nan")
            #     return np.inf
            if x[0] is np.nan:
                print("Alert: x has a nan")
                return np.inf
        except:
            print("Alert: issue with input format")
            return np.inf


        # if goal[:2] == input_x[:2]:
        #     return 1.0

        scale_term  = self.exp.get_solver_scale_term() #0.01 # 1/100
        scale_stage = self.exp.get_solver_scale_stage() #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0

        # # xdiff from preferred line
        # x_path[i] is always the goal
        # x_diff = x - self.x_path[i]

        if terminal or just_term: #abs(i - self.N) < thresh or
            term_cost = self.term_cost(x, i)
            return scale_term * self.term_cost(x, i)
        else:
            if self.FLAG_DEBUG_STAGE_AND_TERM and just_stage != True:
                term_cost = self.term_cost(x, i)

                # difference between this step and the end
                print("x, N, x_end_of_path -> inputs and then term cost")
                print(x, self.N, self.x_path[self.N])
                # term_cost = self.term_cost(x, i)
                print(term_cost)

        # VISIBILITY COMPONENT
        restaurant  = self.exp.get_restaurant()
        observers   = self.exp.get_observers()

        squared_x_cost = 0 #self.get_x_diff_between_pts(x, i)
        squared_u_cost = self.get_u_diff(x, u, i)

        val_angle_diff  = 0


        print("LE GOAL")
        print(goal)
        ### USE ORIGINAL LEGIBILITY WHEN THERE ARE NO OBSERVERS
        is_vis_target, is_vis_secondary           = self.exp.get_visibility_of_all(x)
        islocal_target, islocal_secondary         = self.exp.get_is_local_of_all(x)

        # ADA NEXTTODO
        # cost_dict = self.get_costs_relative_to_goals()

        ###### SET VALUES SET WT
        wt_legib        = 1.0
        wt_lam          = 10.0 #25 #0.125 #5 #25 # 1.0   #* (1.0 / self.exp.get_dt()) this should really be N if anything
        wt_heading      = 1.0
        wt_obstacle     = 1.0   #self.exp.get_solver_scale_obstacle()

        wt_understanding_target     = 1.0
        wt_understanding_secondary  = 1.0

        val_legib       = 0
        val_lam         = 0
        val_obstacle    = 0
        val_heading     = 0
       
        val_understanding_target     = 0
        val_understanding_secondary  = 0

        if_seen = 1.0

        # Get the values if necessary
        if True: #wt_legib > 0:
            # val_legib        = self.get_legibility_dist_stage_cost(start, goal, x, u, i, terminal, if_seen)
            val_legib        = self.get_legibility_component_og(start, goal, x, i, raw=True)
        
        if True: #wt_heading > 0:
            if self.exp.get_state_size() < 4:
                val_heading     = self.get_legibility_heading_stage_cost_3d(x, u, i, goal, if_seen)
            else:
                x_new   = x[:2]
                x_prev  = x[2:]
                print("Getting heading for " + str(x_new) + ' -> ' + str(x_prev) + " from point " + str(x))
                val_heading     = self.get_legibility_heading_stage_cost_from_pt_seq(x_new, x_prev, i, goal, if_seen)

        if wt_lam > 0:
            val_lam         = squared_u_cost #+ (squared_x_cost)
        else:
            print("ALERT: why is lam weight 0?")

        exp_test = False
        if exp_test:
            val_heading = np.exp(val_heading)
            val_legib = np.exp(val_legib)

        if wt_obstacle > 0:
            val_obstacle    = self.get_obstacle_penalty(x, i, goal)

        ######################################
       
        mode_blend                  = self.exp.get_mode_type_blend()
        understanding_target        = 'local'
        understanding_secondary     = 'local'

        target_costs        = 0
        secondary_costs     = []

        max_legib_value     = 2 # max 1 for heading, and 1 for position

        num_factors = len(['legib', 'heading'])
        num_factors = 1
        max_not_vis_penalty     = num_factors * len(self.exp.get_observers())
        max_not_local_penalty   = num_factors * len(self.exp.get_observers())
        max_penalty             = num_factors * len(self.exp.get_observers()) * 5 # * (self.exp.get_N() - i)

        ###### Note: Needs updating for 3 target scenarios
        # if mode_blend == 'mixed':
        #     target_costs        = val_legib + val_heading
        #     secondary_costs     = (1.0 - val_legib) + (1.0 - val_heading)
        # elif mode_blend == 'min':

        # HARDCODED
        # val_heading = 0

        # Take the max of the two values
        if False:
            # ADA TODO check
            target_costs        = min(val_legib, val_heading)

            for i in range(len(self.exp.get_secondary_observers())):
                # sec_val_legib       = (1.0 - val_legib)
                # sec_val_heading     = (1.0 - val_heading)
                # secondary_cost     = max(sec_val_legib, sec_val_heading)

                secondary_costs.append(secondary_cost)

        elif True:
            # p_d = self.prob_distance(start, goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp', 'mode_blend':None})
            # p_d = self.prob_distance(start, goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp_raw', 'mode_blend':None})
            # p_d = self.prob_distance(start, target_goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp', 'mode_blend':None})
            # p_d = self.prob_distance(start, goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp', 'mode_blend':None})
        
            if False:
                if len(input_x) == 4:
                    p_h = self.prob_heading_from_pt_seq(x, x_prev, i, goal, True, override={'mode_heading':'lin', 'mode_dist':None, 'mode_blend':None})
                else:
                    p_h = self.prob_heading(x, u, i, goal, True, override={'mode_heading':'lin', 'mode_dist':None, 'mode_blend':None})
                    # TODO get max p_h
                    # Compare to the current p_h

            goals = self.exp.get_goals()
            P_d_dict = {}
            for maybe_goal in goals:
                prob = self.prob_distance(start, maybe_goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp', 'mode_blend':None})
                # prob = self.get_legibility_component_new(start, maybe_goal, x, i, raw=True)
                # cost = self.get_legibility_cost_ada(start, maybe_goal, x, i, raw=True)

                # prob = self.prob_distance(start, maybe_goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp_local', 'mode_blend':None})

                # cost = self.get_legibility_component_ada2(start, maybe_goal, x, i, raw=True)
                cost = prob


                key = (maybe_goal[0], maybe_goal[1])
                P_d_dict[key] = cost

            cost_d_dict = P_d_dict
            print("Relative costs")
            print(cost_d_dict)

            ###### SINGLE VANILLA CASE
            if True and 50 == 50:
                key = (goal[0], goal[1])
                cost_d = cost_d_dict[key]   #self.prob_distance(start, goal, x, u, i, terminal, True, override={'mode_heading':None, 'mode_dist':'exp', 'mode_blend':None})
                # p_d_target, p_alts = self.get_legibility_component_alts(start, goal, x, u, i, terminal, True, override_block={'mode_heading':None, 'mode_dist':'exp', 'mode_blend':None})
            
                # exp_cost

                FLAG_SECONDARY_CONSIDERED = True
                
                # closeness_scalar = self.get_relative_distance_value(i, start, goal, x, terminal, 'lin')

                val_overall = (1.0 - cost_d) #(1.0 - p_d) # * closeness_scalar #+ max(p_alts)
                max_penalty = 3.0 #0.0 #10.0 #(1.0 - (p_d * .01)) #0.0 #np.exp(2.0)
                wt_lam = 1.0


            target_costs        = val_overall # + val_heading
            # max_penalty         = 10.0
            secondary_costs     = 0


        val_understanding_secondary = 0

        if FLAG_SECONDARY_CONSIDERED:
            status_dict = self.exp.get_vislocal_status_of_point(x)
            for alt_goal in self.exp.get_goals():
                if (self.exp.get_target_goal() != alt_goal):
                    vis, local, vis_angle, dist = status_dict[(alt_goal[0], alt_goal[1])]

                    # np.exp
                    offset = (alt_goal - x[:2]) / self.exp.get_local_distance()
                    # print("offset percent " + str(offset))
                    cost_falloff    = np.exp((np.dot(np.dot(offset.T, np.eye(2)), offset))) #self.get_relative_distance_value(i, start, alt_goal, x, False, 'lin')
                    val_secondary   = 0 #P_d_dict[key] * (cost_falloff) #* .001

                    target_key = (goal[0], goal[1])
                    if vis and local:
                        key = (alt_goal[0], alt_goal[1])

                        k = 2.0
                        # This creates divide by 0 errors
                        # val_secondary = cost_d_dict[key] / (cost_d_dict[key] + cost_d_dict[target_key] + .000001) #* cost_falloff # * .25 # * 10.0

                        val_secondary = cost_d_dict[key] #- cost_d_dict[target_key]

                        print("Secondary costs at " + str(alt_goal))
                        print(P_d_dict[key], cost_falloff)
                        # else:
                        #     print("Not secondary: this is the target")

                    elif local and not vis:
                        val_secondary = 0 #1.0 # * cost_falloff
                    else:
                        print("not relevant secondary at " + str(alt_goal))

                    val_understanding_secondary += val_secondary #* 2.0 # * cost_falloff #* 2.0

        val_understanding_secondary = (val_understanding_secondary)


        ###### UNDERSTANDING COSTS: TARGET
        ### Depending on mode taken from the exp packet...

        #### Also needs update for multi-goal
        # TARGET
        if False:
            if islocal_target and is_vis_target:
                val_understanding_target = target_costs
                print("IS LOCAL == YES")
            elif islocal_target and not is_vis_target:
                val_understanding_target = max_penalty #(target_costs + max_penalty) / 2.0
            else:
                wt_lam = wt_lam
                val_understanding_target = max_penalty
                print("IS LOCAL == NO == " + str(is_vis_target) + "---" + str(islocal_target))


                # Tweak for walking up centerline
                if False:
                    val_understanding_target = np.exp((.5 - p_d)**2)  + max_penalty
        else:
            val_understanding_target = target_costs



        wt_legib     = (wt_legib)
        wt_lam       = (wt_lam)
        wt_heading   = (wt_heading)
        wt_obstacle  = (wt_obstacle)

        val_legib     = (val_legib)
        val_lam       = (val_lam)
        val_heading   = (val_heading)
        val_obstacle  = (val_obstacle)


        if (val_legib) < 0 or (val_lam) < 0 or (val_heading) < 0 or (val_obstacle) < 0:
            print("ALERT: NEGATIVE COST")


        # # Pathing with taking the max penalty, ie min probability
        # if wt_legib > 0 and wt_heading > 0:
        #     if (val_legib) > (val_heading):
        #         max_val = val_legib
        #     else:
        #         max_val = val_heading

        #     val_legib   = max_val
        #     val_heading = max_val
        #     wt_legib    = (.5) * wt_legib
        #     wt_heading  = (.5) * wt_heading


        # J does not need to be in a particular range, it can be any max or min
        J = 0        
        # J += wt_legib       * val_legib     #self.legibility_stage_cost(start, goal, x, u, i, terminal, visibility_coeff)
        # J += wt_heading     * val_heading   #self.get_heading_cost(x, u, i, goal, visibility_coeff)

        J += wt_understanding_target    * val_understanding_target
        J += wt_understanding_secondary * val_understanding_secondary


        J += wt_lam         * val_lam           #u_diff.T.dot(R).dot(u_diff)
        # J += wt_lam_h       * val_lam_h         #u_diff.T.dot(R).dot(u_diff)
        J += wt_obstacle    * val_obstacle      #self.get_obstacle_penalty(x, i, goal)

        # stage_costs = sum([wt_legib*val_legib, wt_lam*val_lam, wt_heading*val_heading, wt_obstacle*val_obstacle])
        # stage_costs = (stage_costs)

        stage_costs = J

        # if stage_costs != J:
        #     print("alert! j math is off")
        #     print("J = " + str(J))
        #     print(stage_costs)

        # if self.FLAG_DEBUG_STAGE_AND_TERM:
        #     print("STAGE,\t TERM")
        #     print(stage_costs, term_cost)

        #     print("[wt_legib, wt_lam, wt_heading, wt_obstacle]")
        #     print([str(wt_legib), str(wt_lam), str(wt_heading), str(wt_obstacle)])            
        #     print("[val_legib, val_lam, val_heading, val_obstacle]")
        #     print([str(val_legib), str(val_lam), str(val_heading), str(val_obstacle)])
        #     print("==")
        #     print([str(wt_legib*val_legib), str(wt_lam*val_lam), str(wt_heading*val_heading), str(wt_obstacle*val_obstacle)])

        #     print(str(sum([wt_legib*val_legib, wt_lam*val_lam, wt_heading*val_heading, wt_obstacle*val_obstacle])))
        #     print("~~~~~~~~~~~~")

        # Don't return term cost here ie (scale_term * term_cost) 
        total = (scale_stage * stage_costs)

        if test_component == 'legib':
            # NOTE: this is just for up close
            return (wt_understanding_target*val_understanding_target)
        elif test_component == 'lam':
            return wt_lam*val_lam
        elif test_component == 'head':
            return wt_heading*val_heading
        elif test_component == 'obs':
            return wt_obstacle*val_obstacle

        return (total)

    def f(t):
        return 1.0

    def dist_between(self, x1, x2):
        # print(x1)
        # print(x2)
        # print(x1[0], x2[0], x1[1], x2[1])

        distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        return distance

    def logsumexp(self, x):
        c = np.max(x)
        total = 0
        for x_val in x:
            total += np.exp(x_val - c)

        return c + np.log(total)

    # # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # def small_value_norm(self, values):
    #     norm_array = []
    #     logsumexp_norm = self.logsumexp(values)
    #     print("logsumexp numerator")
    #     print(logsumexp_norm)
    #     # print(np.logsumexp(values))

    #     # np.exp(x - logsumexp(x))
    #     for x in values:
    #         v = np.exp(x - logsumexp_norm)
    #         norm_array.append(v)

    #     return norm_array

    def small_value_norm(self, values):
        if all(a == 0 for a in values):
            values = [1.0 / len(values) for i in values]

        smallest = np.min([i for i in values if i != 0])

        order = (-np.log(smallest)) # log1p can add to the precision
        if order > 0:
           order = 0
        sf = np.exp(order)
        scaled = [x * sf for x in values]
        tot = sum(scaled)
        norm = [x/tot for x in scaled]

        return norm

    def small_value_norm_v1(self, values):
        no_z = [i for i in values if i != 0]
        no_z = [i for i in values if i != np.nan]
        print(values, no_z)

        # if len(no_z) < len(values):
        #     new_values = []
        #     for v in values:
        #         if v == 0:
        #             new_val = 0
        #         else:
        #             new_val = 1.0 / len(no_z)
        #         new_values.append(new_val)

        #     return new_values

        if len(no_z) == 0:
            if len(values)== 0:
                print("Alert: values has length of 0")
            return [1.0/len(values) for i in values]

        smallest = np.min(no_z)
        no_z_max = np.max(no_z)
        print(no_z, smallest, no_z_max)
        biggest_index = no_z.index(no_z_max)
        # print("smallest")
        # print(smallest)

        if smallest < 10e-160:
            values = [i * 10e160 for i in values]

        # This is about the point where divide by 0 issues happen
        if smallest < 10e-160:
            print("Alert: Smallest caught")
            new_values = []
            for v in values:
                new_values.append(0.0)

            new_values[biggest_index] = 1.0
            # print("v small smallest")
            return new_values

        order = (-np.log10(smallest))
        # print(order)
        if order > 0:
           order = 0
        sf = 10**order
        scaled = [x * sf for x in values]
        tot = sum(scaled)
        # print("tot")
        # print(tot)
        norm = [x/tot for x in scaled]

        return norm

    def get_estimated_cost(self, distance):
        FLAG_SCALE_FOR_VARIANCE = True
        k = 1.0
        if FLAG_SCALE_FOR_VARIANCE:
            # k is the step size of the expected smallest step
            k = self.exp.get_dist_scalar_k()
            print("scaling by ")
            print(k)

        # if num_steps == 0:
        #     return 0

        # est_dist = distance / num_steps
        # est_dist = num_steps * (est_dist * est_dist)

        est_dist = distance * k

        print("Est dist")
        print(distance, k, est_dist)

        return est_dist


    def get_relative_distance_value(self, i_step, start, goal_in, x_input, terminal, mode_dist):
        Q       = np.eye(2) #self.Q_terminal if terminal else self.Q
        x       = x_input[:2]
        goal    = goal_in
        dist    = self.dist_between(x, goal_in)

        dist_linalg = np.linalg.norm(x - goal_in)

        if not np.array_equal(dist, dist_linalg):
            print("WARN: LINALG IS DIFFERENT")
            print("dist, dist_linalg")
            print(dist, dist_linalg)

        print("dist to the goal at <" + str(goal) + ">")
        print(dist)
        if dist < 0:
            print("dist to goal less than 0!")
            exit()

        if mode_dist == 'sqr':
            val = (self.inversely_proportional_to_distance(dist)**2)
            print("mode dist is sqr, dist inv value is " + str(val))
            return val
            # return (self.get_relative_distance_k_sqr(x, goal, self.goals))
        elif mode_dist == 'lin':
            val = (self.inversely_proportional_to_distance(dist))
            print("mode dist is lin, dist inv value is " + str(val))
            return val
            # return (self.get_relative_distance_k(x, goal, self.goals))

        elif mode_dist == 'lin_exp':
            # val = (np.exp(self.inversely_proportional_to_distance(dist)))
            # print("mode dist is lin_exp, dist inv value is " + str(val))
            val = self.get_legibility_component_og(start, goal, x, i_step)
            return val

        elif mode_dist == 'exp':
            J = self.get_legibility_component_og(start, goal, x, i_step)

        elif mode_dist == 'exp_local':
            J = self.get_stage_cost_if_vis(start, goal, x, i_step)

        elif mode_dist == 'exp_raw':
            J = self.get_legibility_component_og(start, goal, x, i_step, raw=True)

        elif mode_dist == 'loc_exp':
            J = self.get_legibility_component_new(start, goal, x, i_step, raw=True)

        else:
            J = np.abs(n / d)
            print("ALERT: UNKNOWN MODE = " + str(mode_dist))

        if self.exp.get_weighted_close_on() is True:
            k = self.get_relative_distance_k(x, goal, self.goals)
        else:
            k = 1.0

        print("mode dist is exp, dist inv value is " + str(J))

        J = (k)*J
        print("J of exp")
        print(J)
        return J

    # def legibility_stage_cost_wrapper(self, start, goal, x, u, i, terminal, visibility_coeff, force_mode=None, pure_prob=False):
    #     # print("Compare between modes")
    #     # prob_exp = self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode='exp', pure_prob=True)
    #     # prob_sqr = self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode='sqr', pure_prob=True)
    #     # prob_lin = self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode='lin', pure_prob=True)

    #     # print("PROBS")
    #     # print(prob_exp, prob_sqr, prob_lin)

    #     # pen_exp = self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode='exp', pure_prob=False)
    #     # pen_sqr = self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode='sqr', pure_prob=False)
    #     # pen_lin = self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode='lin', pure_prob=False)

    #     # print("PENALTIES")
    #     # print(pen_exp, pen_sqr, pen_lin)

        
    #     return self.legibility_stage_cost_helper(start, goal, x, u, i, terminal, visibility_coeff, force_mode=force_mode, pure_prob=pure_prob)


    def get_legibility_component_togoal(self, start, goal, x, i_step, raw=False):
        Q       = np.eye(2)

        # goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        # togoal_diff = (np.array(x) - goal)

        # diff_curr   = start - x
        diff_goal   = x - goal
        # diff_all    = start - goal

        # diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
        diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
        # diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

        total_steps = self.exp.get_N()

        # diff_curr_v = self.get_estimated_cost(diff_curr_v, i_step)
        # diff_goal_v = self.get_estimated_cost(diff_goal_v, self.exp.get_N() - i_step)
        # diff_all_v  = self.get_estimated_cost(diff_all_v, self.exp.get_N())

        # n = - (diff_curr_v) - (diff_goal_v)
        # d = diff_all_v

        value = diff_goal_v

        # J = np.exp(n) / np.exp(d)

        return J

    def get_legibility_component_relative_miscue(self, start, alt_goal, goal, x, i_step, raw=False):
        Q       = np.eye(2)

        diff_good   = x - alt_goal
        diff_bag    = x - goal
        diff_all    = start - goal

        diff_good_v = np.dot(np.dot(diff_good.T,    Q),    diff_good)
        diff_bad_v  = np.dot(np.dot(diff_bad.T,     Q),     diff_bad)
        diff_all_v  = np.dot(np.dot(diff_all.T,     Q),     diff_all)

        # total_steps = self.exp.get_N()
        # diff_curr_v = self.get_estimated_cost(diff_curr_v, i_step)
        # diff_goal_v = self.get_estimated_cost(diff_goal_v, self.exp.get_N() - i_step)
        # diff_all_v  = self.get_estimated_cost(diff_all_v, self.exp.get_N())

        # n = - (diff_curr_v) - (diff_goal_v)
        # d = diff_all_v

        J = np.exp(diff_bad_v - diff_good_v)

        # J = np.exp(n) / np.exp(d)

        return J

    def get_legibility_component_ada2(self, start, true_goal, x, i_step, raw=False):
        Q       = np.eye(2)
        x       = x[:2]

        goal = true_goal

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = start - goal

        diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
        diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
        diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

        total_steps = self.exp.get_N()

        diff_curr_v = self.get_estimated_cost(diff_curr_v) #, i_step)
        diff_goal_v = self.get_estimated_cost(diff_goal_v) #, self.exp.get_N() - i_step)
        diff_all_v  = self.get_estimated_cost(diff_all_v) #, self.exp.get_N())

        n = - (diff_goal_v) - diff_curr_v
        d = diff_all_v

        alt_goal_costs  = []
        correct_goal    = None

        for g in self.exp.get_goals():
            diff_goal   = x - g
            diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
            diff_goal_v = self.get_estimated_cost(diff_goal_v)

            if g == true_goal:
                correct_goal = diff_goal_v
            else:
                alt_goal_costs.append(diff_goal_v)


        print("CUSTOM all goals")
        print(self.exp.get_goals())
        print(true_goal)

        if correct_goal == None:
            print("Dead goal")
            exit()


        J = sum(alt_goal_costs) - correct_goal
        J = np.exp(J) / np.exp(d)

        # J = np.exp(n) / np.exp(d)


        # xmag, ymag = diff_goal
        # goal_dist = np.sqrt(xmag**2 + ymag**2)

        # J = goal_dist / self.exp.get_local_distance()

        # J = np.exp(J)
        return J

    def get_stage_cost_if_vis(self, start, input_goal, x, i_step):
        # function stage_cost(x,u,xref,uref, start, goal1, all_goals, nongoal_scale)
        # Legibility LQR cost at each knot point (depends on both x and u)    

        Q       = np.eye(2)
        x       = x[:2]
        
        a = np.dot(np.dot((start - input_goal).T, Q), (start - input_goal)) 
        b = np.dot(np.dot((start - x).T, Q), (start - x))
        c = np.dot(np.dot((input_goal - x).T, Q), (input_goal - x)) 


        J_g1 = a - b - c
        J_g1 *= 0.5

        log_sum = 0
        for g in self.exp.get_goals():

            goal_diff   = start - goal
            start_diff  = (start - np.array(x))
            togoal_diff = (np.array(x) - goal)

            diff_curr   = start - x
            diff_goal   = x - goal
            diff_all    = start - goal

            diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
            diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
            diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

            total_steps = self.exp.get_N()

            diff_curr_v = self.get_estimated_cost(diff_curr_v) #, i_step)
            diff_goal_v = self.get_estimated_cost(diff_goal_v) #, self.exp.get_N() - i_step)
            diff_all_v  = self.get_estimated_cost(diff_all_v)


            n = - np.dot(np.dot((start - x).T, Q), (start - x)) - np.dot(np.dot((g - x).T, Q), (g - x))
            d = np.dot(np.dot((start - g).T, Q), (start - g))
            log_sum += (np.exp(n) / np.exp(d))
                
        J = J_g1 - np.log(log_sum)

        J *= -1
        # J += 0.5 *  (u-uref)'*R*(u-uref)

        return J


    def get_legibility_component_new(self, start, goal, x, i_step, raw=False):
        Q       = np.eye(2)
        x       = x[:2]

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = start - goal

        diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
        diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
        diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

        total_steps = self.exp.get_N()

        diff_curr_v = self.get_estimated_cost(diff_curr_v) #, i_step)
        diff_goal_v = self.get_estimated_cost(diff_goal_v) #, self.exp.get_N() - i_step)
        diff_all_v  = self.get_estimated_cost(diff_all_v) #, self.exp.get_N())

        n = - (diff_goal_v) # - diff_curr_v
        d = diff_all_v

        J = np.exp(n) / np.exp(d)


        # xmag, ymag = diff_goal
        # goal_dist = np.sqrt(xmag**2 + ymag**2)

        # J = goal_dist / self.exp.get_local_distance()

        # J = np.exp(J)
        return J

    def get_legibility_cost_ada(self, start, real_goal, x, i_step, raw=False):
        Q       = np.eye(2)
        x       = x[:2]


        cost_list = []

        target_remaining    = np.inf
        nextbest_remaining  = np.inf

        longest_dist = 0.0

        J_target    = 1.0
        J_alt       = 1.0

        for goal in self.exp.get_goals():
            goal_diff   = start - goal
            start_diff  = (start - np.array(x))
            togoal_diff = (np.array(x) - goal)

            diff_curr   = start - x
            diff_goal   = x - goal
            diff_all    = start - goal

            diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
            diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
            diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

            total_steps = self.exp.get_N()

            diff_curr_v = self.get_estimated_cost(diff_curr_v) #, i_step)
            diff_goal_v = self.get_estimated_cost(diff_goal_v) #, self.exp.get_N() - i_step)
            diff_all_v  = self.get_estimated_cost(diff_all_v) #, self.exp.get_N())

            n = - (diff_goal_v) - diff_curr_v
            d = diff_all_v

            J = np.exp(n) / np.exp(d)

            dist_to_goal = self.dist_between(x, goal)

            if nextbest_remaining > dist_to_goal:
                # nextbest_remaining = diff_goal
                nextbest_remaining = dist_to_goal
                J_alt = J

            elif goal == real_goal:
                target_remaining = diff_goal
                target_remaining = dist_to_goal
                J_target = J
                print("Found real goal")

            if longest_dist < dist_to_goal:
                # longest_dist = diff_all
                longest_dist = dist_to_goal

            cost_list.append(J)


        # J = np.exp(nextbest_remaining - target_remaining) / np.exp(diff_all_v)   #np.exp(-1 * target_remaining + nextbest_remaining)

        J = (target_remaining - nextbest_remaining) / (longest_dist)
        print("target, next, longest_dist")
        print(target_remaining, nextbest_remaining, longest_dist)
        # J = J ** 2

        # xmag, ymag = diff_goal
        # goal_dist = np.sqrt(xmag**2 + ymag**2)

        # J = goal_dist / self.exp.get_local_distance()

        # J = np.exp(J)
        return J #sum(cost_list)


    def get_relative_legibility_cost(self, start, goal, x, i_step, raw=False):
        Q       = np.eye(2)
        x       = x[:2]

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = start - goal

        diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
        diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
        diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

        total_steps = self.exp.get_N()

        diff_curr_v = self.get_estimated_cost(diff_curr_v) #, i_step)
        diff_goal_v = self.get_estimated_cost(diff_goal_v) #, self.exp.get_N() - i_step)
        diff_all_v  = self.get_estimated_cost(diff_all_v) #, self.exp.get_N())

        n = - (diff_goal_v) - diff_curr_v
        d = diff_all_v

        J = np.exp(n) / np.exp(d)

        
        # xmag, ymag = diff_goal
        # goal_dist = np.sqrt(xmag**2 + ymag**2)

        # J = goal_dist / self.exp.get_local_distance()

        # J = np.exp(J)
        return J


    def get_legibility_component_og(self, start, goal, x, i_step, raw=False):
        Q       = np.eye(2)
        x       = x[:2]

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = start - goal

        diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
        diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
        diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

        total_steps = self.exp.get_N()

        diff_curr_v = self.get_estimated_cost(diff_curr_v) #, i_step)
        diff_goal_v = self.get_estimated_cost(diff_goal_v) #, self.exp.get_N() - i_step)
        diff_all_v  = self.get_estimated_cost(diff_all_v) #, self.exp.get_N())

        n = - (diff_curr_v) - (diff_goal_v)
        d = diff_all_v

        J = np.exp(n) / np.exp(d)

        if raw:
            return J

        return J

    # def get_legibility_component_og(self, start, goal, x, i_step):
    #     Q       = np.eye(2)

    #     goal_diff   = start - goal
    #     start_diff  = (start - np.array(x))
    #     togoal_diff = (np.array(x) - goal)

    #     diff_curr   = start - x
    #     diff_goal   = x - goal
    #     diff_all    = start - goal

    #     diff_curr_v = np.dot(np.dot(diff_curr.T, Q), diff_curr)
    #     diff_goal_v = np.dot(np.dot(diff_goal.T, Q), diff_goal)
    #     diff_all_v  = np.dot(np.dot(diff_all.T, Q), diff_all)

    #     total_steps = self.exp.get_N()
    #     diff_curr_v = self.get_estimated_cost(diff_curr_v, i_step)
    #     diff_goal_v = self.get_estimated_cost(diff_goal_v, self.exp.get_N() - i_step)
    #     diff_all_v = self.get_estimated_cost(diff_all_v, self.exp.get_N())

    #     print("vals")
    #     print(start, x)
    #     print(x, goal)
    #     print(start, goal)

    #     print("exp diffs")
    #     # print(diff_curr, diff_goal, diff_all)
    #     # print(np.dot(np.dot((diff_curr.T), Q), ((diff_curr)))
    #     # print(((diff_goal).T.dot(Q).dot(diff_goal)))
    #     # print((diff_all).T.dot(Q).dot(diff_all))
    #     print(diff_curr_v)
    #     print(diff_goal_v)
    #     print(diff_all_v)
    #     print("==")

    #     # print("exp norms")
    #     # # print(diff_curr, diff_goal, diff_all)
    #     # print(np.linalg.norm(diff_curr))
    #     # print(np.linalg.norm(diff_goal))
    #     # print(np.linalg.norm(diff_all))
    #     # print("==")

    #     # diff_curr   = diff_curr.T
    #     # diff_goal   = diff_goal.T
    #     # diff_all    = diff_all.T


    #     # diff_curr_size = np.linalg.norm(diff_curr)
    #     # diff_goal_size = np.linalg.norm(diff_goal)
    #     # diff_all_size  = np.linalg.norm(diff_all)

    #     if np.array_equal(goal[:2], [3.79, -6.99]) or np.array_equal(goal[:2], [7.41, -6.99]):
    #         # Apply roughly the scalar to match the OG dimensions, 
    #         # versus the units from unity we're using currently for display purposes
    #         diff_curr   = diff_curr  * 100.0
    #         diff_goal   = diff_goal  * 100.0
    #         diff_all    = diff_all  * 100.0

    #         print("exp diffs")
    #         # print(diff_curr, diff_goal, diff_all)
    #         print(diff_curr_v)
    #         print(diff_goal_v)
    #         print(diff_all_v)
    #         print("==")
    #     # else:
    #     #     print("Alert: goal is " + str(goal))

    #     # diff_curr_size = diff_curr_size * diff_curr_size
    #     # diff_goal_size = diff_goal_size * diff_goal_size
    #     # diff_all_size  = diff_all_size * diff_all_size

    #     # n = - (diff_curr.T).dot(Q).dot((diff_curr)) - ((diff_goal).T.dot(Q).dot(diff_goal))
    #     # d = (diff_all).T.dot(Q).dot(diff_all)

    #     n = - (diff_curr_v) - (diff_goal_v)
    #     d = diff_all_v

    #     print("n, d")
    #     print(n, d)
    #     J = np.exp(n) / np.exp(d)

    #     return J

    def get_legib_adamich(self, start_input, goal_input, x_triplet, u_input, i_step, terminal, visibility_coeff, override=None):
        x       = (x_triplet[:2])
        u       = u_input
        start   = (start_input[:2])
        goal    = (goal_input[:2])
        all_goals = self.goals

        goal_values = []
        target_index = -1

        for j in range(len(all_goals)):
            alt_goal = all_goals[j]
            alt_goal_xy = np.asarray(alt_goal[:2])

            goal_values.append(goal_val)

        total = sum([abs(ele) for ele in goal_values])

        # if mode_dist in ['exp', 'exp_lin']:
        #     print("small value norm")
        #     goal_values_norm = self.small_value_norm(goal_values)  
        #     # goal_values_norm_v1 = self.small_value_norm_v1(goal_values)   
        #     print("LOGSUMEXP")
        #     print(goal_values_norm)
        #     # print(goal_values_norm_v1)
        #     dist_prob = goal_values_norm[target_index]
        #     # dist_prob = target_val / np.linalg.norm(goal_values, ord=1)

        #     print("Dist normalized")
        #     print(dist_prob, goal_values)

        # elif total == 0:
        #     print("ALERT: in prob distance division")
        #     num_goals = len(all_goals)
        #     dist_prob = ((1.0/num_goals))
        # else:
        #     dist_prob = (goal_values[target_index]) / total



        return P_oa



    def get_legibility_dist_stage_cost(self, start, goal, x, u, i_step, terminal, visibility_coeff):
        P_oa = self.prob_distance(start, goal, x, u, i_step, terminal, visibility_coeff)

        if (P_oa) < 0:
            print("ALERT: P_oa < 0")
        elif (P_oa) > 1:
            print("ALERT: P_oa > 1")

        print("P_oa is " + str(P_oa))

        # return (np.exp(1.0)) - np.exp(P_oa)
        return (1.0) - P_oa


    def prob_distance(self, start_input, goal_input, x_triplet, u_input, i_step, terminal, visibility_coeff, override=None):
        x       = (x_triplet[:2])
        u       = u_input
        start   = (start_input[:2])
        goal    = (goal_input[:2])
        all_goals = self.goals

        mode_dist    = self.exp.get_mode_type_dist()
        mode_heading = self.exp.get_mode_type_heading()
        mode_blend   = self.exp.get_mode_type_blend()

        if override is not None:
            if 'mode_dist' in override.keys():
                mode_dist = override['mode_dist']
                print("OO: Dist override: " + str(mode_dist))


        if not np.array_equal(goal[:2], self.exp.get_target_goal()[:2]):
            print("Goal and exp goal not the same in prob_distance")
            print(goal[:2], self.exp.get_target_goal()[:2])

        if visibility_coeff == 1 or visibility_coeff == 0:
            pass
        else:
            print("vis is not 1 or 0")


        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("For point at x -> " + str(x))
            debug_dict = {'start':start, 'goal':goal, 'all_goals':self.exp.get_goals(), 'x': x_triplet, 'u': u, 'i':i_step, 'goal': goal, 'visibility_coeff': visibility_coeff, 'N': self.exp.get_N(),'override':override, 'mode_dist': mode_dist}
            print("DIST COST INPUTS")
            print(debug_dict)

            print("TYPE OF DIST: " + str(mode_dist))
            print("GOAL: " + str(goal))

        if np.array_equal(x, goal):
            print("We are on the goal")
            P_dist = (1.0)

            num_goals   = len(all_goals)
            P_oa        = ((1.0/num_goals)*(1.0 - visibility_coeff)) + (((visibility_coeff) * P_dist))
            return P_oa


        goal_values = []
        target_index = -1

        for j in range(len(all_goals)):
            alt_goal = all_goals[j]
            alt_goal_xy = np.asarray(alt_goal[:2])
            goal_val = self.get_relative_distance_value(i_step, start, alt_goal_xy, x, terminal, mode_dist)

            FLAG_NO_OVERSHOOT = False
            if FLAG_NO_OVERSHOOT:
                goal_val = goal_val / self.dist_between(start, alt_goal_xy)

            goal_values.append(goal_val)

            if np.array_equal(goal[:2], alt_goal[:2]):
                target_val = goal_val
                # print("Target found")
                target_index = j

            else:
                # print("Not it")
                # print(goal, alt_goal)
                pass

        print("Target val")
        print(target_val)
        print("All values")
        print([str(ele) for ele in goal_values])
        print("-")

        total = sum([abs(ele) for ele in goal_values])

        if mode_dist in ['exp', 'exp_lin']:
            print("small value norm")
            goal_values_norm = self.small_value_norm(goal_values)  
            # goal_values_norm_v1 = self.small_value_norm_v1(goal_values)   
            print("LOGSUMEXP")
            print(goal_values_norm)
            # print(goal_values_norm_v1)
            dist_prob = goal_values_norm[target_index]
            # dist_prob = target_val / np.linalg.norm(goal_values, ord=1)

            print("Dist normalized")
            print(dist_prob, goal_values)

        elif total == 0:
            print("ALERT: in prob distance division")
            num_goals = len(all_goals)
            dist_prob = ((1.0/num_goals))
        else:
            dist_prob = (goal_values[target_index]) / total

        print("Dist prob " + str(dist_prob))

        P_dist      = (dist_prob)
        num_goals   = len(all_goals)
        P_oa        = ((1.0/num_goals)*(1.0 - visibility_coeff)) + (((visibility_coeff) * P_dist))

        return P_oa

    def get_legibility_component_alts(self, start, goal, x, u, i, terminal, visibility_coeff, override_block=None):
        p_alt_list = []
        goals =  self.exp.get_goals()

        print("Get all components")
        for gi in range(len(self.exp.get_goals())):
            g_test = goals[gi]
            p_d_g = self.prob_distance(start, g_test, x, u, i, terminal, visibility_coeff, override=override_block)

            print("p_d_g == " + str(p_d_g))
            print(g_test)

            if g_test != goal:
                p_alt_list.append(p_d_g)


        p_g_target = self.prob_distance(start, goal, x, u, i, terminal, visibility_coeff, override=override_block)

        return p_g_target, p_alt_list

    def stage_cost(self, x, u, i, terminal=False):
        print("DOING STAGE COST")
        start   = self.start
        goal    = self.target_goal

        x = np.array(x)
        J = self.goal_efficiency_through_point_relative(start, goal, x, terminal)
        return J

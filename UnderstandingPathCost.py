import os
import sys
module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import matplotlib.pyplot as plt
import decimal
import copy
import math

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

from LegiblePathCost import LegiblePathCost
import PathingExperiment as ex

from shapely.geometry import LineString
from shapely.geometry import Point

np.set_printoptions(suppress=True)
np.seterr(divide='raise')
MATH_EPSILON = 0 #.0000001

class UnderstandingPathCost(LegiblePathCost):
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

        LegiblePathCost.__init__(
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
            if i is True:
                print("TELEPORTED ACROSS: table " + str(ct))
                return True
        
        for o in observers:
            ct = o.get_center()
            p = Point(ct[0],ct[1])
            c = p.buffer(OBS_RADIUS).boundary
            i = c.intersection(l)
            if i is True:
                print("TELEPORTED ACROSS: o " + str(ct))
                return True
        
        for g in goals:
            ct = g
            p = Point(ct[0],ct[1])
            c = p.buffer(GOAL_RADIUS).boundary
            i = c.intersection(l)
            if i is True:
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
            if Fpsp == 0:
                print("ALERT no penalty")
        elif obst_dist < obstacle_buffer + obstacle_radius:
            print("Inside the overall force diagram")
            if Fpsp == 0:
                print("ALERT no penalty")
        else:
            print("Not in an obstacle")
            if Fpsp > 0:
                print("ALERT obstacle penalty when not in object")

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
            if not np.array_equal(g, self.exp.get_target_goal()):
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


    def inversely_proportional_to_distance(self, x):
        if x == 0:
            return np.Inf
        return 1.0 / float(x)

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


    def get_angle_between_triplet(self, a, b, c):
        # print(type(a), type(b), type(c))

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

        print("soc math check")
        print(a, b, c, str(ang) + " degrees", alt_check)

        if np.abs(ang - alt_check) > 1.0:
            print("ALERT: HEADING ANGLE MATH MAY BE INCONSISTENT")

        return ang

    def get_heading_of_pt_diff_p2_p1(self, p2, p1):
        unit_vec    = [p1[0] + 1.0, p1[1]]
        heading     = self.get_angle_between_triplet(p2, p1, unit_vec)
        return heading

    def get_legibility_heading_stage_cost(self, x, u, i, goal, visibility_coeff, override=None):
        # if len(x) > 2 and x[2] is None:
        #     return 1.0

        P_oa = self.prob_heading(x, u, i, goal, visibility_coeff, override=override)

        cost = decimal.Decimal(1.0) - decimal.Decimal(P_oa)
        return cost

    def get_legibility_heading_stage_cost_from_pt_seq(self, x_cur, x_prev, i, goal, visibility_coeff, u=None, override=None):
        P_oa = self.prob_heading_from_pt_seq(x_cur, x_prev, i, goal, visibility_coeff, u=u, override=override)

        cost = decimal.Decimal(1.0) - decimal.Decimal(P_oa)
        cost = decimal.Decimal(P_oa)
        return cost

    def prob_heading_from_pt_seq(self, x_cur, x_prev, i_in, goal_in, visibility_coeff, u=None, override=None):
        if not np.array_equal(x_cur, x_prev):
            x_theta = self.get_heading_of_pt_diff_p2_p1(x_cur[:2], x_prev[:2])
        else:
            x_theta = None

        print("PRECALC heading angle")
        print(x_cur, x_prev, x_theta)

        x_triplet = [x_cur[0], x_cur[1], x_theta]

        u_output = u
        # return self.prob_heading(x_triplet, u_output, i_in, goal_in, visibility_coeff, override=override)

        # Version that uses pt differences
        return self.prob_heading_from_pt_seq_alt(x_cur, x_prev, u_output, i_in, goal_in, visibility_coeff, override=override)

    # This function only takes size 3 vectors
    def prob_heading_from_pt_seq_alt(self, x_cur, x_prev, u_in, i_in, goal_in, visibility_coeff, override=None):
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
            print("ALERT: Goal and exp goal not the same in prob heading")
            print(goal, self.exp.get_target_goal())
            # exit()

        debug_dict = {'x': [x_cur, x_prev], 'u': u, 'i':i, 'goal': goal, 'start': self.exp.get_start(), 'all_goals':self.exp.get_goals(), 'visibility_coeff': visibility_coeff, 'N': self.exp.get_N(), 'override': override, 'mode_heading': mode_heading}
        print("HEADING EFFORT COST INPUTS")
        print(debug_dict)


        target_vector           = None
        all_goal_headings       = []

        target_index = -1
        x       = x_cur
        x_prev  = x_prev

        all_effort_measures         = []
        for j in range(len(self.goals)):
            # print("Goal angle diff for " + str(robot_theta) + " -> " + str(ghead))
            # goal_angle_diff  = self.get_min_rotate_angle_diff(robot_theta, ghead)
            # print("goal angle diff " + str(goal_angle_diff))
            # effort_made = decimal.Decimal(180.0 - goal_angle_diff)
            # print("effort made " + str(effort_made))

            alt_goal = all_goals[j]
            # goal_vector = alt_goal - x_current
            # all_goal_vectors.append(goal_vector)

            if np.array_equal(alt_goal[:2], goal[:2]):
                print("Yes, is target")
                target_index = j

            effort_made = self.get_angle_between_triplet(x_prev, x, alt_goal)

            if float(effort_made) > 180 or float(effort_made) < 0:
                print("ALERT: Get angle between needs some work")
                # exit()

            if mode_heading is 'sqr':
                effort_made = effort_made**2

            all_effort_measures.append(effort_made)

        print("All effort measures")
        print(all_effort_measures)

        target_val  = all_effort_measures[target_index]
        total       = sum(all_effort_measures)
        num_goals   = len(all_goals)

        try:
            all_probs = [x/total for x in all_effort_measures]
            print(all_probs)

            P_heading   = decimal.Decimal((target_val) / total)
        except (ValueError, decimal.InvalidOperation, ZeroDivisionError):
            print("Error! ...")
            P_heading = decimal.Decimal(1.0 / num_goals)


        P_oa        = decimal.Decimal((1.0/num_goals)*(1.0 - visibility_coeff)) + ((decimal.Decimal(visibility_coeff) * P_heading))
        return P_oa

    # This function only takes size 3 vectors
    def prob_heading(self, x_triplet, u_in, i_in, goal_in, visibility_coeff, override=None):
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
            return decimal.Decimal(0.0)
        else:
            robot_theta = x_triplet[2]

        if not np.array_equal(goal[:2], self.exp.get_target_goal()[:2]):
            print("Goal and exp goal not the same in prob heading")
            print(goal, self.exp.get_target_goal())
            # exit()

        # # if we are at the goal, we by definition are arriving correctly
        # if self.dist_between(x_current, goal) < 1.1:
        #     P_heading   = decimal.Decimal(1.0)

        #     num_goals   = len(all_goals)
        #     P_oa        = decimal.Decimal((1.0/num_goals)*(1.0 - visibility_coeff)) + ((decimal.Decimal(visibility_coeff) * P_heading))
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
            effort_made = decimal.Decimal(180.0 - goal_angle_diff)
            print("effort made " + str(effort_made))

            # effort_two = self.get_angle_between_triplet(a, b, c)

            if float(effort_made) > 180 or float(effort_made) < 0:
                print("ALERT: Get angle between needs some work")
                # exit()
            if goal_angle_diff < 0:
                print(goal_angle_diff)
                print("eeek negative angle diff")
                # exit()

            if mode_heading is 'sqr':
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
            P_heading = decimal.Decimal(1.0 / num_goals)


        num_goals   = len(all_goals)
        P_oa        = decimal.Decimal((1.0/num_goals)*(1.0 - visibility_coeff)) + ((decimal.Decimal(visibility_coeff) * P_heading))

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
        if i is 0:
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

        x               = x_triplet[:2]
        squared_x_cost  = self.get_x_diff(x, i)

        print("IS TERM EVIL?")
        print(x_triplet, self.x_path[i], squared_x_cost)

        terminal_cost = squared_x_cost

        if terminal_cost is np.nan:
            print("Terminal cost of nan, but how?")
            print(x)
            print(x_diff)
            print(self.x_path)
            terminal_cost = np.inf

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("term cost squared x cost")
            print(squared_x_cost)
            pass

        # We want to value this highly enough that we don't not end at the goal
        # terminal_coeff = 100.0
        coeff_terminal = self.exp.get_solver_scale_term() * (1.0 / self.exp.get_dt())
        terminal_cost = terminal_cost * coeff_terminal

        print("actual terminal cost")
        print(terminal_cost)

        # Once we're at the goal, the terminal cost is 0
        
        # Attempted fix for paths which do not hit the final mark
        # if squared_x_cost > .001:
        #     terminal_cost *= 1000.0

        return terminal_cost

    def get_u_diff(self, x, u, i):
        u_calc = x[:2] - x[2:]

        if np.array_equal(u_calc, u):
            print("NA: U calc as promised!")
        else:
            print("ALERT: U is wrong?")
            print(x, u, u_calc)
            print(x[:2], x[2:])
            
        # u = u_calc

        if u.any() == None:
            return 0.0

        # print("incoming " + str(u))
        R = np.eye(2)
        u_diff      = np.abs(u - self.u_path[i])
        u_diff      = np.abs(u - np.asarray([0, 0]))
        val_u_diff  = u_diff.T.dot(R).dot(u_diff)

        if u[0] is np.nan or u[1] is np.nan:
            print("FLAT PENALTY FOR NANS")
            return val_u_diff * 10000.0

        print("udiff calc")
        print(u, "-", self.u_path[i], u_diff, val_u_diff)
        return val_u_diff

    def get_x_diff(self, x_input, i):
        print(x_input.eval())

        x_ref   = np.asarray([self.x_path[i][..., 0], self.x_path[i][..., 1]])
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

    # original version for plain path following
    def l(self, input_x, input_u, input_i, terminal=False, just_term=False, just_stage=False):
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

        Q = self.Qf if terminal else self.Q
        R = self.R
        start   = self.start
        goal    = self.target_goal
        thresh  = .0001

        # u = copy.copy(input_u)
        # x = copy.copy(input_x)
        x           = input_x[..., :]
        x_cur       = input_x[..., 0:1]
        x_prev      = input_x[..., 2:3]

        u = input_u[..., :]

        i = copy.copy(input_i)
        print("INPUT U IS " + str(u))
        if u is None:
            u = np.asarray([np.inf, np.inf])


        scale_term  = self.exp.get_solver_scale_term() #0.01 # 1/100
        scale_stage = self.exp.get_solver_scale_stage() #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0

        term_cost = self.term_cost(x, i)

        # # xdiff from preferred line
        # x_path[i] is always the goal
        # x_diff = x - self.x_path[i]

        if terminal or just_term: #abs(i - self.N) < thresh or
            return scale_term * (1.0 / self.exp.get_dt()) * self.term_cost(x, i)
        else:
            if self.FLAG_DEBUG_STAGE_AND_TERM:
                # difference between this step and the end
                print("x, N, x_end_of_path -> inputs and then term cost")
                print(x, self.N, self.x_path[self.N])
                # term_cost = self.term_cost(x, i)
                print(term_cost)

        # VISIBILITY COMPONENT
        restaurant  = self.exp.get_restaurant()
        observers   = self.exp.get_observers()

        squared_x_cost = self.get_x_diff(x, i)
        squared_u_cost = self.get_u_diff(x, u, i)

        val_angle_diff  = 0

        ### USE ORIGINAL LEGIBILITY WHEN THERE ARE NO OBSERVERS
        if self.exp.get_is_oa_on() is True:
            if len(observers) > 0:
                visibility  = legib.get_visibility_of_pt_w_observers_ilqr(x, observers, normalized=True)
            else:
                visibility  = 1.0
        else:
            visibility = 1.0

        FLAG_OA_MIN_VIS = False
        f_func     = self.get_f()
        f_value    = visibility

        # KEEP THE VIS VALUE IF F_VIS_LIN, OR...
        if self.exp.get_f_label() is ex.F_VIS_BIN:
            if f_value > 0:
                f_value = 1.0
            else:
                f_value = 0.0
        elif self.exp.get_f_label() is ex.F_NONE:
            f_value = 1.0

        visibility_coeff = f_value

        wt_legib     = 10.0
        wt_lam       = 10.0 * (1.0 / self.exp.get_dt())
        wt_heading   = 10.0
        wt_obstacle  = 1.0 #self.exp.get_solver_scale_obstacle()
        
        # If heading is not on, set the weight to not on
        if self.exp.get_mode_type_dist() is not None and self.exp.get_mode_type_heading() is None:
            # wt_legib    = wt_legib + wt_heading
            wt_heading  = 0.0

        # If a heading-only scenario, shift weighting to that
        if self.exp.get_mode_type_dist() is None and self.exp.get_mode_type_heading() is not None:
            # wt_heading  = wt_heading + wt_legib
            wt_legib    = 0.0
            wt_lam      = wt_lam

        if self.exp.get_mode_type_dist() is None and self.exp.get_mode_type_heading() is None:
            wt_legib    = 0
            wt_heading  = 0
            wt_lam      = wt_lam

        if self.exp.get_mode_type_dist() is 'sqr' and self.exp.get_mode_type_heading() is None:
            wt_lam      *= 1.0
            wt_legib    *= 1.5

        elif self.exp.get_mode_type_dist() is 'lin' and self.exp.get_mode_type_heading() is None:
            wt_lam      *= 1.0
            wt_legib    *= 1.5

        elif self.exp.get_mode_type_dist() is 'exp' and self.exp.get_mode_type_heading() is None:
            wt_legib    *= 1.0 #10.0
            wt_heading  *= 1.0 #10.0
            wt_lam      *= 1.0 #10.0


        val_legib       = 0
        val_lam         = 0
        val_obstacle    = 0
        val_heading     = 0

        # Get the values if necessary
        if wt_legib > 0:
            val_legib       = self.get_legibility_dist_stage_cost(start, goal, x, u, i, terminal, visibility_coeff)
        if wt_heading > 0:
            if self.exp.get_state_size() < 4:
                val_heading     = self.get_legibility_heading_stage_cost(x, u, i, goal, visibility_coeff)
            else:
                x_new   = x[:2]
                x_prev  = x[2:]
                val_heading     = self.get_legibility_heading_stage_cost_from_pt_seq(x_new, x_prev, i, goal, visibility_coeff, u=u)

        if wt_lam > 0:
            val_lam         = squared_u_cost #+ (.1 * squared_x_cost)
        else:
            print("ALERT: why is lam weight 0?")

        if wt_obstacle > 0:
            val_obstacle    = self.get_obstacle_penalty(x, i, goal)

        wt_legib     = decimal.Decimal(wt_legib)
        wt_lam       = decimal.Decimal(wt_lam)
        wt_heading   = decimal.Decimal(wt_heading)
        wt_obstacle  = decimal.Decimal(wt_obstacle)

        val_legib     = decimal.Decimal(val_legib)
        val_lam       = decimal.Decimal(val_lam)
        val_heading   = decimal.Decimal(val_heading)
        val_obstacle  = decimal.Decimal(val_obstacle)

        # Pathing with taking the max penalty, ie min probability
        if wt_legib > 0 and wt_heading > 0:
            if float(val_legib) > float(val_heading):
                max_val = val_legib
            else:
                max_val = val_heading

            val_legib   = max_val
            val_heading = max_val
            wt_legib    = decimal.Decimal(.5) * wt_legib
            wt_heading  = decimal.Decimal(.5) * wt_heading


        # J does not need to be in a particular range, it can be any max or min
        J = 0        
        J += wt_legib       * val_legib     #self.legibility_stage_cost(start, goal, x, u, i, terminal, visibility_coeff)
        J += wt_heading     * val_heading   #self.get_heading_cost(x, u, i, goal, visibility_coeff)

        J += wt_lam         * val_lam           #u_diff.T.dot(R).dot(u_diff)
        # J += wt_lam_h       * val_lam_h         #u_diff.T.dot(R).dot(u_diff)
        J += wt_obstacle    * val_obstacle      #self.get_obstacle_penalty(x, i, goal)

        stage_costs = sum([wt_legib*val_legib, wt_lam*val_lam, wt_heading*val_heading, wt_obstacle*val_obstacle])
        stage_costs = float(stage_costs)

        # if stage_costs != J:
        #     print("alert! j math is off")
        #     print("J = " + str(J))
        #     print(stage_costs)

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

            print("[wt_legib, wt_lam, wt_heading, wt_obstacle]")
            print([str(wt_legib), str(wt_lam), str(wt_heading), str(wt_obstacle)])            
            print("[val_legib, val_lam, val_heading, val_obstacle]")
            print([str(val_legib), str(val_lam), str(val_heading), str(val_obstacle)])
            print("==")
            print([str(wt_legib*val_legib), str(wt_lam*val_lam), str(wt_heading*val_heading), str(wt_obstacle*val_obstacle)])

            print(str(sum([wt_legib*val_legib, wt_lam*val_lam, wt_heading*val_heading, wt_obstacle*val_obstacle])))
            print("~~~~~~~~~~~~")

        # Don't return term cost here ie (scale_term * term_cost) 
        total = (scale_stage * stage_costs)
        return float(total)

    def f(t):
        return 1.0

    def dist_between(self, x1, x2):
        distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        return distance

    def get_relative_distance_value(self, start, goal, x_input, terminal, mode_dist):
        Q       = np.eye(2) #self.Q_terminal if terminal else self.Q
        x       = x_input
        dist    = self.dist_between(x, goal)

        print("dist to the goal <" + str(goal) + ">")
        print(dist)
        if dist < 0:
            print("dist to goal less than 0!")
            exit()

        if mode_dist is 'sqr':
            val = decimal.Decimal(self.inversely_proportional_to_distance(dist)**2)
            print("mode dist is sqr, dist inv value is " + str(val))
            return val
            # return decimal.Decimal(self.get_relative_distance_k_sqr(x, goal, self.goals))
        elif mode_dist is 'lin':
            val = decimal.Decimal(self.inversely_proportional_to_distance(dist))
            print("mode dist is lin, dist inv value is " + str(val))
            return val
            # return decimal.Decimal(self.get_relative_distance_k(x, goal, self.goals))

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = start - goal

        print("exp diffs")
        print(diff_curr, diff_goal, diff_all)

        diff_curr   = diff_curr.T
        diff_goal   = diff_goal.T
        diff_all    = diff_all.T

        n = - (diff_curr.T).dot(Q).dot((diff_curr)) - ((diff_goal).T.dot(Q).dot(diff_goal))
        d = (diff_all).T.dot(Q).dot(diff_all)

        n = decimal.Decimal(n)
        d = decimal.Decimal(d)

        print("n, d")
        print(n, d)

        # J = np.exp(n) / np.exp(d)

        if mode_dist is 'exp':
            J = np.exp(n) / np.exp(d)
        else:
            J = np.abs(n / d)
            print("ALERT: UNKNOWN MODE = " + str(mode_dist))

        if self.exp.get_weighted_close_on() is True:
            k = self.get_relative_distance_k(x, goal, self.goals)
        else:
            k = 1.0

        J = decimal.Decimal(k)*J
        print("J of exp")
        print(J)
        return J


    def get_legibility_dist_stage_cost(self, start, goal, x, u, i_step, terminal, visibility_coeff):
        P_oa = self.prob_distance(start, goal, x, u, i_step, terminal, visibility_coeff)

        return decimal.Decimal(1.0) - P_oa


    def prob_distance(self, start_input, goal_input, x_triplet, u_input, i_step, terminal, visibility_coeff, override=None):
        x       = np.asarray(x_triplet[:2])
        u       = u_input
        start   = np.asarray(start_input[:2])
        goal    = np.asarray(goal_input[:2])
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
            print(goal, self.exp.get_target_goal())      

        if visibility_coeff == 1 or visibility_coeff == 0:
            pass
        else:
            print("vis is not 1 or 0")


        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("For point at x -> " + str(x))
            debug_dict = {'start':start, 'goal':goal, 'all_goals':self.exp.get_goals(), 'x': x_triplet, 'u': u, 'i':i_step, 'goal': goal, 'visibility_coeff': visibility_coeff, 'N': self.exp.get_N(),'override':override, 'mode_dist': mode_dist}
            print("DIST COST INPUTS")
            print(debug_dict)
            # print("TYPE OF DIST: " + str(mode_dist))

        if np.array_equal(x, goal):
            print("We are on the goal")
            P_dist = decimal.Decimal(1.0)

            num_goals   = len(all_goals)
            P_oa        = decimal.Decimal((1.0/num_goals)*(1.0 - visibility_coeff)) + ((decimal.Decimal(visibility_coeff) * P_dist))
            return P_oa


        goal_values = []
        for alt_goal in all_goals:
            alt_goal_xy = np.asarray(alt_goal[:2])
            goal_val = self.get_relative_distance_value(start, alt_goal_xy, x, terminal, mode_dist) 
            goal_values.append(goal_val)

            if np.array_equal(goal[:2], alt_goal[:2]):
                target_val = goal_val
                print("Target found")

        print("Target val")
        print(target_val)
        print("All values")
        print([str(ele) for ele in goal_values])

        total = sum([abs(ele) for ele in goal_values])

        try:        
            # dist_prob = (total - target_val) / (total)
            print(target_val, total)
            dist_prob = (target_val) / total
        except:
            print("ALERT: in prob distance division")
            dist_prob = decimal.Decimal((1.0/num_goals))

        print("Dist prob " + str(dist_prob))

        P_dist      = decimal.Decimal(dist_prob)
        num_goals   = len(all_goals)
        P_oa        = decimal.Decimal((1.0/num_goals)*(1.0 - visibility_coeff)) + ((decimal.Decimal(visibility_coeff) * P_dist))

        return P_oa


    def stage_cost(self, x, u, i, terminal=False):
        print("DOING STAGE COST")
        start   = self.start
        goal    = self.target_goal

        x = np.array(x)
        J = self.goal_efficiency_through_point_relative(start, goal, x, terminal)
        return J

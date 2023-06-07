import os
import sys
module_path = os.path.abspath(os.path.join('../../ilqr'))
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

np.seterr(divide='raise')
MATH_EPSILON = 0 #.0000001

class AdaLegibilityQRCost(LegiblePathQRCost):
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

    def get_obstacle_penalty_given_obj(self, x, i, obst_center, obstacle_radius):
        obstacle_radius = obstacle_radius # the physical obj size
        obstacle_buffer = self.exp.get_obstacle_buffer() # the area in which the force will apply
        threshold       = obstacle_buffer

        # obst_dist is the distance between the point and the center of the obj
        obst_dist = obst_center - x
        obst_dist = np.abs(np.linalg.norm(obst_dist))
        # obst_dist = self.get_closest_point_on_line(x, i, obst_center)

        # rho is the distance between the object's edge and the pt
        rho             = obst_dist - obstacle_radius #- self.exp.get_obstacle_buffer()
        # if rho is negative, we are inside the sphere
        eta             = 1.0

        # if obst_dist < (threshold + obstacle_radius)
        if rho > obstacle_buffer:
            return 0

        # if rho is positive, in the force zone
        # if rho is negative, in the 


        rho = rho
        # vector component
        d_rho_top = np.linalg.norm(obst_center - x)
        d_rho_dx = d_rho_top / rho

        print("rho")
        print(rho)

        a = 1.0 / (rho)
        b = (1.0 / obstacle_radius) #TODO Not obstacle_buffer?
        c = 1.0/(rho**2)

        print("a, b, c, a-b, d_rho_x")
        print(str([a, b, c, a-b, d_rho_dx]))

        print("a - b, c, d_rho_x")
        print(str([a-b, c, d_rho_dx]))

        # value = (a - b) * (c)

        # value = (eta * ((1.0/rho) - (1.0/threshold)) *
        #         1.0/(rho**2) * d_rho_dx)

        value = eta * (a - b) * c * d_rho_dx

        print("d_rho_dx")
        print(d_rho_dx)

        # if value is np.nan:
        #     value = np.Inf
        #     print("oh no")
        #     exit()

        print("Obstacle intersection!")
        print("Given " + str(x) + " -> " + str(obst_center))
        print("We at a dist of " + str(rho) + " from the surface (rho)")
        print(str(obst_dist) + " minus " + str(obstacle_radius))

        if obst_dist < (threshold + self.exp.get_obstacle_buffer()):
            print("Inside the overall force diagram")

        if obst_dist < threshold:
            print("Inside the actual obj")

        print("obspenalty is: ")
        print(value)

        return value

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


    ##### METHODS FOR ANGLE MATH
    def get_angle_between(self, p2, p1):
        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        ang1    = np.arctan2(*p1[::-1])
        ang2    = np.arctan2(*p2[::-1])
        heading = np.rad2deg((ang1 - ang2) % (2 * np.pi))

        heading = self.get_minimum_rotation_to(heading)

        # Heading is in degrees
        return heading

    def angle_diff(self, a1, a2):
        # target - source
        a = a1 - a2
        diff = (a + 180) % 360 - 180

        return diff


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

    def get_heading_stage_cost_debug(self, x, u, i, goal, visibility_coeff, force_mode=None, x_prev=None, pure_prob=False):
        all_goals = self.goals

        # If on first timestep and no angle yet, even prior on all
        if i is 0:
            return (1.0 / len(self.goals))

        x1 = x
        x0 = x - u

        robot_vector    = x1 - x0
        target_vector   = None
        all_goal_vectors    = []

        for alt_goal in all_goals:
            goal_vector = alt_goal - x1

            if np.array_equal(alt_goal, goal):
                target_vector = goal_vector
            else:
                pass
                # print("no, mismatch of " + str(alt_goal) + " != " + str(goal))
            all_goal_vectors.append(goal_vector)

        all_goal_angles   = []
        for gvec in all_goal_vectors:
            goal_angle = self.get_angle_between(robot_vector, gvec)

            all_goal_angles.append(goal_angle)

        target_angle = self.get_angle_between(robot_vector, target_vector)

        g_vals = []
        g_vals_if_infinity = []
        for j in range(len(all_goal_angles)):
            gval = self.inversely_proportional_to_angle(all_goal_angles[j])

            if self.exp.get_mode_heading_err_sqr() is True or force_mode is 'sqr':
                gval = gval * gval
            
            # if self.exp.get_weighted_close_on() is True:
            #     k = self.get_relative_distance_k(x1, goals[i], goals)
            # else:
            #     k = 1.0

            # by default, evenly weight all goals
            k = 1.0
            g_vals.append(k * gval)
            g_vals_if_infinity.append((k, gval == np.Inf))

        target_val = self.inversely_proportional_to_angle(target_angle)

        return {'robot_vec': robot_vector, 'x0': x0, 'x1':x1, 'target_angle': target_angle, 'target_val': target_val, 'all_angles':all_goal_angles}

    def get_heading_stage_cost(self, x, u, i, goal, visibility_coeff, force_mode=None, pure_prob=False):
        return 0


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


    # How far away is the final step in the path from the goal?
    def term_cost(self, x, i):
        start = self.start
        goal1 = self.target_goal
        
        # Qf = self.Q_terminal
        Qf = self.Qf
        R = self.R

        x_diff = x - self.x_path[i]
        squared_x_cost = .5 * x_diff.T.dot(Qf).dot(x_diff)
        # squared_x_cost = squared_x_cost * squared_x_cost

        terminal_cost = squared_x_cost

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("term cost squared x cost")
            print(squared_x_cost)

        # We want to value this highly enough that we don't not end at the goal
        # terminal_coeff = 100.0
        coeff_terminal = self.exp.get_solver_coeff_terminal()
        terminal_cost = terminal_cost * coeff_terminal

        # Once we're at the goal, the terminal cost is 0
        
        # Attempted fix for paths which do not hit the final mark
        # if squared_x_cost > .001:
        #     terminal_cost *= 1000.0

        return terminal_cost


    # ADANOTE: This is the primary cost function
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

        scale_term = self.exp.get_solver_scale_term() #0.01 # 1/100
        scale_stage = self.exp.get_solver_scale_stage() #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0

        term_cost = 0 #self.term_cost(x, i)

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

        # VISIBILITY COMPONENT
        restaurant  = self.exp.get_restaurant()
        observers   = self.exp.get_observers()

        u_diff = u - self.u_path[i]
        val_u_diff      = u_diff.T.dot(R).dot(u_diff)
        val_angle_diff  = 0 #ang_diff * ang_diff

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

        # ADANOTE: THIS IS WHERE YOU ALTER WEIGHTS
        # YOU LIKELY AREN'T USING HEADING, SO legib is the one to increase exaggeration
        # lam is the one for making it higher cost to be curvy
        # N in the scenario setup is the number of steps we solve for, it's not here
        # obstacle is the penalty for penetrating obstacles

        wt_legib     = 10 #-(1/30.0) #100.0
        wt_lam       = 5 #2 #.05 #also tried .1
        wt_heading   = 10 #0.5 #100000.0
        wt_obstacle  = 100000.0 #self.exp.get_solver_scale_obstacle()

        # ADANOTE This encourages efficiency, and faster solving
        x_diff = x - self.x_path[i]
        squared_x_cost = .5 * x_diff.T.dot(Q).dot(x_diff)
        
        if self.exp.get_is_heading_on() is False:
            wt_legib    = wt_legib + wt_heading
            wt_heading  = 0.0

        if self.exp.get_mode_pure_heading() is True:
            wt_heading  = wt_heading + wt_legib
            wt_legib    = 0.0

        if self.exp.get_mode_dist_legib_on() is False:
            wt_legib = 0

        wt_legib     = decimal.Decimal(wt_legib)
        wt_lam       = decimal.Decimal(wt_lam)
        wt_heading   = decimal.Decimal(wt_heading)
        wt_obstacle  = decimal.Decimal(wt_obstacle)

        val_legib       = self.legibility_stage_cost(start, goal, x, u, i, terminal, visibility_coeff)
        val_lam         = val_u_diff + (.00001 * squared_x_cost * 0) # ADANOTE This ratio is up to tuning
        val_obstacle    = self.get_obstacle_penalty(x, i, goal)
        val_heading     = self.get_heading_stage_cost(x, u, i, goal, visibility_coeff)

        val_legib     = decimal.Decimal(val_legib)
        val_lam       = decimal.Decimal(val_lam)
        val_heading   = decimal.Decimal(val_heading)
        val_obstacle  = decimal.Decimal(val_obstacle)

        # J does not need to be in a particular range, it can be any max or min
        J = 0        
        J += wt_legib       * val_legib     #self.legibility_stage_cost(start, goal, x, u, i, terminal, visibility_coeff)
        J += wt_heading     * val_heading   #self.get_heading_cost(x, u, i, goal, visibility_coeff)

        J += wt_lam         * val_lam           #u_diff.T.dot(R).dot(u_diff)
        # J += wt_lam_h       * val_lam_h         #u_diff.T.dot(R).dot(u_diff)
        J += wt_obstacle    * val_obstacle      #self.get_obstacle_penalty(x, i, goal)

        stage_costs = sum([wt_legib*val_legib, wt_lam*val_lam, wt_obstacle*val_obstacle, wt_heading*val_heading])
        stage_costs = float(stage_costs)

        if stage_costs != J:
            print("alert! j math is off")
            print("J = " + str(J))
            print(stage_costs)

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

            print("[wt_legib, wt_lam, wt_obstacle, wt_heading]")
            print([wt_legib, wt_lam, wt_obstacle, wt_heading])            
            print("[val_legib, val_lam, val_obstacle, val_heading]")
            print([val_legib, val_lam, val_obstacle, val_heading])
            print("==")
            print([wt_legib*val_legib, wt_lam*val_lam, wt_obstacle*val_obstacle, wt_heading*val_heading])

            print(str(sum([wt_legib*val_legib, wt_lam*val_lam, wt_obstacle*val_obstacle, wt_heading*val_heading])))


        total = (scale_term * term_cost) + (scale_stage * stage_costs)

        return float(total)

    def f(t):
        return 1.0

    def dist_between(self, x1, x2):
        distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        return distance

    def get_relative_distance_value(self, start, goal, x, terminal, force_mode=None):
        Q = self.Q_terminal if terminal else self.Q
        dist = self.dist_between(x, goal)

        print("dist to goal " + str(goal))
        print(dist)
        if dist < 0:
            print("dist to goal less than 0!")
            exit()

        if force_mode is 'sqr':
            return decimal.Decimal(self.inversely_proportional_to_distance(dist)**2)
            # return decimal.Decimal(self.get_relative_distance_k_sqr(x, goal, self.goals))
        elif force_mode is 'lin':
            return decimal.Decimal(self.inversely_proportional_to_distance(dist))
            # return decimal.Decimal(self.get_relative_distance_k(x, goal, self.goals))

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        diff_curr   = start - x
        diff_goal   = x - goal
        diff_all    = start - goal

        diff_curr   = diff_curr.T
        diff_goal   = diff_goal.T
        diff_all    = diff_all.T

        n = - (diff_curr.T).dot(Q).dot((diff_curr)) - ((diff_goal).T.dot(Q).dot(diff_goal))
        d = (diff_all).T.dot(Q).dot(diff_all)

        n = decimal.Decimal(n)
        d = decimal.Decimal(d)

        # J = np.exp(n) / np.exp(d)

        if force_mode is not None:
            if force_mode is 'exp':
                J = np.exp(n) / np.exp(d)
            elif force_mode is 'sqr':
                pass
                # J = n / (d * d)
            elif force_mode is 'lin':
                pass
                # J = n / (d)
        else:
            J = np.abs(n / d)

        if self.exp.get_weighted_close_on() is True:
            k = self.get_relative_distance_k(x, goal, self.goals)
        else:
            k = 1.0

        J = decimal.Decimal(k)*J
        return J

    def legibility_stage_cost(self, start, goal, x, u, i_step, terminal, visibility_coeff, force_mode=None, pure_prob=False):
        P_oa = self.prob_distance(start, goal, x, u, i_step, terminal, visibility_coeff, force_mode=None, pure_prob=False)

        return decimal.Decimal(1.0) - P_oa


    def prob_distance(self, start, goal, x, u, i_step, terminal, visibility_coeff, force_mode=None, pure_prob=False):
        # TODO verify force mode is happy and correct
        if force_mode is None:
            force_mode = self.exp.get_mode_dist_type()

        if not np.array_equal(goal, self.exp.get_target_goal()):
            print("Goal and exp goal not the same")
            print(goal, self.exp.get_target_goal())            
            exit()

        if visibility_coeff == 1 or visibility_coeff == 0:
            print("vis is not 1 or 0")
            pass
        else:
            exit()

        debug_dict = {'start':start, 'goal':goal, 'all_goals':self.exp.get_goals(), 'x': x, 'u': u, 'i':i_step, 'goal': goal, 'visibility_coeff': visibility_coeff, 'force_mode':force_mode, 'pure_prob':pure_prob}
        print("DIST COST INPUTS")
        print(debug_dict)

        if np.array_equal(x, goal):
            if pure_prob:
                return .6
            else:
                return .4

        # visibility coeff is 1.0 if in vision, 0 if no
        all_goals = self.goals
        
        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("For point at x -> " + str(x))

        goal_values = []
        for alt_goal in all_goals:
            goal_val = self.get_relative_distance_value(start, alt_goal, x, terminal, force_mode=force_mode) 
            goal_values.append(goal_val)

            if np.array_equal(goal, alt_goal):
                target_val = goal_val

        print("Target val")
        print(target_val)
        print("All values")
        print(goal_values)
        total = sum([abs(ele) for ele in goal_values])
        print(total)

        # dist_prob = (total - target_val) / (total)
        dist_prob = (target_val) / total

        P_dist      = decimal.Decimal(dist_prob)
        num_goals   = len(all_goals)
        P_oa        = decimal.Decimal((1.0/num_goals)*(1.0 - visibility_coeff)) + ((decimal.Decimal(visibility_coeff) * P_dist))

        return P_oa

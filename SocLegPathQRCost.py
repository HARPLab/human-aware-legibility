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

from scipy.optimize import approx_fprime

import utility_legibility as legib
import utility_environ_descrip as resto
import pipeline_generate_paths as pipeline
import pdb

from LegiblePathQRCost import LegiblePathQRCost
import PathingExperiment as ex

class SocLegPathQRCost(LegiblePathQRCost):
    FLAG_DEBUG_J = False
    FLAG_DEBUG_STAGE_AND_TERM = True

    FLAG_COST_PATH_OVERALL  = True
    FLAG_OBS_FLAT_PENALTY   = True

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

    def get_obstacle_penalty(self, x, goal):
        TABLE_RADIUS    = self.exp.get_table_radius()
        OBS_RADIUS      = .1
        GOAL_RADIUS     = .15 #.05

        tables      = self.exp.get_tables()
        goals       = self.goals
        observers   = self.exp.get_observers()

        obstacle_penalty = 0
        for table in tables:
            obstacle = table.get_center()
            obs_dist = obstacle - x
            obs_dist = np.abs(np.linalg.norm(obs_dist))
            # Flip so edges lower cost than center

            if obs_dist < TABLE_RADIUS:
                obs_dist = TABLE_RADIUS - obs_dist
                print("table obstacle dist for " + str(x) + " " + str(obs_dist))
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
                print("obs obstacle dist for " + str(x) + " " + str(obs_dist))
                # obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                # OBSTACLE PENALTY NOW ALWAYS SCALED TO RANGE 0 -> 1
                if self.FLAG_OBS_FLAT_PENALTY:
                    obstacle_penalty += 1.0

                obstacle_penalty += np.abs(obs_dist / OBS_RADIUS) #**2

        for g in goals:
            if g is not goal:
                obstacle = g
                obs_dist = obstacle - x
                obs_dist = np.abs(np.linalg.norm(obs_dist))
                # Flip so edges lower cost than center

                if obs_dist < GOAL_RADIUS:
                    obs_dist = GOAL_RADIUS - obs_dist
                    print("goal obstacle dist for " + str(x) + " " + str(obs_dist))
                    print(str(g))
                    # obstacle_penalty += (obs_dist)**2 * self.scale_obstacle

                    # OBSTACLE PENALTY NOW ALWAYS SCALED TO RANGE 0 -> 1
                    if self.FLAG_OBS_FLAT_PENALTY:
                        obstacle_penalty += 1.0
                    obstacle_penalty += np.abs(obs_dist / OBS_RADIUS) #**2

        return obstacle_penalty


    ##### METHODS FOR ANGLE MATH
    def get_angle_between(self, p2, p1):
        # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
        ang1    = np.arctan2(*p1[::-1])
        ang2    = np.arctan2(*p2[::-1])
        heading = np.rad2deg((ang1 - ang2) % (2 * np.pi))

        # Heading is in degrees
        return heading

    def angle_diff(self, a1, a2):
        # target - source
        a = a1 - a2
        diff = (a + 180) % 360 - 180

        return diff


    def get_relative_distance_k(self, x, goal, goals):
        total_distance = 0.0

        for g in goals:
            dist = g - x
            dist = np.abs(np.linalg.norm(dist))

            total_distance += dist

        target_goal_dist = goal - x
        tg_dist = np.abs(np.linalg.norm(target_goal_dist))

        return 1.0 - (tg_dist / total_distance)


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
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def get_heading_cost(self, x, u, i, goal):
        if i is 0:
            return 0

        goals       = self.goals

        x1 = x
        if i > 0:
            x0 = self.x_path[i - 1]
        else:
            x0 = x

        print("Points in a row")
        print(x0, x1)

        robot_vector    = x1 - x0
        target_vector   = None
        all_goal_vectors    = []

        for alt_goal in goals:
            goal_vector = alt_goal - x1 #[x1, alt_goal]

            if alt_goal is goal:
                target_vector = goal_vector
            all_goal_vectors.append(goal_vector)

        print("robot vector")
        print(robot_vector)

        print("all goal vectors")
        print(all_goal_vectors)

        all_goal_angles   = []
        for gvec in all_goal_vectors:
            goal_angle = self.angle_between(robot_vector, gvec)
            all_goal_angles.append(goal_angle)

        target_angle = self.angle_between(robot_vector, target_vector)

        print("all target angles")
        print(all_goal_angles)

        angles_squared = []
        for i in range(len(all_goal_angles)):
            gangle = all_goal_angles[i]
            gang_sqr = gangle * gangle

            if self.exp.get_weighted_close_on() is True:
                k = self.get_relative_distance_k(x1, goals[i], goals)
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

    def get_heading_cost_v1(self, x, u, i, goal):
        if i is 0:
            return 0

        goals       = self.goals

        x1 = x
        if i > 0:
            x0 = self.x_path[i - 1]
        else:
            x0 = x

        print("Points in a row")
        print(x0, x1)

        robot_heading = self.get_angle_between(x0, x1)
        alt_goal_headings = []

        for alt_goal in goals:
            goal_heading = self.get_angle_between(x1, alt_goal)

            if alt_goal is goal:
                target_heading = goal_heading
            else:
                alt_goal_headings.append(goal_heading)

            print(alt_goal)
            print(" is at heading ")
            print(goal_heading)

        good_part = 180 - np.abs(self.angle_diff(robot_heading, target_heading))
        good_part = good_part**2

        bad_parts = 0
        total = good_part
        alt_goal_part_log = []

        for i in range(len(alt_goal_headings)):
            alt_head = alt_goal_headings[i]
            bad_part = 180 - np.abs(self.angle_diff(robot_heading, alt_head))
            bad_part = bad_part**2

            print("Part 1")
            print(180 - np.abs(self.angle_diff(robot_heading, alt_head)))
            print(180 - np.abs(self.angle_diff(robot_heading, alt_head)))
            print("Part 2 = squared")
            print(bad_part)


            if self.exp.get_weighted_close_on() is True:
                k = self.get_relative_distance_k(x, goals[i], goals)
            else:
                k = 1.0

            # scale either evenly, or proportional to closeness
            bad_part = bad_part * k

            # bad_part += 1.0 / self.angle_diff(robot_heading, alt_head)
            bad_parts += bad_part
            total += bad_part
            alt_goal_part_log.append(bad_part)
            print("For goal at alt heading " + str(alt_head))
            print(bad_part) 

        print("Total is " + str(total))
        print(type(total))

        # fix to nan so there's no divide by zero error
        if total == 0.0:
            print("total is now 1.0 to avoid nan error")
            total += 1.0

        heading_clarity_cost = bad_part / (total)
        alt_goal_part_log = alt_goal_part_log / (total)

        print("Heading component of pathing ")
        print("Given x of " + str(x) + " and robot heading of " + str(robot_heading))
        print("for goals " + str(goals))
        print(alt_goal_part_log)
        print(heading_clarity_cost)
        print("good parts, bad parts")
        print(good_part, bad_parts)

        return heading_clarity_cost


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

        scale_term = self.exp.get_solver_scale_term() #0.01 # 1/100
        scale_stage = self.exp.get_solver_scale_stage() #1.5

        if just_term:   
            scale_stage = 0

        if just_stage:
            scale_term  = 0

        term_cost = 0 #self.term_cost(x, i)

        # # xdiff from preferred line
        # x_path[i] is always the goal
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

        # VISIBILITY COMPONENT
        restaurant  = self.exp.get_restaurant()
        observers   = self.exp.get_observers()

        ### USE ORIGINAL LEGIBILITY WHEN THERE ARE NO OBSERVERS
        if self.exp.get_is_oa_on() is True:
            if len(observers) > 0:
                visibility  = legib.get_visibility_of_pt_w_observers_ilqr(x, observers, normalized=True)
            else:
                visibility  = 1.0
        else:
            visibility = 1.0

        FLAG_OA_MIN_VIS = False

        # if FLAG_OA_MIN_VIS:
        #     if visibility == 0:
        #         visibility = .01

        # f_func     = self.get_f()
        # f_value    = f_func(i)

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


        if self.exp.get_norm_on() is False:
            wt_legib     = 0.8 #100.0
            wt_lam       = 0.01
            wt_heading   = 0.2 #100000.0
            wt_obstacle  = 100000.0 #self.exp.get_solver_scale_obstacle()

        else:
            ##### SET WEIGHTS
            wt_legib     = f_value * .9 #1000.0
            wt_lam       = .1 * (1 - wt_legib)
            wt_heading   = .1 * (1 - wt_legib) #100000.0
            wt_obstacle  = 100000.0 #self.exp.get_solver_scale_obstacle()

        if self.exp.get_is_heading_on() is False:
            wt_heading = 0.0

        if self.exp.get_mode_pure_heading() is True:
            wt_legib = 0.0


        # BATCH 2
        # # NORMALIZED AROUND IN/OUT OF SIGHT
        # wt_legib     = f_value * .9 #1000.0
        # wt_lam       = 1.5 * (1 - wt_legib)
        # # wt_control   = .4 * (1 - wt_legib)
        # wt_heading   = .5 * (1 - wt_legib) #100000.0
        # wt_obstacle  = 10000.0 #self.exp.get_solver_scale_obstacle()

        # # BATCH 3 2:43pm
        # # NORMALIZED AROUND IN/OUT OF SIGHT
        # wt_legib     = f_value * .3 #1000.0
        # wt_lam       = 1.5 * (1 - wt_legib)
        # # wt_control   = .4 * (1 - wt_legib)
        # wt_heading   = .5 * (1 - wt_legib) #100000.0
        # wt_obstacle  = 10000.0 #self.exp.get_solver_scale_obstacle()


        # J does not need to be in a particular range, it can be any max or min
        J = 0        
        J += (wt_legib      * self.michelle_stage_cost(start, goal, x, u, i, terminal))
        J += (wt_lam        * u_diff.T.dot(R).dot(u_diff))
        # J += (wt_lam        * x_diff.T.dot(Q).dot(x_diff))
        J += (wt_obstacle)  * self.get_obstacle_penalty(x, goal)
        J += (wt_heading)   * self.get_heading_cost(x, u, i, goal)

        stage_costs = J

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("STAGE,\t TERM")
            print(stage_costs, term_cost)

        total = (scale_term * term_cost) + (scale_stage * stage_costs)

        return float(total)

    # # original version for plain path following
    # def l(self, x, u, i, terminal=False, just_term=False, just_stage=False):
    #     """Instantaneous cost function.
    #     Args:
    #         x: Current state [state_size].
    #         u: Current control [action_size]. None if terminal.
    #         i: Current time step.
    #         terminal: Compute terminal cost. Default: False.
    #     Returns:
    #         Instantaneous cost (scalar).
    #     """
    #     Q = self.Qf if terminal else self.Q
    #     R = self.R
    #     start = self.start
    #     goal = self.target_goal
    #     thresh = .0001

    #     term_cost = 0

    #     if terminal or just_term: # or abs(i - self.N) < thresh:
    #         return self.term_cost(x, i) #*1000
    #     else:
    #         if self.FLAG_DEBUG_STAGE_AND_TERM:
    #             # difference between this step and the end
    #             print("x, N, x_end_of_path -> inputs and then term cost")
    #             print(x, self.N, self.x_path[self.N])
    #             # term_cost = self.term_cost(x, i)


    #     # VISIBILITY COMPONENT
    #     restaurant  = self.exp.get_restaurant()
    #     observers   = restaurant.get_observers()

    #     visibility  = 1 #legib.get_visibility_of_pt_w_observers_ilqr(x, observers, normalized=True)
    #     FLAG_OA_MIN_VIS = False

    #     if FLAG_OA_MIN_VIS:
    #         if visibility == 0:
    #             visibility = .01

    #     # f_func     = self.get_f()
    #     # f_value    = f_func(i)

    #     # if f_value != visibility:
    #     #     exit()

    #     f_value    = visibility

    #     # PATH COST PENALTY COMPONENT
    #     x_diff = x - self.x_path[self.N]
    #     Q_path_cost = np.identity(2)
    #     path_squared_x_cost = .5 * x_diff.T.dot(Q_path_cost).dot(x_diff)

    #     stage_costs = self.michelle_stage_cost(start, goal, x, u, i, terminal) * f_value
    #     stage_costs = stage_costs + self.exp.get_lambda_cost_path_coeff() * path_squared_x_cost
    
    #     if self.FLAG_DEBUG_STAGE_AND_TERM:
    #         print("STAGE,\t TERM")
    #         print(stage_costs, term_cost)

    #     # Log of remixes of term and stage cost weightings
    #     # term_cost      = decimal.Decimal.ln(decimal.Decimal(term_cost)) 
    #     # stage_costs    = decimal.Decimal.ln(stage_costs)
    #     # if i < 30:
    #     #     stage_scale = 200
    #     #     term_scale = 0.1
    #     # else:
    #     #     stage_scale = 10
    #     #     term_scale = 1
    #     # stage_scale = max([(self.N - i), 20])
    #     # term_scale = 100
    #     # stage_scale = 10
    #     # term_scale = 1
    #     # stage_scale = max([self.N-i, 10])
    #     # stage_scale = abs(self.N-i)
    #     # term_scale = i/self.N
    #     # term_scale = 1
    #     # stage_scale = 50

    #     scale_term  = self.scale_term #0.01 # 1/100
    #     scale_stage = self.scale_stage #1.5

    #     if just_term:   
    #         scale_stage = 0

    #     if just_stage:
    #         scale_term  = 0

    #     total = (scale_term * term_cost) + (scale_stage * stage_costs)

    #     return float(total)

    def f(t):
        return 1.0

    def get_total_stage_cost(self, start, goal, x, u, i, terminal):
        N = self.N
        R = self.R

        stage_costs = 0.0 #u_diff.T.dot(R).dot(u_diff)
        
        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("u = " + str(u))
            print("Getting stage cost")

        for j in range(i):
            u_diff = u - self.u_path[j]

            stage_costs += (0.5 * u_diff.T.dot(R).dot(u_diff))

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("total stage cost " + str(stage_costs))

        return stage_costs

    def michelle_stage_cost(self, start, goal, x, u, i, terminal=False):
        Q = self.Q_terminal if terminal else self.Q
        R = self.R

        all_goals = self.goals

        goal_diff   = start - goal
        start_diff  = (start - np.array(x))
        togoal_diff = (np.array(x) - goal)

        u_diff = u - self.u_path[i]

        if len(self.u_path) == 0:
            return 0

        a = (goal_diff.T).dot(Q).dot((goal_diff))
        b = (start_diff.T).dot(Q).dot((start_diff))
        c = (togoal_diff.T).dot(Q).dot((togoal_diff))

        # (start-goal1)'*Q*(start-goal1) - (start-x)'*Q*(start-x) +  - (x-goal1)'*Q*(x-goal1) 
        J_g1 = a - b - c
        # J_g1 *= .5

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("For point at x -> " + str(x))
            # print("Jg1 " + str(J_g1))

        log_sum = 0.0
        total_sum = 0.0
        for alt_goal in all_goals:
            # n = - ((start-x)'*Q*(start-x) + 5) - ((x-goal)'*Q*(x-goal)+10)
            # d = (start-goal)'*Q*(start-goal)
            # log_sum += (exp(n )/exp(d))* scale

            
            diff_curr   = start - x
            diff_goal   = x - alt_goal
            diff_all    = start - alt_goal

            diff_curr   = diff_curr.T
            diff_goal   = diff_goal.T
            diff_all    = diff_all.T

            n = - (diff_curr.T).dot(Q).dot((diff_curr)) - ((diff_goal).T.dot(Q).dot(diff_goal))
            d = (diff_all).T.dot(Q).dot(diff_all)

            this_goal = np.exp(n) / np.exp(d)

            if self.exp.get_weighted_close_on() is True:
                k = self.get_relative_distance_k(x, alt_goal, self.goals)
            else:
                k = 1.0

            this_goal = this_goal * k

            total_sum += this_goal

            print("n: " + str(n) + ", d: " + str(d))
            print("thisgoal: " + str(this_goal))


            if goal != alt_goal:
                log_sum += (1 * this_goal)
                if self.FLAG_DEBUG_STAGE_AND_TERM:
                    # print("Value for alt target goal " + str(alt_goal))
                    print("This is the nontarget goal: " + str(alt_goal) + " -> " + str(this_goal))
            else:
                # print("Value for our target goal " + str(goal))
                # J_g1 = this_goal
                log_sum += this_goal
                if self.FLAG_DEBUG_STAGE_AND_TERM:
                    print("This is the target goal " + str(alt_goal) + " -> " + str(this_goal))
    

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("Jg1, total")
            print(J_g1, total_sum)

        J = J_g1 - (np.log(total_sum))
        J = -1.0 * J

        if self.FLAG_DEBUG_STAGE_AND_TERM:
            print("overall J " + str(J))

        J += (0.5 * u_diff.T.dot(R).dot(u_diff))

        # # We want the path to be smooth, so we incentivize small and distributed u

        return J

    def stage_cost(self, x, u, i, terminal=False):
        print("DOING STAGE COST")
        start   = self.start
        goal    = self.target_goal

        x = np.array(x)
        J = self.goal_efficiency_through_point_relative(start, goal, x, terminal)
        return J

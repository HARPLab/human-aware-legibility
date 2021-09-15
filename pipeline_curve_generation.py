import table_path_code as resto
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import cv2
import matplotlib.pylab as plt
import math
import copy
import time
import decimal
import random
import os
from pandas.plotting import table
import matplotlib.gridspec as gridspec
import klampt_smoothing as chunkify
from collections import defaultdict
from shapely.geometry import LineString
import scipy.interpolate as interpolate

import sys
# sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/ModelPredictiveTrajectoryGenerator/')
# sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/StateLatticePlanner/')

# start 		= r.get_start()
# goals 		= r.get_goals_all()
# goal 		= r.get_current_goal()
# observers 	= r.get_observers()
# tables 		= r.get_tables()
# waypoints 	= r.get_waypoints()
# SCENARIO_IDENTIFIER = r.get_scenario_identifier()

FLAG_SAVE 				= True
FLAG_VIS_GRID 			= False
FLAG_EXPORT_HARDCODED 	= False
FLAG_REDO_PATH_CREATION = False #True #False #True #False
FLAG_REDO_ENVIR_CACHE 	= False #True #False #True
FLAG_MIN_MODE			= False

VISIBILITY_TYPES 		= resto.VIS_CHECKLIST
NUM_CONTROL_PTS 		= 3

NUMBER_STEPS = 30

PATH_TIMESTEPS = 15

resto_pickle = 'pickle_vis'
vis_pickle = 'pickle_resto'
FILENAME_PATH_ASSESS = 'path_assessment/'

FLAG_PROB_HEADING = False
FLAG_PROB_PATH = True
FLAG_EXPORT_SPLINE_DEBUG = False

# PATH_COLORS = [(138,43,226), (0,255,255), (255,64,64), (0,201,87)]

SAMPLE_TYPE_CENTRAL 	= 'central'
SAMPLE_TYPE_CENTRAL_SPARSE 	= 'central-sparsy'
SAMPLE_TYPE_DEMO 		= 'demo'
SAMPLE_TYPE_CURVE_TEST	= 'ctest'
SAMPLE_TYPE_NEXUS_POINTS= 'nn_fin4'
SAMPLE_TYPE_SPARSE		= 'sparse'
SAMPLE_TYPE_SYSTEMATIC 	= 'systematic'
SAMPLE_TYPE_HARDCODED 	= 'hardcoded'
SAMPLE_TYPE_VISIBLE 	= 'visible'
SAMPLE_TYPE_INZONE 		= 'in_zone'
SAMPLE_TYPE_SHORTEST	= 'minpaths'
SAMPLE_TYPE_FUSION		= 'fusion'


ENV_START_TO_HERE 		= 'start_to_here'
ENV_HERE_TO_GOALS 		= 'here_to_goals'
ENV_VISIBILITY_PER_OBS 	= 'vis_per_obs'

premade_path_sampling_types = [SAMPLE_TYPE_DEMO, SAMPLE_TYPE_SHORTEST, SAMPLE_TYPE_CURVE_TEST]
non_metric_columns = ["path", "goal", 'path_length', 'path_cost', 'sample_points']

bug_counter = defaultdict(int)
curvatures = []
max_curvatures = []


def f_cost(t1, t2):
	a = resto.dist(t1, t2)
	# return a
	return np.abs(a * a)

def f_path_length(t1, t2):
	a = resto.dist(t1, t2)
	return a
	# return np.abs(a * a)

def f_path_cost(path):
	cost = 0
	for i in range(len(path) - 1):
		cost = cost + f_cost(path[i], path[i + 1])

	return cost

def f_convolved(val_list, f_function):
	tstamps = range(len(val_list))
	ret = []
	for t in tstamps:
		ret.append(f_function(t) * val_list[t])
	return ret

def f_vis_exp1(t, pt, aud):
	return (f_og(t) * f_vis3(pt, aud))


def f_og(t, path):
	# len(path)
	return NUMBER_STEPS - t

def f_novis(t, obs):
	return 1

# # Given the observers of a given location, in terms of distance and relative heading
# # Ada final equation TODO verify all correct
# def f_vis_single(p, observers):
# 	# dist_units = 100
# 	angle_cone = 120.0 / 2
# 	distance_cutoff = 2000

# 	# Given a list of entries in the format 
# 	# ((obsx, obsy), angle, distance)
# 	if len(observers) == 0:
# 		return 1
	
# 	vis = 0
# 	for obs in observers:
# 		if obs == None:
# 			return 0
# 		else:
# 			angle, dist = obs.get_obs_to_pt_relationship(p)
# 			# print((angle, dist))

# 		if angle < angle_cone and dist < distance_cutoff:
# 			vis += np.abs(angle_cone - angle)

# 	# print(vis)
# 	return vis

def f_naked(t, pt, aud, path):
	return decimal.Decimal(1.0)

# Ada final equation
# f_VIS TODO VERIFY
def f_exp_single(t, pt, aud, path):
	# if this is the omniscient case, return the original equation
	if len(aud) == 0 and path is not None:
		return float(60 - t)
		# return float(len(path) - t)
	elif len(aud) == 0:
		# print('ping')
		return 1.0

	# if in the (x, y) OR (x, y, t) case we can totally 
	# still run this equation
	val = get_visibility_of_pt_w_observers(pt, aud, normalized=False)
	return val

def f_exp_single_normalized(t, pt, aud, path):
	# if this is the omniscient case, return the original equation
	if len(aud) == 0 and path is not None:
		return float(len(path) - t + 1)
		# return float(len(path) - t)
	elif len(aud) == 0:
		# print('ping')
		return 1.0

	# if in the (x, y) OR (x, y, t) case we can totally 
	# still run this equation
	val = get_visibility_of_pt_w_observers(pt, aud, normalized=True)
	return val


# ADA TODO MASTER VISIBILITY EQUATION
def get_visibility_of_pt_w_observers(pt, aud, normalized=True):
	observers = []
	score = []

	reasonable_set_sizes = [0, 1, 5]
	if len(aud) not in reasonable_set_sizes:
		print(len(aud))
		exit()

	# section for alterating calculculation for a few 
	# out of the whole set; mainly for different combination techniques
	# if len(aud) == 5:
	# 	aud = [aud[2], aud[4]]

	MAX_DISTANCE = 500
	for observer in aud:
		obs_orient 	= observer.get_orientation() + 90
		# if obs_orient != 300:
		# 	print(obs_orient)
		# 	exit()
		obs_FOV 	= observer.get_FOV()

		angle 		= angle_between_points(observer.get_center(), pt)
		distance 	= resto.dist(pt, observer.get_center())
		# print("~~~")
		# print(observer.get_center())
		# print(distance)
		# print(pt)
		
		# print(ang)
		a = angle - obs_orient
		signed_angle_diff = (a + 180) % 360 - 180
		angle_diff = abs(signed_angle_diff)

		# if (pt[0] % 100 == 0) and (pt[1] % 100 == 0):
		# 	print(str(pt) + " -> " + str(observer.get_center()) + " = angle " + str(angle))
		# 	print("observer looking at... " + str(obs_orient))
		# 	print("angle diff = " + str(angle_diff))

		# print(angle, distance)
		# observation = (pt, angle, distance)
		# observers.append(observation)

		half_fov = (obs_FOV / 2.0)
		# print(half )
		if angle_diff < half_fov:
			from_center = half_fov - angle_diff
			if normalized:
				from_center = from_center / (half_fov)

			# from_center = from_center * from_center
			score.append(from_center)
		else:
			if normalized:
				score.append(0)
			else:
				score.append(1)

		# 	# full credit at the center of view
		# 	offset_multiplier = np.abs(angle_diff) / obs_FOV

		# 	# # 1 if very close
		# 	# distance_bonus = (MAX_DISTANCE - distance) / MAX_DISTANCE
		# 	# score += (distance_bonus*offset_multiplier)
		# 	score = offset_multiplier
		# 	score = distance

	# combination method for multiple viewers: minimum value
	if len(score) > 0:
		# score = min(score)
		score = sum(score)
	else:
		score = 0
	return score

# Ada: Final equation
# TODO Cache this result for a given path so far and set of goals
def prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):
	start = r.get_start()
	g_array = []
	g_target = 0
	for g in goals:
		p_raw = unnormalized_prob_goal_given_path(r, p_n1, pt, g, goals, cost_path_to_here, exp_settings)
		g_array.append(p_raw)
		if g is goal:
			g_target = p_raw

	if(sum(g_array) == 0):
		print("weird g_array")
		return 0

	return g_target / (sum(g_array))

# Ada: final equation
def unnormalized_prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):
	decimal.getcontext().prec = 60
	is_og = exp_settings['prob_og']

	start = r.get_start()

	if is_og:
		c1 = decimal.Decimal(cost_path_to_here)
	else:
		c1 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(r.get_start()), resto.to_xy(pt), exp_settings))	

	
	c2 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(pt), resto.to_xy(goal), exp_settings))
	c3 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(start), resto.to_xy(goal), exp_settings))

	# print(c2)
	# print(c3)
	a = np.exp((-c1 + -c2))
	b = np.exp(-c3)
	# print(a)
	# print(b)

	ratio 		= a / b

	if math.isnan(ratio):
		ratio = 0

	return ratio

def prob_goal_given_heading(start, pn, pt, goal, goals, cost_path_to_here):

	g_probs = prob_goals_given_heading(pn, pt, goals)
	g_index = goals.index(goal)

	return g_probs[g_index]


def f_angle_prob(heading, goal_theta):
	diff = (np.abs(np.abs(heading - goal_theta) - 180))
	return diff * diff


def prob_goals_given_heading(p0, p1, goals):
	# works with
	# diff = (np.abs(np.abs(heading - goal_theta) - 180))

	# find the heading from start to pt
	# start to pt
	# TODO theta
	heading = resto.angle_between(p0, p1)
	# print("heading: " + str(heading))
	
	# return an array of the normalized prob of each goal from this current heading
	# for each goal, find the probability this angle is pointing to it



	probs = []
	for goal in goals:
		# 180 		= 0
		# 0 or 360 	= 1
		# divide by other options, 
		# so if 2 in same dir, 50/50 odds
		goal_theta = resto.angle_between(p0, goal)
		prob = f_angle_prob(heading, goal_theta)
		probs.append(prob)


	divisor = sum(probs)
	# divisor = 1.0

	return np.true_divide(probs, divisor)
	# return ratio


def tests_prob_heading():
	p1 = (0,0)
	p2 = (0,1)

	goals 	= [(1,1), (-1,1), (0, -1)]
	correct = [ 0.5,  0.5, -0.]
	result 	= prob_goals_given_heading(p1, p2, goals)

	if np.array_equal(correct, result):
		pass
	else:
		print("Error in heading probabilities")

	goals 	= [(4,0), (5,0)]
	correct = [ 0.5,  0.5]
	result 	= prob_goals_given_heading(p1, p2, goals)

	print(goals)
	print(result)

	goals 	= [(4,1), (4,1)]
	correct = [ 0.5,  0.5]
	result 	= prob_goals_given_heading(p1, p2, goals)	

	print(goals)
	print(result)

	print("ERR")
	goals 	= [(1,1), (-1,1), (-1,-1), (1,-1)]
	correct = [ 0.5,  0.5, 0, 0]
	result 	= prob_goals_given_heading(p1, p2, goals)	

	print(goals)
	print(result)

	goals 	= [(2,0), (0,2), (-2,0), (0,-2)]
	correct = [ 0.25,  0.5, 0.25, 0]
	result 	= prob_goals_given_heading(p1, p2, goals)	

	print(goals)
	print(result)


def get_costs_along_path(path):
	output = []
	ci = 0
	csf = 0
	for pi in range(len(path)):
		
		cst = f_cost(path[ci], path[pi])
		csf = csf + cst
		log = (path[pi], csf)
		ci = pi
		output.append(log)
		
	return output

# returns a list of the path length so far at each point
def get_path_length(path):
	total = 0
	output = [0]

	for i in range(len(path) - 1):
		link_length = f_path_length(path[i], path[i + 1])
		total = total + link_length
		output.append(total)

	return output, total

	# output = []
	# ci = 0
	# csf = 0
	# total = 0
	# for pi in range(len(path)):
	# 	cst = f_path_length(path[ci], path[pi])
	# 	total += cst
	# 	ci = pi
	# 	output.append(total) #log
		
	# return output, total

def get_min_viable_path(r, goal, exp_settings):
	path_option = construct_single_path_with_angles_spline(exp_settings, r.get_start(), goal, [], fn_export_from_exp_settings(exp_settings))
	path_option = chunkify.chunkify_path(exp_settings, path_option)
	return path_option

def get_min_viable_path_length(r, goal, exp_settings):
	path_option = get_min_viable_path(r, goal, exp_settings)
	return get_path_length(path_option)[1]

def get_min_direct_path(r, p0, p1, exp_settings):
	path_option = [p0, p1]
	path_option = chunkify.chunkify_path(exp_settings, path_option)
	return path_option

def get_dist(p0, p1):
	p0_x, p0_y = p0
	p1_x, p1_y = p1

	min_distance = np.sqrt((p0_x-p1_x)**2 + (p0_y-p1_y)**2)
	return min_distance

def get_min_direct_path_cost_between(r, p0, p1, exp_settings):
	dist = get_dist(p0, p1)
	dt = chunkify.get_dt(exp_settings)
	cost_chunk = dt * dt
	num_chunks = int()

	leftover = dist - (dt*num_chunks)
	cost = (num_chunks * cost_chunk) + (leftover*leftover)

	return cost
	# f_path_cost(path_option)

def get_min_direct_path_length(r, p0, p1, exp_settings):
	return get_dist(p0, p1)

# Given a 
def f_legibility(r, goal, goals, path, aud, f_function, exp_settings):
	FLAG_is_denominator = exp_settings['is_denominator']
	if f_function is None and FLAG_is_denominator:
		f_function = f_exp_single
	elif f_function is None:
		f_function = f_exp_single_normalized

	if path is None or len(path) == 0:
		return 0
	min_realistic_path_length = exp_settings['min_path_length'][goal]
	# print("min_realistic_path_length -> " + str(min_realistic_path_length))
	
	legibility = decimal.Decimal(0)
	divisor = decimal.Decimal(0)
	total_dist = decimal.Decimal(0)

	LAMBDA = decimal.Decimal(exp_settings['lambda'])
	epsilon = decimal.Decimal(exp_settings['epsilon'])

	start = path[0]
	total_cost = decimal.Decimal(0)
	aug_path = get_costs_along_path(path)

	path_length_list, length_of_total_path = get_path_length(path)
	length_of_total_path = decimal.Decimal(length_of_total_path)

	# Previously this was a variable, 
	# now it's constant due to our constant-speed chunking
	delta_x = decimal.Decimal(1.0) #length_of_total_path / len(aug_path)

	t = 1
	p_n = path[0]
	divisor = epsilon
	numerator = decimal.Decimal(0.0)

	f_log = []
	p_log = []
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))
		prob_goal_given = prob_goal_given_path(r, p_n, pt, goal, goals, cost_to_here, exp_settings)
		f_log.append(float(f))
		p_log.append(prob_goal_given)

		if prob_goal_given > 1 or prob_goal_given < 0:
			print(prob_goal_given)
			print("!!!")

		if FLAG_is_denominator or len(aud) == 0:
			numerator += (prob_goal_given * f) # * delta_x)
			divisor += f #* delta_x
		else:
			numerator += (prob_goal_given * f) # * delta_x)
			divisor += decimal.Decimal(1.0) #* delta_x

		t = t + 1
		total_cost += decimal.Decimal(f_cost(p_n, pt))
		p_n = pt

	if divisor == 0:
		legibility = 0
	else:
		legibility = (numerator / divisor)

	total_cost =  - LAMBDA*total_cost
	overall = legibility + total_cost

	# if len(aud) == 0:
	# 	print(numerator)
	# 	print(divisor)
	# 	print(f_log)
	# 	print(p_log)
	# 	print(legibility)
	# 	print(overall)
	# 	print()

	if legibility > 1.0 or legibility < 0:
		# print("BAD L ==> " + str(legibility))
		# r.get_obs_label(aud)
		goal_index = r.get_goal_index(goal)
		category = r.get_obs_label(aud)
		bug_counter[goal_index, category] += 1

	elif (legibility == 1):
		goal_index = r.get_goal_index(goal)
		category = r.get_obs_label(aud)
		bug_counter[goal_index, category] += 1

		# print(len(aud))
		if exp_settings['kill_1'] == True:
			overall = 0.0

	return overall

# Given a 
def f_env(r, goal, goals, path, aud, f_function, exp_settings):
	fov = exp_settings['fov']
	FLAG_is_denominator = exp_settings['is_denominator']
	if path is None or len(path) == 0:
		return 0

	if f_function is None and FLAG_is_denominator:
		f_function = f_exp_single
	elif f_function is None:
		f_function = f_exp_single_normalized

	if FLAG_is_denominator:
		vis_cutoff = 1
	else:
		half_fov = fov / 2.0
		vis_cutoff = 0

	count = 0
	aug_path = get_costs_along_path(path)

	path_length_list, length_of_total_path = get_path_length(path)
	length_of_total_path = decimal.Decimal(length_of_total_path)

	epsilon = exp_settings['epsilon']

	t = 1
	p_n = path[0]
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))
	
		# if f is greater than 0, this indicates being in-view
		if f > vis_cutoff:
			count += 1

		# if it's not at least 0, then out of sight, not part of calc
		else:
			count = 0.0

		t += 1

	return count


def get_costs(path, target, obs_sets):
	vals = []

	for aud in obs_sets:
		new_val = f_cost()

	return vals

def get_legibilities(resto, path, target, goals, obs_sets, f_vis, exp_settings):
	vals = {}
	f_vis = exp_settings['f_vis']

	# print("manually: naked")
	naked_prob = f_legibility(resto, target, goals, path, [], f_naked, exp_settings)
	vals['naked'] = naked_prob

	for key in obs_sets.keys():
		aud = obs_sets[key]
		new_leg = f_legibility(resto, target, goals, path, aud, None, exp_settings)
		new_env = f_env(resto, target, goals, path, aud, f_vis, exp_settings)

		vals[key] = new_leg
		vals[key + "-env"] = new_env

	return vals


# https://medium.com/@jaems33/understanding-robot-motion-path-smoothing-5970c8363bc4
def smooth_slow(path, weight_data=0.5, weight_smooth=0.1, tolerance=1):
	"""
	Creates a smooth path for a n-dimensional series of coordinates.
	Arguments:
		path: List containing coordinates of a path
		weight_data: Float, how much weight to update the data (alpha)
		weight_smooth: Float, how much weight to smooth the coordinates (beta).
		tolerance: Float, how much change per iteration is necessary to keep iterating.
	Output:
		new: List containing smoothed coordinates.
	"""

	dims = len(path[0])
	new = [[0, 0]] * len(path)
	# print(new)
	change = tolerance

	while change >= tolerance:
		change = 0.0
		prev_change = change
		
		for i in range(1, len(new) - 1):
			for j in range(dims):

				x_i = path[i][j]
				y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j]

				y_i_saved = y_i
				y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
				new[i][j] = y_i

				change += abs(y_i - y_i_saved)

		# print(change)
		if prev_change == change:
			return new
	return new


def smoothed(blocky_path, r):
	points = []
	
	xys = blocky_path

	ts = [t/NUMBER_STEPS for t in range(NUMBER_STEPS + 1)]
	bezier = resto.make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points


	return smooth(blocky_path)
	return blocky_path

def generate_single_path_grid(restaurant, target, vis_type, n_control):
	sample_pts 	= restaurant.sample_points(n_control, target, vis_type)
	path = construct_single_path(restaurant.get_start(), target, sample_pts)
	path 		= smoothed(path, restaurant)
	return path


def generate_single_path(restaurant, target, vis_type, n_control):
	valid_path = False

	while (not valid_path):
		sample_pts 	= restaurant.sample_points(n_control, target, vis_type)
		path 		= construct_single_path_bezier(restaurant.get_start(), target, sample_pts)
		valid_path  = is_valid_path(restaurant, path)

		if (not valid_path):
			# print("regenerating")
			pass

	return path

def at_pt(a, b, tol):
	return (abs(a - b) < tol)

def construct_single_path(start, end, sample_pts):
	points = [start]
	GRID_SIZE = 10
	# NUMBER_STEPS should be the final length

	# randomly walk there
	cx, cy, ctheta = start
	targets = sample_pts + [end]

	# print(sample_pts)

	for target in targets:
		tx, ty = resto.to_xy(target)
		x_sign, y_sign = 1, 1
		if tx < cx:
			x_sign = -1
		if ty < cy:
			y_sign = -1

		counter = 0

		# print("cx:" + str(cx) + " tx:" + str(tx))
		# print("cy:" + str(cy) + " ty:" + str(ty))

		# print(not at_pt(cx, tx, GRID_SIZE))
		# print(not at_pt(cy, ty, GRID_SIZE))

		# Abs status 
		while not at_pt(cx, tx, GRID_SIZE) or not at_pt(cy, ty, GRID_SIZE):
			# print("in loop")
			counter = counter + 1
			axis = random.randint(0, 1)
			if axis == 0 and not at_pt(cx, tx, GRID_SIZE):
				cx = cx + (x_sign * GRID_SIZE)
			elif not at_pt(cy, ty, GRID_SIZE):
				cy = cy + (y_sign * GRID_SIZE)

			new_pt = (cx, cy)
			points.append(new_pt)

		points.append(target)

	return points

def get_max_turn_along_path(path):
	angle_list = []
	is_counting = False
	for i in range(len(path) - 4):
		p1, p2, p3 = path[i], path[i + 2], path[i + 4]
		angle = angle_of_turn([p1, p2], [p2, p3])
		print(str((p1, p2, p3)) + "->" + str(angle))

		if resto.dist(p1,p2) > 2 or resto.dist(p2, p3) > 2:
			angle_list.append(abs(angle))
			curvatures.append(angle)
		# else:
		# 	print("too short, rejected")
	
	# print(angle_list)
	max_curvature = max(angle_list)
	# min_curvature = min(angle_list)
	# print(max_curvature)
	max_curvatures.append(max_curvature)
	# print(angle_list.index(max_curvature))

	return max_curvature

# def check_curvature(path):
# 	lx = [x for x,y in path]
# 	ly = [y for x,y in path]

# 	#first derivatives 
# 	dx= np.gradient(lx)
# 	dy = np.gradient(ly)

# 	#second derivatives 
# 	d2x = np.gradient(dx)
# 	d2y = np.gradient(dy)

# 	#calculation of curvature from the typical formula
# 	curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
# 	# curvature = curvature[~np.isnan(curvature)]
# 	curvature = curvature[2:-2]
# 	print(curvature)
# 	max_curvature = max(curvature)
# 	print(max_curvature)

# 	# curvatures.append(max_curvature)
# 	return max_curvature

def get_hi_low_of_pts(r):
	pt_list = copy.copy(r.get_goals_all())
	pt_list.append(r.get_start())

	first = pt_list[0]
	px, py, ptheta = first
	low_x, hi_x = px, px
	low_y, hi_y = py, py

	for pt in pt_list:
		px, py = pt[0], pt[1]

		if low_x > px:
			low_x = px

		if low_y > py:
			low_y = py

		if hi_x < px:
			hi_x = px

		if hi_y < py:
			hi_y = py

	return low_x, hi_x, low_y, hi_y


def is_valid_path(r, path, exp_settings):
	tables = r.get_tables()
	# print(len(tables))

	start = r.get_start()
	sx, sy, stheta = start
	gx0, gy0, gt0 = r.get_goals_all()[0]
	gx1, gy1, gt1 = r.get_goals_all()[1]
	# print("sampling central")
	
	low_x, hi_x, low_y, hi_y = get_hi_low_of_pts(r)

	for p in path:
		if p[0] < start[0] - 2:
			# print(p)
			return False

	line = LineString(path)
	if not line.is_simple:
		return False

	# max_turn = get_max_turn_along_path(path)
	# if max_turn >= exp_settings['angle_cutoff']:
	# 	return False

	# Checks for table intersection
	for t in tables:
		if t.intersects_path(path):
			return False

	BOUND_CHECK_RIGHT = True
	right_buffer = exp_settings['right-bound']
	# Checks for remaining in bounds
	
	for i in range(len(path) - 1):
		pt1 = path[i]
		pt2 = path[i + 1]
		
		# print((pt1, pt2))

		px, py = pt1[0], pt1[1]

		if BOUND_CHECK_RIGHT:
			if px > hi_x + right_buffer:
				return False

		if px < 0:
			return False
		if py < 0:
			return False
		if px > 1350:
			return False
		if py > 1000:
			return False




	return True

def as_tangent(start_angle):
	# start_angle is assumed to be in degrees
	a = np.deg2rad(start_angle)

	dx = np.cos(a)
	dy = np.sin(a)

	return [dx, dy]

def as_tangent_test(sa):
	sa = 90
	print(sa)
	print(as_tangent(sa))
	sa = 0
	print(sa)
	print(as_tangent(sa))

def path_formatted(xs, ys):
	# print(ys)
	xs = [int(x) for x in xs]
	ys = [int(y) for y in ys]
	return list(zip(xs, ys))

def get_pre_goal_pt(goal, exp_settings):
	x, y, theta = goal
	k = exp_settings['angle_strength']
	# print(k)

	if theta == resto.DIR_NORTH:
		y = y - k
	if theta == resto.DIR_SOUTH:
		y = y + k
	if theta == resto.DIR_EAST:
		x = x + k
	if theta == resto.DIR_WEST:
		x = x - k

	return (x, y, theta)

# https://hal.archives-ouvertes.fr/hal-03017566/document
def construct_single_path_with_angles_bspline(exp_settings, start, goal, sample_pts, fn, is_weak=False):
	if len(sample_pts) == 0:
		return [start, goal]
		# return construct_single_path_with_angles_spline(exp_settings, start, goal, sample_pts, fn, is_weak=False)

	x, y = [], []
	xy_0 = start
	xy_n = goal
	xy_mid = sample_pts

	xy_pre_n = get_pre_goal_pt(goal, exp_settings)

	x.append(xy_0[0])
	y.append(xy_0[1])

	for i in range(len(sample_pts)):
		spt = sample_pts[i]
		sx = spt[0]
		sy = spt[1]
		x.append(sx)
		y.append(sy)

	x.append(xy_pre_n[0])
	y.append(xy_pre_n[1])

	x.append(xy_n[0])
	y.append(xy_n[1])

	# Subtract 90 to turn path angle into tangent
	start_angle = xy_0[2] - 90
	# Do the reverse for the ending point
	end_angle 	= xy_n[2] + 90

	# Strength of how much we're enforcing the exit angle
	k = exp_settings['angle_strength']

	x = np.array(x)
	y = np.array(y)

	# print(path_formatted(x, y))

	tck,u = interpolate.splprep([x,y],s=0)
	unew = np.arange(0,1.01,0.01)
	out = interpolate.splev(unew,tck)

	path = path_formatted(out[0], out[1])
	return path

# https://hal.archives-ouvertes.fr/hal-03017566/document
def construct_single_path_with_angles_spline(exp_settings, start, goal, sample_pts, fn, is_weak=False):
	# print("WITH ANGLE")
	xy_0 = start
	xy_n = goal
	xy_mid = sample_pts	

	x = []
	y = []

	x.append(xy_0[0])
	y.append(xy_0[1])

	for i in range(len(sample_pts)):
		spt = sample_pts[i]
		sx = spt[0]
		sy = spt[1]
		x.append(sx)
		y.append(sy)

	x.append(xy_n[0])
	y.append(xy_n[1])

	# Subtract 90 to turn path angle into tangent
	start_angle = xy_0[2] - 90
	# Do the reverse for the ending point
	end_angle 	= xy_n[2] + 90

	# Strength of how much we're enforcing the exit angle
	k = exp_settings['angle_strength']
	
	if is_weak:
		t1 = np.array(as_tangent(start_angle)) * k * .001
	else:
		t1 = np.array(as_tangent(start_angle)) * k

	tn = np.array(as_tangent(end_angle)) * k

	# print(type(t1))
	# tangent vectors

	# print("Tangents")
	# print(t1)
	# print(tn)


	Px=np.concatenate(([t1[0]],x,[tn[0]]))
	Py=np.concatenate(([t1[1]],y,[tn[1]]))

	# interpolation equations
	n = len(x)
	phi = np.zeros((n+2,n+2))
	for i in range(n):
		phi[i+1,i]=1
		phi[i+1,i+1]=4
		phi[i+1,i+2]=1

	# end condition constraints
	phi=np.zeros((n+2,n+2))
	for i in range(n):
		phi[i+1,i] = 1
		phi[i+1,i+1] = 4
		phi[i+1,i+2] = 1 
	phi[0,0] = -3
	phi[0,2] = 3
	phi[n+1,n-1] = -3
	phi[n+1,n+1] = 3
	# passage matrix
	phi_inv = np.linalg.inv(phi)
	# control points
	Qx=6*phi_inv.dot(Px)
	Qy=6*phi_inv.dot(Py)
	# figure plot
	# plt.figure(figsize=(12, 5))
	t=np.linspace(0,1,num=101)

	length = 1000
	width = 1375

	plt.xlim([0, width])
	plt.ylim([0, length])

	x_all = []
	y_all = []

	for k in range(0,n-1):
		x_t = 1.0/6.0*(((1-t)**3)*Qx[k]+(3*t**3-6*t**2+4)*Qx[k+1]+(-3*t**3+3*t**2+3*t+1)*Qx[k+2]+(t**3)*Qx[k+3])
		y_t = 1.0/6.0*(((1-t)**3)*Qy[k]+(3*t**3-6*t**2+4)*Qy[k+1]+(-3*t**3+3*t**2+3*t+1)*Qy[k+2]+(t**3)*Qy[k+3]) 
		
		x_all.extend(x_t)
		y_all.extend(y_t)

	if FLAG_EXPORT_SPLINE_DEBUG:
		plt.plot(x_t,y_t,'k',linewidth=2.0,color='orange')

		print("Saving the path I made lalalala")
		plt.plot(x, y, 'ko', label='fit knots',markersize=15.0)
		plt.plot(Qx, Qy, 'o--', label='control points',markersize=15.0)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend(loc='upper left', ncol=2)
		fn_spline = fn_pathpickle_from_exp_settings(exp_settings) + 'sample-cubic_spline_imposed_tangent_direction.png'
		plt.savefig(fn_spline)
		# plt.show()
		plt.clf()

	return path_formatted(x_all, y_all)

def construct_single_path_bezier(start, end, sample_pts):
	points = []
	
	xys = [start] + sample_pts + [end]

	ts = [t/NUMBER_STEPS for t in range(NUMBER_STEPS + 1)]
	bezier = resto.make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points

def create_path_options(num_paths, target, restaurant, vis_type):
	path_list = []
	for i in range(num_paths):
		path_option = generate_single_path(restaurant, target, vis_type)
		path_list.append(path_option)

	return path_list

def generate_paths(num_paths, restaurant, vis_types):
	path_options = {}
	for target in restaurant.get_goals_all():
		for vis_type in vis_types:
			path_options[target][vis_type] = create_path_options(num_paths, target, restaurant, vis_type)
	return path_options


def get_vis_labels():
	vis_labels, dummy = get_visibilities([], [], [], [])
	return vis_labels

def minMax(x):
	return pd.Series(index=['min','max'],data=[x.min(),x.max()])

def determine_lambda(r):
	start = r.get_start()
	goals = r.get_goals_all()
	lambda_val = 0
	costs = []

	for g in goals:
		p = generate_single_path(r, g, None, 0)

		p_cost = f_path_cost(p)
		costs.append(p_cost)


	final_cost = max(costs)


	pass

def inspect_heatmap(df):
	# print(df)

	length 		= df['x'].max()
	width 		= df['y'].max()
	max_multi 	= df['VIS_MULTI'].max()
	max_a 		= df['VIS_A'].max()
	max_b 		= df['VIS_B'].max()
	max_omni 	= df['VIS_OMNI'].max()

	# print((length, width))
	print((max_omni, max_multi))

	img = np.zeros((length,width), np.uint8)

	# df = df.transpose()
	for x in range(width):
		for y in range(length):
			val = df[(df['x'] == x) & (df['y'] == y) ]
			v = val['VIS_MULTI']
			fill = int(255.0 * (v / max_multi) )
			img[x, y] = fill

		print(x)


	cv2.imwrite('multi_heatmap'+ '.png', img) 

# df.at[i,COL_PATHING] = get_pm_label(row)

def inspect_visibility(options, restaurant, ti, fn):
	options = options[0]

	for pkey in options.keys():
		print(pkey)
		path = options[pkey][0]
		# print('saving fig')


		t = range(len(path))
		v = get_vis_graph_info(path, restaurant)
		# vo, va, vb, vm = v

		fig = plt.figure()
		ax1 = fig.add_subplot(111)

		for key in v.keys():
			ax1.scatter(t, v[key], s=10, c='r', marker="o", label=key)

		# ax1.scatter(t, va, s=10, c='b', marker="o", label="Vis A")
		# ax1.scatter(t, vb, s=10, c='y', marker="o", label="Vis B")
		# ax1.scatter(t, vm, s=10, c='g', marker="o", label="Vis Multi")

		ax1.set_title('visibility of ' + pkey)
		plt.legend(loc='upper left');
		
		plt.savefig(fn + "-" + str(ti) + "-" + pkey + '-vis' + '.png')
		plt.clf()

		# f1 = f_convolved(v1, f_og)
		# f2 = f_convolved(v2, f_og)
		# f3 = f_convolved(v3, f_og)
		# f4 = f_convolved(v4, f_og)
		# f5 = f_convolved(v5, f_og)

		# fig = plt.figure()
		# ax1 = fig.add_subplot(111)

		# ax1.scatter(x, f1, s=10, c='b', marker="o", label=vl1)
		# ax1.scatter(x, f2, s=10, c='r', marker="o", label=vl2)
		# ax1.scatter(x, f3, s=10, c='g', marker="o", label=vl3)
		# ax1.scatter(x, f4, s=10, c='y', marker="o", label=vl4)
		# ax1.scatter(x, f5, s=10, c='grey', marker="o", label=vl5)
		# ax1.set_title('f_remix for best path to goal ' + goal)
		# plt.legend(loc='upper left');
			
def get_vis_graph_info(path, restaurant):
	vals_dict = {}

	obs_sets = restaurant.get_obs_sets()

	for aud_i in obs_sets.keys():
		vals = []
		for t in range(len(path)):

			# goal, goals, path, df_obs
			new_val = f_remix3(t, path[t], obs_sets[aud_i], path)
			# print(new_val)
			# exit()

			vals.append(new_val)

		vals_dict[aud_i] = vals

	return vals_dict
	# return vo, va, vb, vm



def inspect_details(detail_dict, fn):
	if FLAG_PROB_HEADING:
		return
	return

	vis_labels = get_vis_labels()
	vl1 = vis_labels[0]
	vl2 = vis_labels[1]
	vl3 = vis_labels[2]
	vl4 = vis_labels[3]
	vl5 = vis_labels[4]

	for pkey in detail_dict.keys():
		# print('saving fig')
		paths_details = detail_dict[pkey]

		for detail in paths_details:
			v1 = detail[vl1]
			v2 = detail[vl2]
			v3 = detail[vl3]
			v4 = detail[vl4]
			v5 = detail[vl5]

			goal_index = detail['target_index']
			goal = resto.UNITY_GOAL_NAMES[goal_index]

			x = range(len(v1))
	
			fig = plt.figure()
			ax1 = fig.add_subplot(111)

			ax1.scatter(x, v1, s=10, c='b', marker="o", label=vl1)
			ax1.scatter(x, v2, s=10, c='r', marker="o", label=vl2)
			ax1.scatter(x, v3, s=10, c='g', marker="o", label=vl3)
			ax1.scatter(x, v4, s=10, c='y', marker="o", label=vl4)
			ax1.scatter(x, v5, s=10, c='grey', marker="o", label=vl5)
			ax1.set_title('visibility of best path to goal ' + goal)
			plt.legend(loc='upper left');
			
			plt.savefig(fn + 'vis' + '.png')
			plt.clf()

			f1 = f_convolved(v1, f_og)
			f2 = f_convolved(v2, f_og)
			f3 = f_convolved(v3, f_og)
			f4 = f_convolved(v4, f_og)
			f5 = f_convolved(v5, f_og)

			fig = plt.figure()
			ax1 = fig.add_subplot(111)

			ax1.scatter(x, f1, s=10, c='b', marker="o", label=vl1)
			ax1.scatter(x, f2, s=10, c='r', marker="o", label=vl2)
			ax1.scatter(x, f3, s=10, c='g', marker="o", label=vl3)
			ax1.scatter(x, f4, s=10, c='y', marker="o", label=vl4)
			ax1.scatter(x, f5, s=10, c='grey', marker="o", label=vl5)
			ax1.set_title('f_remix for best path to goal ' + goal)
			plt.legend(loc='upper left');
			
			plt.savefig(fn + goal + '-' + pkey + '-convolved' + '.png')
			plt.clf()



def combine_list_of_dicts(all_options):
	new_dict = {}
	keys = {}

	for option in all_options:
		keys = option.keys() | keys

	for key in keys:
		new_dict[key] = []

	for option in all_options:
		for key in keys:
			new_dict[key].append(option[key])
	
	return new_dict

def get_hardcoded():
	start = (104, 477)
	end = (1035, 567)
	l1 = construct_single_path_bezier(start, end, [(894, 265)])

	labels = ['max-lo-fcombo', 'max-la-fcombo', 'max-lb-fcombo', 'max-lm-fcombo']
	p1 = [(104, 477), (141, 459), (178, 444), (215, 430), (251, 417), (287, 405), (322, 395), (357, 386), (391, 379), (425, 373), (459, 368), (492, 365), (525, 363), (557, 363), (588, 364), (620, 366), (651, 370), (681, 375), (711, 381), (740, 389), (769, 398), (798, 409), (826, 421), (854, 434), (881, 449), (908, 465), (934, 483), (960, 502), (985, 522), (1010, 543), (1035, 567)]
	p2 = [(104, 477), (147, 447), (190, 419), (231, 394), (272, 371), (312, 350), (351, 331), (390, 315), (427, 301), (464, 289), (499, 280), (534, 273), (568, 268), (601, 265), (634, 265), (665, 267), (696, 271), (726, 277), (755, 286), (783, 297), (810, 310), (836, 325), (862, 343), (886, 363), (910, 385), (933, 410), (955, 437), (976, 466), (996, 497), (1016, 531), (1035, 567)]
	p3 = [(104, 477), (124, 447), (145, 419), (167, 394), (190, 371), (213, 350), (237, 332), (262, 315), (288, 301), (314, 290), (341, 280), (369, 273), (397, 268), (427, 266), (457, 265), (487, 267), (519, 271), (551, 278), (584, 286), (617, 297), (652, 310), (687, 326), (722, 343), (759, 363), (796, 386), (834, 410), (873, 437), (912, 466), (952, 497), (993, 531), (1035, 567)]
	p4 = [(104, 477), (146, 446), (187, 418), (228, 392), (268, 369), (307, 348), (345, 329), (383, 313), (420, 298), (456, 286), (491, 277), (525, 269), (559, 264), (592, 262), (624, 261), (656, 263), (686, 267), (716, 274), (745, 282), (774, 293), (801, 307), (828, 322), (854, 340), (879, 361), (904, 383), (928, 408), (950, 435), (973, 464), (994, 496), (1015, 530), (1035, 567)]
	p5 = [(104, 477), (98, 509), (95, 540), (95, 569), (97, 596), (101, 620), (108, 643), (118, 663), (130, 682), (145, 698), (162, 712), (182, 725), (204, 735), (229, 743), (256, 749), (286, 753), (318, 755), (353, 755), (390, 753), (430, 749), (472, 742), (517, 734), (565, 724), (615, 711), (667, 697), (722, 680), (779, 662), (839, 641), (902, 618), (967, 593), (1035, 567)]

	# options = {}
	# options[labels[0]] = [p5]
	# options[labels[1]] = [p1]
	# options[labels[2]] = [p2]
	# options[labels[3]] = [p3]

	# RSS Workshop paper points 
	options = {}
	options[labels[0]] = [p5] # RED
	options[labels[1]] = [p3] # YELLOW
	options[labels[2]] = [p2] # BLUE
	options[labels[3]] = [l1] # GREEN

	return options


# remove invalid paths
def trim_paths(r, all_paths_dict, goal, exp_settings, reverse=False):
	trimmed_paths = []
	trimmed_sp = []
	removed_paths = []

	all_paths = all_paths_dict['path']
	sp = all_paths_dict['sp']

	print(len(all_paths))
	print(len(sp))

	for pi in range(len(all_paths)):
		p = all_paths[pi]
		is_valid = is_valid_path(r, p, exp_settings)
		if is_valid and reverse == False:
			trimmed_paths.append(p)
			trimmed_sp.append(sp[pi])
		elif not is_valid and reverse == True:
			trimmed_paths.append(p)
			trimmed_sp.append(sp[pi])

		if not is_valid and reverse == False:
			removed_paths.append(p)
		elif is_valid and reverse == True:
			removed_paths.append(p)

	if reverse == False:
		print("Paths trimmed: " + str(len(all_paths)) + " -> " + str(len(trimmed_paths)))
	return trimmed_paths, removed_paths, trimmed_sp

def get_mirrored(r, sample_sets):
	start = r.get_start()
	sx, sy, st = start
	mirror_sets = []
	# print(path)
	for ss in sample_sets:
		new_path = []
		for p in ss:
			new_y_offset = (sy - p[1])
			new_y = sy + new_y_offset
			new_pt = (p[0], new_y)
			new_path.append(new_pt)

		mirror_sets.append(new_path)
	return mirror_sets

def get_mirrored_path(r, path):
	start = r.get_start()
	sx, sy, st = start
	new_path = []
	for p in path:
		new_y_offset = (sy - p[1])
		new_y = sy + new_y_offset
		new_pt = (p[0], new_y)
		new_path.append(new_pt)
	return new_path

def get_sample_points_sets(r, start, goal, exp_settings):
	# sampling_type = 'systematic'
	# sampling_type = 'visible'
	# sampling_type = 'in_zone'

	sample_sets = []
	# resolution = 10
	SAMPLE_BUFFER = 150

	sampling_type = exp_settings['sampling_type']

	if sampling_type == SAMPLE_TYPE_SYSTEMATIC or sampling_type == SAMPLE_TYPE_FUSION:
		width = r.get_width()
		length = r.get_length()

		xi_range = range(int(width / resolution))
		yi_range = range(int(length / resolution))

		for xi in xi_range:
			for yi in yi_range:
				x = int(resolution * xi)
				y = int(resolution * yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

	if sampling_type == SAMPLE_TYPE_DEMO:
		start = (104, 477)
		end = (1035, 567)
		l1 = construct_single_path_bezier(start, end, [(894, 265)])

		p1 = [(104, 477), (141, 459), (178, 444), (215, 430), (251, 417), (287, 405), (322, 395), (357, 386), (391, 379), (425, 373), (459, 368), (492, 365), (525, 363), (557, 363), (588, 364), (620, 366), (651, 370), (681, 375), (711, 381), (740, 389), (769, 398), (798, 409), (826, 421), (854, 434), (881, 449), (908, 465), (934, 483), (960, 502), (985, 522), (1010, 543), (1035, 567)]
		p2 = [(104, 477), (147, 447), (190, 419), (231, 394), (272, 371), (312, 350), (351, 331), (390, 315), (427, 301), (464, 289), (499, 280), (534, 273), (568, 268), (601, 265), (634, 265), (665, 267), (696, 271), (726, 277), (755, 286), (783, 297), (810, 310), (836, 325), (862, 343), (886, 363), (910, 385), (933, 410), (955, 437), (976, 466), (996, 497), (1016, 531), (1035, 567)]
		p3 = [(104, 477), (124, 447), (145, 419), (167, 394), (190, 371), (213, 350), (237, 332), (262, 315), (288, 301), (314, 290), (341, 280), (369, 273), (397, 268), (427, 266), (457, 265), (487, 267), (519, 271), (551, 278), (584, 286), (617, 297), (652, 310), (687, 326), (722, 343), (759, 363), (796, 386), (834, 410), (873, 437), (912, 466), (952, 497), (993, 531), (1035, 567)]
		p4 = [(104, 477), (146, 446), (187, 418), (228, 392), (268, 369), (307, 348), (345, 329), (383, 313), (420, 298), (456, 286), (491, 277), (525, 269), (559, 264), (592, 262), (624, 261), (656, 263), (686, 267), (716, 274), (745, 282), (774, 293), (801, 307), (828, 322), (854, 340), (879, 361), (904, 383), (928, 408), (950, 435), (973, 464), (994, 496), (1015, 530), (1035, 567)]
		p5 = [(104, 477), (98, 509), (95, 540), (95, 569), (97, 596), (101, 620), (108, 643), (118, 663), (130, 682), (145, 698), (162, 712), (182, 725), (204, 735), (229, 743), (256, 749), (286, 753), (318, 755), (353, 755), (390, 753), (430, 749), (472, 742), (517, 734), (565, 724), (615, 711), (667, 697), (722, 680), (779, 662), (839, 641), (902, 618), (967, 593), (1035, 567)]
		p6 = l1

		p1 = chunkify.chunkify_path(exp_settings, p1)
		p2 = chunkify.chunkify_path(exp_settings, p2)
		p3 = chunkify.chunkify_path(exp_settings, p3)
		p4 = chunkify.chunkify_path(exp_settings, p4)
		p5 = chunkify.chunkify_path(exp_settings, p5)
		p6 = chunkify.chunkify_path(exp_settings, p6)
		p7 = get_min_viable_path(r, goal, exp_settings)

		sample_sets = [p1, p2, p3, p4, p5, p6, p7]

	if sampling_type == SAMPLE_TYPE_NEXUS_POINTS:
		sample_sets = []
		imported_0 = {((1005, 257, 180), 'naked'): [(504, 107)], ((1005, 257, 180), 'omni'): [(504, 107)], ((1005, 257, 180), 'a'): [(504, 407)], ((1005, 257, 180), 'b'): [(804, 407)], ((1005, 257, 180), 'c'): [(804, 407)], ((1005, 257, 180), 'd'): [(804, 407)], ((1005, 257, 180), 'e'): [(804, 407)], ((1005, 617, 0), 'naked'): [(504, 407)], ((1005, 617, 0), 'omni'): [(504, 407)], ((1005, 617, 0), 'a'): [(504, 407)], ((1005, 617, 0), 'b'): [(504, 407)], ((1005, 617, 0), 'c'): [(504, 407)], ((1005, 617, 0), 'd'): [(504, 407)], ((1005, 617, 0), 'e'): [(504, 407)]}
		imported_1 = {((1005, 257, 180), 'naked'): [(508, 111)], ((1005, 257, 180), 'omni'): [(508, 111)], ((1005, 257, 180), 'a'): [(503, 411)], ((1005, 257, 180), 'b'): [(800, 411)], ((1005, 257, 180), 'c'): [(807, 411)], ((1005, 257, 180), 'd'): [(807, 411)], ((1005, 257, 180), 'e'): [(807, 402)], ((1005, 617, 0), 'naked'): [(500, 411)], ((1005, 617, 0), 'omni'): [(500, 411)], ((1005, 617, 0), 'a'): [(500, 411)], ((1005, 617, 0), 'b'): [(499, 411)], ((1005, 617, 0), 'c'): [(499, 411)], ((1005, 617, 0), 'd'): [(499, 411)], ((1005, 617, 0), 'e'): [(500, 411)]}
		imported_2 = {((1005, 257, 180), 'naked'): [(624, 107)], ((1005, 257, 180), 'omni'): [(624, 107)], ((1005, 257, 180), 'a'): [(474, 407)], ((1005, 257, 180), 'b'): [(774, 407)], ((1005, 257, 180), 'c'): [(924, 407)], ((1005, 257, 180), 'd'): [(984, 407)], ((1005, 257, 180), 'e'): [(984, 407)], ((1005, 617, 0), 'naked'): [(414, 737)], ((1005, 617, 0), 'omni'): [(414, 737)], ((1005, 617, 0), 'a'): [(414, 737)], ((1005, 617, 0), 'b'): [(414, 737)], ((1005, 617, 0), 'c'): [(804, 647)], ((1005, 617, 0), 'd'): [(804, 647)], ((1005, 617, 0), 'e'): [(804, 587)]}
		imported_3 = {((1005, 257, 180), 'naked'): [(534, 122)], ((1005, 257, 180), 'omni'): [(534, 122)], ((1005, 257, 180), 'a'): [(624, 422)], ((1005, 257, 180), 'b'): [(744, 422)], ((1005, 257, 180), 'c'): [(909, 422)], ((1005, 257, 180), 'd'): [(984, 407)], ((1005, 257, 180), 'e'): [(984, 407)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(399, 752)], ((1005, 617, 0), 'b'): [(429, 752)], ((1005, 617, 0), 'c'): [(819, 647)], ((1005, 617, 0), 'd'): [(819, 647)], ((1005, 617, 0), 'e'): [(819, 572)]}
		imported_4 = {((1005, 257, 180), 'naked'): [(533, 125)], ((1005, 257, 180), 'omni'): [(533, 125)], ((1005, 257, 180), 'a'): [(619, 423)], ((1005, 257, 180), 'b'): [(743, 425)], ((1005, 257, 180), 'c'): [(908, 425)], ((1005, 257, 180), 'd'): [(987, 402)], ((1005, 257, 180), 'e'): [(987, 402)], ((1005, 617, 0), 'naked'): [(426, 723)], ((1005, 617, 0), 'omni'): [(426, 723)], ((1005, 617, 0), 'a'): [(396, 755)], ((1005, 617, 0), 'b'): [(426, 755)], ((1005, 617, 0), 'c'): [(822, 650)], ((1005, 617, 0), 'd'): [(822, 650)], ((1005, 617, 0), 'e'): [(822, 573)]}
		imported_5 = {((1005, 257, 180), 'naked'): [(540, 124)], ((1005, 257, 180), 'omni'): [(540, 124)], ((1005, 257, 180), 'a'): [(617, 429)], ((1005, 257, 180), 'b'): [(737, 433)], ((1005, 257, 180), 'c'): [(992, 413)], ((1005, 257, 180), 'd'): [(992, 413)], ((1005, 257, 180), 'e'): [(990, 409)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(390, 757)], ((1005, 617, 0), 'b'): [(426, 763)], ((1005, 617, 0), 'c'): [(830, 648)], ((1005, 617, 0), 'd'): [(830, 648)], ((1005, 617, 0), 'e'): [(824, 571)]}

		imported_res_2 = {((1005, 257, 180), 'naked'): [(1152, 431)], ((1005, 257, 180), 'omni'): [(1152, 431)], ((1005, 257, 180), 'a'): [(486, 431)], ((1005, 257, 180), 'b'): [(738, 431)], ((1005, 257, 180), 'c'): [(1044, 431)], ((1005, 257, 180), 'd'): [(1152, 431)], ((1005, 257, 180), 'e'): [(1152, 431)], ((1005, 617, 0), 'naked'): [(426, 725)], ((1005, 617, 0), 'omni'): [(426, 725)], ((1005, 617, 0), 'a'): [(372, 761)], ((1005, 617, 0), 'b'): [(426, 761)], ((1005, 617, 0), 'c'): [(426, 731)], ((1005, 617, 0), 'd'): [(426, 725)], ((1005, 617, 0), 'e'): [(1116, 449)]}
		imported_res_3 = {((1005, 257, 180), 'naked'): [(545, 123)], ((1005, 257, 180), 'omni'): [(545, 123)], ((1005, 257, 180), 'a'): [(616, 434)], ((1005, 257, 180), 'b'): [(734, 434)], ((1005, 257, 180), 'c'): [(983, 426)], ((1005, 257, 180), 'd'): [(992, 413)], ((1005, 257, 180), 'e'): [(990, 409)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(375, 768)], ((1005, 617, 0), 'b'): [(429, 776)], ((1005, 617, 0), 'c'): [(832, 648)], ((1005, 617, 0), 'd'): [(832, 648)], ((1005, 617, 0), 'e'): [(824, 571)]}

		imported_res_4 = {((1005, 257, 180), 'naked'): [(564, 107)], ((1005, 257, 180), 'omni'): [(564, 107)], ((1005, 257, 180), 'a'): [(384, 407)], ((1005, 257, 180), 'b'): [(744, 407)], ((1005, 257, 180), 'c'): [(924, 407)], ((1005, 257, 180), 'd'): [(1014, 407)], ((1005, 257, 180), 'e'): [(1014, 407)], ((1005, 617, 0), 'naked'): [(834, 557)], ((1005, 617, 0), 'omni'): [(834, 557)], ((1005, 617, 0), 'a'): [(834, 497)], ((1005, 617, 0), 'b'): [(834, 587)], ((1005, 617, 0), 'c'): [(1014, 467)], ((1005, 617, 0), 'd'): [(1014, 467)], ((1005, 617, 0), 'e'): [(1014, 467)]}
		imported_res_5 = {((1005, 257, 180), 'naked'): [(1114, 422)], ((1005, 257, 180), 'omni'): [(1114, 422)], ((1005, 257, 180), 'a'): [(384, 432)], ((1005, 257, 180), 'b'): [(734, 432)], ((1005, 257, 180), 'c'): [(969, 432)], ((1005, 257, 180), 'd'): [(1154, 432)], ((1005, 257, 180), 'e'): [(1154, 432)], ((1005, 617, 0), 'naked'): [(429, 722)], ((1005, 617, 0), 'omni'): [(429, 722)], ((1005, 617, 0), 'a'): [(374, 642)], ((1005, 617, 0), 'b'): [(649, 762)], ((1005, 617, 0), 'c'): [(1009, 622)], ((1005, 617, 0), 'd'): [(1154, 442)], ((1005, 617, 0), 'e'): [(1154, 442)]}
		imported_res_6 = {((1005, 257, 180), 'naked'): [(1149, 431)], ((1005, 257, 180), 'omni'): [(1149, 431)], ((1005, 257, 180), 'a'): [(387, 433)], ((1005, 257, 180), 'b'): [(731, 434)], ((1005, 257, 180), 'c'): [(958, 433)], ((1005, 257, 180), 'd'): [(1151, 434)], ((1005, 257, 180), 'e'): [(1167, 429)], ((1005, 617, 0), 'naked'): [(426, 789)], ((1005, 617, 0), 'omni'): [(426, 789)], ((1005, 617, 0), 'a'): [(371, 643)], ((1005, 617, 0), 'b'): [(640, 775)], ((1005, 617, 0), 'c'): [(1004, 623)], ((1005, 617, 0), 'd'): [(1163, 444)], ((1005, 617, 0), 'e'): [(1163, 443)]}

		nexus_icon_3 = {((1005, 257, 180), 'naked'): [(644, 397)], ((1005, 257, 180), 'omni'): [(644, 397)], ((1005, 257, 180), 'a'): [(1134, 427)], ((1005, 257, 180), 'b'): [(1114, 427)], ((1005, 257, 180), 'c'): [(914, 427)], ((1005, 257, 180), 'd'): [(734, 427)], ((1005, 257, 180), 'e'): [(384, 427)], ((1005, 617, 0), 'naked'): [(664, 547)], ((1005, 617, 0), 'omni'): [(664, 547)], ((1005, 617, 0), 'a'): [(1124, 447)], ((1005, 617, 0), 'b'): [(1124, 447)], ((1005, 617, 0), 'c'): [(824, 647)], ((1005, 617, 0), 'd'): [(664, 757)], ((1005, 617, 0), 'e'): [(374, 637)]}
		nexus_icon = {((1005, 257, 180), 'naked'): [(1150, 431)], ((1005, 257, 180), 'omni'): [(1150, 431)], ((1005, 257, 180), 'a'): [(1180, 430)], ((1005, 257, 180), 'b'): [(1152, 434)], ((1005, 257, 180), 'c'): [(971, 434)], ((1005, 257, 180), 'd'): [(734, 433)], ((1005, 257, 180), 'e'): [(386, 434)], ((1005, 617, 0), 'naked'): [(429, 790)], ((1005, 617, 0), 'omni'): [(429, 790)], ((1005, 617, 0), 'a'): [(1178, 442)], ((1005, 617, 0), 'b'): [(1172, 445)], ((1005, 617, 0), 'c'): [(971, 444)], ((1005, 617, 0), 'd'): [(643, 788)], ((1005, 617, 0), 'e'): [(372, 644)]}
		nexus_icon_2 = {((1005, 257, 180), 'naked'): [(594, 197)], ((1005, 257, 180), 'omni'): [(594, 197)], ((1005, 257, 180), 'a'): [(1134, 427)], ((1005, 257, 180), 'b'): [(1114, 427)], ((1005, 257, 180), 'c'): [(914, 427)], ((1005, 257, 180), 'd'): [(734, 427)], ((1005, 257, 180), 'e'): [(384, 427)], ((1005, 617, 0), 'naked'): [(414, 727)], ((1005, 617, 0), 'omni'): [(414, 727)], ((1005, 617, 0), 'a'): [(1124, 447)], ((1005, 617, 0), 'b'): [(1124, 447)], ((1005, 617, 0), 'c'): [(824, 647)], ((1005, 617, 0), 'd'): [(664, 757)], ((1005, 617, 0), 'e'): [(374, 637)]}

		central_20_points = {((1005, 257, 180), 'naked'): [(384, 107)], ((1005, 257, 180), 'omni'): [(384, 107)], ((1005, 257, 180), 'a'): [(1124, 427)], ((1005, 257, 180), 'b'): [(1124, 427)], ((1005, 257, 180), 'c'): [(904, 427)], ((1005, 257, 180), 'd'): [(764, 427)], ((1005, 257, 180), 'e'): [(384, 427)], ((1005, 617, 0), 'naked'): [(664, 707)], ((1005, 617, 0), 'omni'): [(664, 707)], ((1005, 617, 0), 'a'): [(1124, 447)], ((1005, 617, 0), 'b'): [(1124, 447)], ((1005, 617, 0), 'c'): [(824, 647)], ((1005, 617, 0), 'd'): [(664, 747)], ((1005, 617, 0), 'e'): [(664, 507)]}
		central_15_points = {((1005, 257, 180), 'naked'): [(384, 107)], ((1005, 257, 180), 'omni'): [(384, 107)], ((1005, 257, 180), 'a'): [(1119, 422)], ((1005, 257, 180), 'b'): [(1119, 422)], ((1005, 257, 180), 'c'): [(909, 422)], ((1005, 257, 180), 'd'): [(744, 422)], ((1005, 257, 180), 'e'): [(384, 422)], ((1005, 617, 0), 'naked'): [(399, 722)], ((1005, 617, 0), 'omni'): [(399, 722)], ((1005, 617, 0), 'a'): [(1059, 452)], ((1005, 617, 0), 'b'): [(1059, 452)], ((1005, 617, 0), 'c'): [(819, 647)], ((1005, 617, 0), 'd'): [(429, 752)], ((1005, 617, 0), 'e'): [(399, 587)]}

		central_15_points = {((1005, 257, 180), 'naked'): [(384, 107)], ((1005, 257, 180), 'omni'): [(384, 107)], ((1005, 257, 180), 'a'): [(1104, 422)], ((1005, 257, 180), 'b'): [(1104, 422)], ((1005, 257, 180), 'c'): [(924, 422)], ((1005, 257, 180), 'd'): [(744, 422)], ((1005, 257, 180), 'e'): [(384, 422)], ((1005, 617, 0), 'naked'): [(429, 737)], ((1005, 617, 0), 'omni'): [(429, 737)], ((1005, 617, 0), 'a'): [(1059, 452)], ((1005, 617, 0), 'b'): [(1059, 452)], ((1005, 617, 0), 'c'): [(834, 632)], ((1005, 617, 0), 'd'): [(429, 767)], ((1005, 617, 0), 'e'): [(429, 587)]}

		# central points with new method
		central_15_points = {((1005, 257, 180), 'naked'): [(389, 107)], ((1005, 257, 180), 'omni'): [(389, 107)], ((1005, 257, 180), 'a'): [(954, 357)], ((1005, 257, 180), 'b'): [(949, 392)], ((1005, 257, 180), 'c'): [(904, 432)], ((1005, 257, 180), 'd'): [(764, 427)], ((1005, 257, 180), 'e'): [(399, 427)], ((1005, 617, 0), 'naked'): [(374, 732)], ((1005, 617, 0), 'omni'): [(374, 732)], ((1005, 617, 0), 'a'): [(949, 437)], ((1005, 617, 0), 'b'): [(949, 442)], ((1005, 617, 0), 'c'): [(844, 627)], ((1005, 617, 0), 'd'): [(649, 762)], ((1005, 617, 0), 'e'): [(374, 607)]}
		central_15_points = {((1005, 257, 180), 'naked'): [(399, 107)], ((1005, 257, 180), 'omni'): [(399, 107)], ((1005, 257, 180), 'a'): [(924, 362)], ((1005, 257, 180), 'b'): [(924, 407)], ((1005, 257, 180), 'c'): [(894, 422)], ((1005, 257, 180), 'd'): [(759, 422)], ((1005, 257, 180), 'e'): [(399, 422)], ((1005, 617, 0), 'naked'): [(384, 737)], ((1005, 617, 0), 'omni'): [(384, 737)], ((1005, 617, 0), 'a'): [(924, 437)], ((1005, 617, 0), 'b'): [(924, 437)], ((1005, 617, 0), 'c'): [(879, 437)], ((1005, 617, 0), 'd'): [(384, 752)], ((1005, 617, 0), 'e'): [(384, 602)]}

		# best with cutoff 20
		central_15_points = {((1005, 257, 180), 'naked'): [(399, 107)], ((1005, 257, 180), 'omni'): [(399, 107)], ((1005, 257, 180), 'a'): [(894, 332)], ((1005, 257, 180), 'b'): [(894, 422)], ((1005, 257, 180), 'c'): [(894, 422)], ((1005, 257, 180), 'd'): [(759, 422)], ((1005, 257, 180), 'e'): [(399, 422)], ((1005, 617, 0), 'naked'): [(384, 722)], ((1005, 617, 0), 'omni'): [(384, 722)], ((1005, 617, 0), 'a'): [(894, 422)], ((1005, 617, 0), 'b'): [(804, 467)], ((1005, 617, 0), 'c'): [(789, 662)], ((1005, 617, 0), 'd'): [(384, 752)], ((1005, 617, 0), 'e'): [(384, 647)]}

		imported_0 = list(imported_0.values())
		imported_1 = list(imported_1.values())
		imported_2 = list(imported_2.values())
		imported_3 = list(imported_3.values())
		imported_4 = list(imported_4.values())
		imported_5 = list(imported_5.values())
		# imported_5 = list(imported_6.values())

		imported_res_2 = list(imported_res_2.values())
		imported_res_3 = list(imported_res_3.values())
		imported_res_4 = list(imported_res_4.values())
		imported_res_5 = list(imported_res_5.values())
		imported_res_6 = list(imported_res_6.values())


		# central_20_points = list(central_20_points.values())
		central_15_points = list(central_15_points.values())


		# nexus_icon = list(nexus_icon.values())[7:9]
		nexus_icon = [[(429, 790)], [(1150, 431)]]

		imported = []
		# imported.extend(imported_1)
		# imported.extend(imported_2)
		# imported.extend(imported_3)
		# imported.extend(imported_4)
		# imported.extend(imported_5)
		# imported.extend(imported_res_2)
		# imported.extend(imported_res_3)
		# imported.extend(imported_res_4)
		# imported.extend(imported_res_5)
		# imported.extend(imported_res_6)
		# imported.extend(nexus_icon)
		# imported.extend(central_20_points)
		imported.extend(central_15_points)



		new_imported = []
		for imp in imported:
			if imp not in new_imported:
				new_imported.append(imp)
		imported = new_imported

		# mirror_sets = get_mirrored(r, imported)
		# for p in mirror_sets:
		# 	if p not in imported:
		# 		imported.append(p)


		resolution = 1

		augmented = []
		search_hi = 15
		search_lo = -1 * search_hi

		for xi in range(search_lo, search_hi, resolution):
			for yi in range(search_lo, search_hi, resolution):
				for imp in imported:
					new_set = []
					for p in imp:
						# print(p)
						# print(xi, yi)
						new_pt = (p[0] + xi, p[1] + yi)
						new_set.append(new_pt)
					augmented.append(new_set)

		# print(augmented)
		sample_sets.extend(augmented)

	if sampling_type == SAMPLE_TYPE_CURVE_TEST:
		# test_dict = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 425), (204, 420), (204, 413), (204, 404), (205, 394), (206, 382), (208, 369), (209, 354), (213, 338), (216, 320), (221, 301), (228, 281), (237, 259), (248, 238), (262, 217), (280, 197), (303, 180), (329, 166), (361, 159), (397, 157), (436, 162), (479, 170), (523, 184), (566, 199), (608, 216), (647, 231), (685, 247), (721, 262), (756, 276), (788, 289), (818, 300), (846, 309), (872, 317), (896, 321), (917, 325), (936, 326), (953, 325), (968, 321), (979, 316), (988, 308), (995, 300), (999, 291), (1001, 283), (1003, 275), (1004, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 425), (204, 420), (204, 413), (204, 404), (205, 394), (206, 382), (208, 369), (209, 354), (213, 338), (216, 320), (221, 301), (228, 281), (237, 259), (248, 238), (262, 217), (280, 197), (303, 180), (329, 166), (361, 159), (397, 157), (436, 162), (479, 170), (523, 184), (566, 199), (608, 216), (647, 231), (685, 247), (721, 262), (756, 276), (788, 289), (818, 300), (846, 309), (872, 317), (896, 321), (917, 325), (936, 326), (953, 325), (968, 321), (979, 316), (988, 308), (995, 300), (999, 291), (1001, 283), (1003, 275), (1004, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (215, 436), (220, 435), (227, 435), (235, 435), (245, 434), (256, 433), (269, 433), (285, 432), (301, 431), (320, 430), (341, 430), (363, 429), (387, 428), (414, 427), (443, 426), (473, 426), (505, 426), (539, 425), (574, 426), (611, 426), (650, 427), (688, 429), (724, 430), (758, 430), (790, 430), (820, 429), (848, 426), (874, 422), (897, 416), (918, 409), (936, 400), (951, 390), (965, 378), (975, 365), (983, 352), (990, 339), (995, 327), (998, 315), (1000, 303), (1003, 293), (1003, 284), (1004, 276), (1004, 269), (1004, 264), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (216, 436), (222, 436), (230, 435), (239, 435), (250, 435), (263, 434), (278, 433), (295, 433), (314, 432), (335, 432), (358, 431), (384, 430), (411, 429), (440, 428), (472, 428), (505, 427), (540, 426), (577, 426), (615, 426), (655, 426), (695, 426), (734, 427), (771, 428), (805, 429), (836, 430), (863, 430), (889, 429), (911, 427), (931, 423), (949, 416), (963, 408), (974, 396), (983, 384), (989, 370), (994, 357), (997, 343), (1000, 330), (1001, 317), (1003, 306), (1003, 295), (1004, 286), (1004, 277), (1004, 270), (1004, 265), (1004, 260), (1005, 258), (1005, 257), (1005, 257)], ((1005, 257, 180), 'c'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (212, 436), (217, 436), (224, 436), (233, 436), (244, 435), (257, 435), (272, 435), (289, 434), (309, 434), (330, 433), (355, 432), (381, 432), (411, 431), (443, 431), (476, 430), (513, 429), (552, 428), (592, 428), (635, 427), (679, 426), (725, 426), (772, 426), (818, 425), (861, 426), (899, 426), (934, 427), (964, 428), (989, 429), (1009, 430), (1025, 429), (1034, 423), (1033, 413), (1029, 400), (1024, 387), (1020, 374), (1016, 360), (1013, 346), (1010, 333), (1008, 320), (1007, 308), (1006, 297), (1006, 288), (1005, 279), (1005, 272), (1005, 266), (1005, 261), (1005, 259), (1005, 257), (1005, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (218, 435), (225, 435), (235, 434), (246, 433), (259, 432), (275, 431), (293, 430), (313, 429), (336, 427), (361, 425), (389, 423), (420, 421), (453, 419), (488, 417), (527, 414), (567, 411), (610, 409), (655, 407), (701, 404), (750, 402), (799, 400), (848, 398), (894, 397), (936, 396), (973, 396), (1006, 397), (1034, 399), (1056, 402), (1072, 405), (1075, 406), (1065, 401), (1053, 392), (1043, 382), (1034, 371), (1027, 359), (1021, 346), (1016, 334), (1012, 321), (1009, 310), (1008, 299), (1006, 289), (1005, 280), (1005, 273), (1005, 267), (1005, 262), (1005, 259), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (218, 435), (225, 435), (235, 434), (246, 433), (259, 432), (275, 431), (293, 430), (313, 429), (336, 427), (361, 425), (389, 423), (420, 421), (453, 419), (488, 417), (527, 414), (567, 411), (610, 409), (655, 407), (701, 404), (750, 402), (799, 400), (848, 398), (894, 397), (936, 396), (973, 396), (1006, 397), (1034, 399), (1056, 402), (1072, 405), (1075, 406), (1065, 401), (1053, 392), (1043, 382), (1034, 371), (1027, 359), (1021, 346), (1016, 334), (1012, 321), (1009, 310), (1008, 299), (1006, 289), (1005, 280), (1005, 273), (1005, 267), (1005, 262), (1005, 259), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(204, 437), (203, 437), (203, 437), (202, 438), (202, 440), (200, 443), (198, 447), (196, 452), (193, 459), (189, 467), (185, 476), (180, 487), (175, 499), (170, 514), (165, 529), (160, 547), (155, 566), (152, 586), (150, 608), (150, 630), (155, 652), (166, 673), (183, 691), (208, 701), (240, 706), (277, 705), (319, 698), (365, 688), (414, 674), (463, 660), (511, 645), (558, 630), (603, 615), (646, 602), (686, 589), (725, 578), (761, 569), (795, 561), (827, 554), (856, 549), (882, 547), (906, 545), (927, 546), (946, 549), (962, 553), (975, 559), (985, 566), (992, 575), (998, 583), (1000, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(204, 437), (203, 437), (203, 437), (202, 438), (202, 440), (200, 443), (198, 447), (196, 452), (193, 459), (189, 467), (185, 476), (180, 487), (175, 499), (170, 514), (165, 529), (160, 547), (155, 566), (152, 586), (150, 608), (150, 630), (155, 652), (166, 673), (183, 691), (208, 701), (240, 706), (277, 705), (319, 698), (365, 688), (414, 674), (463, 660), (511, 645), (558, 630), (603, 615), (646, 602), (686, 589), (725, 578), (761, 569), (795, 561), (827, 554), (856, 549), (882, 547), (906, 545), (927, 546), (946, 549), (962, 553), (975, 559), (985, 566), (992, 575), (998, 583), (1000, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'a'): [(204, 437), (203, 437), (203, 437), (202, 438), (201, 440), (199, 443), (197, 446), (193, 451), (189, 458), (184, 465), (178, 474), (173, 485), (166, 497), (158, 510), (150, 525), (141, 542), (133, 560), (126, 580), (120, 601), (115, 623), (115, 647), (120, 669), (133, 688), (157, 700), (187, 706), (224, 705), (267, 699), (313, 689), (364, 676), (415, 661), (466, 647), (515, 632), (562, 618), (608, 605), (651, 592), (693, 581), (732, 571), (769, 562), (802, 556), (834, 551), (862, 548), (889, 546), (912, 546), (933, 548), (951, 551), (965, 556), (978, 562), (987, 570), (994, 578), (999, 587), (1002, 595), (1003, 602), (1004, 607), (1004, 612), (1004, 615), (1004, 617), (1004, 617)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (205, 438), (207, 440), (210, 443), (214, 447), (218, 451), (224, 457), (231, 464), (239, 472), (249, 481), (260, 491), (274, 504), (288, 517), (305, 530), (323, 546), (344, 562), (366, 579), (390, 596), (416, 615), (444, 633), (475, 649), (507, 666), (542, 682), (579, 694), (618, 704), (658, 707), (696, 703), (732, 694), (764, 681), (794, 666), (821, 650), (846, 633), (869, 616), (889, 600), (908, 586), (925, 572), (940, 561), (955, 553), (967, 548), (978, 545), (988, 549), (994, 555), (999, 563), (1001, 573), (1002, 582), (1003, 590), (1004, 598), (1004, 604), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 442), (215, 445), (220, 449), (227, 454), (235, 460), (245, 468), (256, 476), (270, 486), (285, 496), (302, 508), (321, 520), (342, 535), (365, 550), (390, 565), (418, 582), (447, 599), (479, 617), (513, 634), (548, 651), (585, 667), (625, 682), (666, 695), (707, 704), (746, 707), (783, 704), (815, 695), (844, 682), (870, 667), (892, 650), (911, 633), (928, 615), (942, 599), (955, 584), (966, 570), (975, 558), (983, 550), (991, 546), (997, 548), (1000, 556), (1002, 564), (1003, 574), (1004, 583), (1004, 591), (1004, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 440), (216, 443), (222, 445), (229, 450), (239, 454), (250, 459), (264, 465), (279, 472), (296, 480), (315, 489), (337, 499), (361, 509), (387, 520), (415, 532), (446, 544), (479, 557), (514, 571), (551, 584), (590, 597), (631, 610), (673, 622), (717, 633), (759, 641), (799, 646), (835, 647), (868, 643), (897, 636), (922, 626), (943, 614), (959, 600), (973, 586), (984, 572), (992, 559), (998, 548), (1002, 538), (1006, 535), (1006, 543), (1006, 553), (1005, 563), (1005, 572), (1005, 582), (1005, 591), (1005, 598), (1005, 605), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 440), (216, 443), (222, 445), (229, 450), (239, 454), (250, 459), (264, 465), (279, 472), (296, 480), (315, 489), (337, 499), (361, 509), (387, 520), (415, 532), (446, 544), (479, 557), (514, 571), (551, 584), (590, 597), (631, 610), (673, 622), (717, 633), (759, 641), (799, 646), (835, 647), (868, 643), (897, 636), (922, 626), (943, 614), (959, 600), (973, 586), (984, 572), (992, 559), (998, 548), (1002, 538), (1006, 535), (1006, 543), (1006, 553), (1005, 563), (1005, 572), (1005, 582), (1005, 591), (1005, 598), (1005, 605), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)]}
		test_dict = {((1005, 257, 180), 'naked'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (237, 436), (250, 436), (265, 436), (282, 435), (301, 435), (324, 435), (348, 435), (377, 434), (407, 434), (441, 433), (477, 433), (516, 433), (558, 432), (603, 432), (650, 431), (700, 431), (752, 430), (806, 430), (861, 430), (918, 429), (974, 429), (1029, 429), (1079, 430), (1124, 430), (1162, 431), (1191, 432), (1210, 434), (1198, 432), (1178, 428), (1157, 422), (1136, 415), (1115, 406), (1096, 397), (1078, 386), (1062, 375), (1049, 363), (1038, 351), (1028, 338), (1021, 326), (1016, 314), (1012, 303), (1008, 293), (1007, 283), (1005, 276), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'omni'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (237, 436), (250, 436), (265, 436), (282, 435), (301, 435), (324, 435), (348, 435), (377, 434), (407, 434), (441, 433), (477, 433), (516, 433), (558, 432), (603, 432), (650, 431), (700, 431), (752, 430), (806, 430), (861, 430), (918, 429), (974, 429), (1029, 429), (1079, 430), (1124, 430), (1162, 431), (1191, 432), (1210, 434), (1198, 432), (1178, 428), (1157, 422), (1136, 415), (1115, 406), (1096, 397), (1078, 386), (1062, 375), (1049, 363), (1038, 351), (1028, 338), (1021, 326), (1016, 314), (1012, 303), (1008, 293), (1007, 283), (1005, 276), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'a'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (215, 436), (221, 436), (228, 436), (236, 435), (245, 435), (257, 435), (270, 434), (285, 434), (302, 434), (320, 433), (341, 433), (363, 433), (387, 432), (413, 432), (442, 432), (472, 432), (504, 432), (537, 432), (573, 433), (610, 433), (648, 434), (687, 435), (723, 436), (757, 436), (789, 435), (819, 433), (847, 430), (873, 425), (896, 418), (917, 410), (936, 401), (951, 390), (964, 378), (975, 365), (983, 352), (990, 339), (994, 326), (998, 314), (1001, 302), (1002, 292), (1003, 283), (1004, 275), (1004, 269), (1004, 263), (1004, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'b'): [(204, 437), (204, 436), (204, 436), (206, 436), (208, 436), (211, 436), (216, 436), (222, 436), (230, 436), (239, 436), (250, 435), (263, 435), (278, 435), (295, 435), (314, 434), (335, 434), (358, 434), (383, 433), (411, 433), (440, 433), (472, 432), (505, 432), (540, 432), (577, 432), (615, 432), (655, 432), (695, 433), (734, 434), (771, 435), (804, 436), (835, 436), (864, 436), (888, 434), (911, 432), (931, 426), (948, 419), (962, 410), (974, 399), (982, 386), (989, 372), (993, 358), (997, 344), (999, 331), (1001, 318), (1003, 306), (1003, 296), (1004, 286), (1004, 278), (1004, 271), (1004, 265), (1004, 261), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'c'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (212, 436), (218, 436), (225, 436), (235, 436), (246, 436), (259, 436), (274, 436), (292, 435), (312, 435), (335, 435), (360, 435), (388, 434), (418, 434), (451, 434), (486, 433), (524, 433), (564, 433), (607, 432), (651, 432), (697, 432), (744, 432), (794, 432), (842, 432), (888, 432), (929, 433), (966, 434), (999, 434), (1025, 436), (1047, 436), (1062, 435), (1064, 428), (1057, 418), (1048, 407), (1040, 394), (1032, 381), (1026, 367), (1020, 353), (1016, 340), (1013, 327), (1010, 314), (1007, 303), (1006, 292), (1006, 283), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'd'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (238, 436), (250, 436), (265, 436), (282, 436), (301, 436), (324, 435), (349, 435), (377, 435), (407, 435), (441, 434), (478, 434), (517, 434), (559, 433), (604, 433), (651, 433), (701, 433), (752, 432), (806, 432), (862, 432), (919, 432), (976, 432), (1031, 432), (1081, 432), (1126, 433), (1163, 434), (1193, 435), (1213, 436), (1199, 434), (1178, 430), (1157, 423), (1135, 416), (1115, 407), (1095, 397), (1078, 387), (1063, 375), (1049, 363), (1038, 350), (1029, 338), (1021, 326), (1015, 314), (1011, 302), (1008, 292), (1007, 283), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 257, 180), 'e'): [(204, 437), (204, 436), (204, 436), (206, 436), (209, 436), (213, 436), (220, 436), (228, 436), (238, 436), (250, 436), (265, 435), (282, 435), (301, 435), (324, 434), (349, 434), (377, 433), (407, 433), (441, 432), (478, 432), (516, 431), (559, 431), (604, 430), (651, 429), (700, 429), (752, 428), (806, 428), (862, 427), (919, 427), (976, 427), (1031, 426), (1081, 427), (1125, 427), (1163, 428), (1193, 429), (1213, 431), (1199, 429), (1179, 426), (1157, 420), (1135, 413), (1115, 405), (1096, 396), (1078, 385), (1062, 374), (1049, 362), (1037, 350), (1029, 337), (1021, 325), (1015, 314), (1011, 302), (1008, 292), (1007, 283), (1005, 275), (1005, 269), (1005, 263), (1005, 260), (1004, 258), (1004, 257), (1004, 257)], ((1005, 617, 0), 'naked'): [(203, 437), (204, 437), (204, 437), (203, 438), (203, 440), (203, 443), (203, 448), (204, 453), (204, 460), (204, 469), (205, 479), (206, 490), (207, 503), (210, 518), (212, 534), (216, 552), (222, 571), (229, 591), (238, 612), (249, 633), (263, 653), (282, 672), (305, 688), (332, 701), (364, 708), (400, 709), (440, 704), (483, 694), (527, 681), (570, 666), (611, 650), (651, 635), (688, 619), (725, 605), (759, 592), (791, 580), (821, 569), (849, 560), (875, 554), (898, 549), (920, 546), (939, 546), (956, 548), (970, 552), (981, 558), (989, 566), (995, 574), (999, 583), (1002, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'omni'): [(203, 437), (204, 437), (204, 437), (204, 439), (204, 441), (204, 444), (204, 448), (205, 454), (206, 461), (207, 469), (208, 480), (210, 491), (212, 505), (216, 520), (220, 536), (226, 554), (232, 573), (240, 594), (251, 614), (262, 636), (278, 657), (296, 677), (319, 695), (346, 709), (377, 718), (414, 722), (452, 718), (494, 709), (537, 696), (579, 680), (620, 664), (658, 647), (695, 631), (731, 616), (764, 601), (796, 588), (825, 576), (852, 567), (877, 559), (900, 553), (921, 549), (940, 548), (956, 549), (970, 552), (981, 559), (990, 566), (995, 574), (999, 583), (1002, 591), (1003, 599), (1004, 605), (1004, 610), (1004, 614), (1005, 616), (1005, 617), (1005, 617)], ((1005, 617, 0), 'a'): [(204, 437), (203, 437), (203, 437), (203, 439), (203, 441), (203, 444), (203, 449), (202, 455), (202, 462), (202, 471), (202, 482), (202, 494), (202, 509), (203, 525), (204, 542), (206, 561), (209, 582), (214, 604), (219, 628), (228, 652), (238, 677), (252, 701), (270, 722), (293, 741), (323, 754), (357, 760), (395, 758), (437, 750), (482, 736), (526, 719), (569, 702), (610, 683), (650, 664), (688, 646), (724, 629), (758, 614), (791, 600), (820, 587), (848, 576), (874, 567), (897, 560), (918, 556), (938, 554), (954, 554), (968, 557), (980, 562), (989, 568), (995, 576), (999, 584), (1001, 592), (1003, 600), (1004, 606), (1004, 611), (1004, 614), (1004, 616), (1004, 617)], ((1005, 617, 0), 'b'): [(203, 437), (204, 437), (204, 437), (204, 439), (204, 441), (204, 444), (205, 449), (205, 455), (206, 462), (208, 471), (209, 482), (211, 494), (214, 508), (218, 524), (222, 542), (227, 561), (234, 581), (241, 603), (251, 626), (262, 650), (276, 674), (293, 697), (314, 719), (338, 738), (367, 753), (401, 762), (439, 763), (480, 756), (523, 743), (565, 727), (606, 709), (644, 690), (682, 671), (717, 652), (750, 635), (782, 618), (811, 603), (839, 589), (865, 578), (888, 569), (910, 562), (929, 557), (947, 554), (962, 555), (975, 557), (985, 563), (993, 570), (998, 578), (1001, 586), (1003, 594), (1004, 601), (1004, 607), (1004, 612), (1004, 615), (1005, 617), (1005, 617)], ((1005, 617, 0), 'c'): [(203, 437), (204, 437), (204, 437), (206, 438), (208, 439), (211, 440), (216, 443), (222, 445), (229, 450), (239, 454), (250, 459), (263, 465), (279, 472), (296, 480), (315, 489), (337, 499), (361, 509), (387, 520), (415, 533), (446, 545), (479, 558), (513, 571), (550, 585), (589, 598), (630, 611), (672, 622), (716, 633), (758, 641), (798, 646), (834, 646), (867, 642), (896, 635), (920, 625), (941, 613), (958, 599), (971, 585), (982, 571), (990, 558), (997, 547), (1001, 538), (1005, 535), (1006, 542), (1005, 552), (1005, 562), (1005, 572), (1005, 582), (1005, 590), (1005, 598), (1005, 604), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'd'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (213, 439), (219, 441), (226, 443), (235, 445), (247, 449), (261, 453), (276, 457), (295, 462), (316, 467), (339, 474), (365, 481), (394, 489), (426, 497), (460, 506), (497, 515), (536, 526), (578, 536), (623, 546), (669, 557), (718, 569), (769, 580), (821, 590), (874, 600), (927, 609), (977, 617), (1023, 621), (1063, 623), (1096, 620), (1122, 613), (1139, 599), (1142, 582), (1135, 566), (1123, 552), (1109, 541), (1093, 533), (1077, 529), (1062, 528), (1048, 530), (1036, 535), (1027, 543), (1020, 552), (1014, 562), (1010, 572), (1008, 581), (1006, 590), (1005, 598), (1005, 604), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)], ((1005, 617, 0), 'e'): [(203, 437), (204, 437), (204, 437), (206, 437), (208, 438), (213, 439), (219, 441), (226, 443), (235, 445), (247, 449), (261, 453), (276, 457), (295, 462), (316, 467), (339, 474), (365, 481), (394, 489), (426, 497), (460, 506), (497, 515), (536, 526), (578, 536), (623, 546), (669, 557), (718, 569), (769, 580), (821, 590), (874, 600), (927, 609), (977, 617), (1023, 621), (1063, 623), (1096, 620), (1122, 613), (1139, 599), (1142, 582), (1135, 566), (1123, 552), (1109, 541), (1093, 533), (1077, 529), (1062, 528), (1048, 530), (1036, 535), (1027, 543), (1020, 552), (1014, 562), (1010, 572), (1008, 581), (1006, 590), (1005, 598), (1005, 604), (1005, 610), (1005, 614), (1004, 616), (1004, 617), (1004, 617)]}

		sample_sets = list(test_dict.values())
		# print(len(sample_sets))
		sample_sets = [sample_sets[3], sample_sets[6], sample_sets[8]]
		sample_sets = [sample_sets[2]]

		mirror_sets = [] #get_mirrored(r, sample_sets)
		# print(len(mirror_sets))
		for p in mirror_sets:
			sample_sets.append(p)

		# print(len(sample_sets))
		# exit()
		# print(test_dict.keys())
		# sample_sets = [sample_sets[0]]


	if sampling_type == SAMPLE_TYPE_SHORTEST or sampling_type == SAMPLE_TYPE_FUSION:
		goals = r.get_goals_all()
		# sample_sets = []

		for g in goals:
			min_path = get_min_viable_path(r, goal, exp_settings)
			sample_sets.append(min_path)

	if sampling_type == SAMPLE_TYPE_CENTRAL or sampling_type == SAMPLE_TYPE_FUSION:
		resolution = exp_settings['resolution']
		low_x, hi_x, low_y, hi_y = get_hi_low_of_pts(r)
		# print(get_hi_low_of_pts(r))

		SAMPLE_BUFFER = 150

		hi_y 	+= SAMPLE_BUFFER
		low_y 	-= SAMPLE_BUFFER
		hi_x 	+= SAMPLE_BUFFER

		xi_range = range(low_x, hi_x, resolution)
		yi_range = range(low_y, hi_y, resolution)

		for xi in range(low_x, hi_x, resolution):
			for yi in range(low_y, hi_y, resolution):
				# print(xi, yi)
				x = int(xi)
				y = int(yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

		# print(point_set)

	if sampling_type == SAMPLE_TYPE_CENTRAL_SPARSE:
		resolution_sparse = exp_settings['resolution'] * 3

		low_x, hi_x, low_y, hi_y = get_hi_low_of_pts(r)
		
		SAMPLE_BUFFER = 150

		hi_y 	+= SAMPLE_BUFFER
		low_y 	-= SAMPLE_BUFFER
		hi_x 	+= SAMPLE_BUFFER

		xi_range = range(low_x, hi_x, resolution_sparse)
		yi_range = range(low_y, hi_y, resolution_sparse)

		# print(list(xi_range))
		# print(list(yi_range))


		count = 0
		for xi in xi_range:
			for yi in yi_range:
				# print(xi, yi)
				x = int(xi)
				y = int(yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

		mirror_sets = get_mirrored(r, sample_sets)
		for p in mirror_sets:
			sample_sets.append(p)

		# print(sample_sets)
		# print(start)
		# exit()

		# k = len(sample_sets)
		# k = int(.3*k)
		# k = 44
		# sample_sets = sample_sets[k:k+5]

		# print(sample_sets)

	if sampling_type == SAMPLE_TYPE_HARDCODED:
		sx, sy, stheta = start
		gx, gy, gt = goal

		mx = int((sx + gx)/2.0)
		my = int((sy + gy)/2.0)

		resolution = 200
		point_set = []
		sample_sets.append(point_set)

		# point_set = [(sx + resolution,		sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx + 2*resolution,	sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx + 3*resolution,	sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx + 4*resolution,	sy)]
		# sample_sets.append(point_set)


		point_set = [(mx + resolution,		my)]
		sample_sets.append(point_set)
		point_set = [(mx + 2*resolution,	my)]
		sample_sets.append(point_set)
		point_set = [(mx + 3*resolution,	my)]
		sample_sets.append(point_set)
		point_set = [(mx + 4*resolution,	my)]
		sample_sets.append(point_set)


		# point_set = [(sx - resolution,sy)]
		# sample_sets.append(point_set)
		# point_set = [(sx,sy + resolution)]
		# sample_sets.append(point_set)
		# point_set = [(sx,sy - resolution)]
		# sample_sets.append(point_set)
		# point_set = [(sx - resolution,sy + resolution)]
		# sample_sets.append(point_set)
		# point_set = [(sx - resolution,sy - resolution)]
		# sample_sets.append(point_set)

		# point_set = [(gx - resolution,gy)]
		# sample_sets.append(point_set)
		# # point_set = [(gx - resolution,gy + resolution)]
		# # sample_sets.append(point_set)
		# # point_set = [(gx - resolution,gy - resolution)]
		# # sample_sets.append(point_set)
		# point_set = [(gx - 2*resolution,gy)]
		# sample_sets.append(point_set)

		# # point_set = [(gx - 2*resolution,gy + resolution)]
		# # sample_sets.append(point_set)
		# # point_set = [(gx - 2*resolution,gy - resolution)]
		# # sample_sets.append(point_set)

		# point_set = [(gx - 3*resolution,gy)]
		# sample_sets.append(point_set)

		# point_set = [(gx - 4*resolution,gy)]
		# sample_sets.append(point_set)


	return sample_sets

def lam_to_str(lam):
	return str(lam)#.replace('.', ',')

def eps_to_str(eps):
	return str(eps)#.replace('.', ',')

def fn_export_from_exp_settings(exp_settings):
	title 				= exp_settings['title']
	sampling_type 		= exp_settings['sampling_type']
	eps 				= exp_settings['epsilon']
	lam 				= exp_settings['lambda']
	n_chunks 			= exp_settings['num_chunks']
	chunking_type 		= exp_settings['chunk_type']
	astr 				= exp_settings['angle_strength']
	FLAG_is_denominator = exp_settings['is_denominator']
	rez 				= exp_settings['resolution']
	f_label 			= exp_settings['f_vis_label']
	fov 				= exp_settings['fov']
	prob_og 			= exp_settings['prob_og']
	right_bound 		= exp_settings['right-bound']
	# exp_settings['f_vis']			= f_exp_single_normalized
	# exp_settings['angle_cutoff']	= 70

	is_denom = 0
	if FLAG_is_denominator:
		is_denom = 1

	is_denom 	= str(is_denom)
	eps 		= eps_to_str(eps)
	lam 		= lam_to_str(lam)
	prob_og 	= str(int(prob_og))

	unique_title = title + "_fnew=" + str(is_denom) + "_"
	unique_title += sampling_type + "-lam" + lam + "_" + str(chunking_type) + "-" + str(n_chunks) 
	unique_title += "-as-" + str(astr) + 'fov=' + str(fov)
	unique_title += "-rb" + str(right_bound)
	unique_title += 'pog=' + prob_og

	fn = FILENAME_PATH_ASSESS + unique_title + "/"

	if not os.path.exists(fn):
		os.mkdir(fn)

	fn += unique_title
	return fn


def fn_pathpickle_from_exp_settings(exp_settings, goal_index):
	sampling_type = exp_settings['sampling_type']
	n_chunks = exp_settings['num_chunks']
	angle_str = exp_settings['angle_strength']
	res = exp_settings['resolution']

	fn_pickle = FILENAME_PATH_ASSESS + "export-" + sampling_type + "-g" + str(goal_index)
	fn_pickle += "ch" + str(n_chunks) +"as" + str(angle_str) + "res" + str(res) +  ".pickle"
	print("{" + fn_pickle + "}")
	return fn_pickle


def fn_pathpickle_envir_cache(exp_settings):
	sampling_type = exp_settings['sampling_type']
	n_chunks = exp_settings['num_chunks']
	angle_str = exp_settings['angle_strength']
	res = exp_settings['resolution']
	f_vis_label 	= exp_settings['f_vis_label']
	FLAG_is_denominator = exp_settings['is_denominator']
	is_denom = 0
	if FLAG_is_denominator:
		is_denom = 1
	is_denom = str(is_denom)

	fn_pickle = FILENAME_PATH_ASSESS + "export-envir-cache-" + sampling_type + "-" + f_vis_label
	fn_pickle += "ch" + str(n_chunks) +"as" + str(angle_str) + "res" + str(res) +  ".pickle"

	# print("{" + fn_pickle + "}")
	return fn_pickle

def numpy_to_image(data):
	pretty_data = copy.copy(data.T)
	pretty_data = cv2.flip(pretty_data, 0)
	return pretty_data


def export_envir_cache_pic(r, data, label, g_index, exp_settings):
	pretty_data = numpy_to_image(data)	

	# We use this to flip vertically
	r_height = 1000

	fig, ax = plt.subplots()
	ax.imshow(pretty_data, interpolation='nearest')
	obs_sets = r.get_obs_sets()
	xs, ys = [], []

	for ok in obs_sets.keys():
		obs_xy = obs_sets[ok]
		if len(obs_xy) > 0:
			obs_xy = obs_xy[0].get_center()
			x = obs_xy[0]
			y = r_height - obs_xy[1]
			xs.append(x)
			ys.append(y)
	
	ax.plot(xs, ys, 'ro')
	xs, ys = [], []

	for goal in r.get_goals_all():
		gx, gy = goal[0], goal[1]
		xs.append(gx)
		ys.append(r_height - gy)

	ax.plot(xs, ys, 'go')
	xs, ys = [], []

	for table in r.get_tables():
		table = table.get_center()
		tx, ty = table[0], table[1]
		xs.append(tx)
		ys.append(r_height - ty)

	ax.plot(xs, ys, 'yo')
	xs, ys = [], []

	start = r.get_start()
	sx = start[0]
	sy = r_height - start[1]
	xs.append(sx)
	ys.append(sy)

	ax.plot(xs, ys, 'bo')
	xs, ys = [], []

	plt.tight_layout()

	plt.savefig(fn_export_from_exp_settings(exp_settings) + "g="+ str(g_index) +  '-' + label + '-plot'  + '.png')
	plt.clf()

def get_envir_cache(r, exp_settings):
	f_vis 		= exp_settings['f_vis']
	f_vis_label	= exp_settings['f_vis_label']
	
	fn_pickle = fn_pathpickle_envir_cache(exp_settings)

	if os.path.isfile(fn_pickle):
		with open(fn_pickle, "rb") as f:
			try:
				envir_cache = pickle.load(f)		
				print("\tImported pickle of envir cache @ " + f_pickle)

			except Exception: # so many things could go wrong, can't be more specific.
				pass

	if FLAG_REDO_ENVIR_CACHE or not os.path.isfile(fn_pickle):
		envir_cache = {}
		tic = time.perf_counter()
		print("Getting start to here dict")
		envir_cache[ENV_START_TO_HERE] = get_dict_cost_start_to_here(r, exp_settings)
		toc = time.perf_counter()
		print(f"\tCalculated start to here in {toc - tic:0.4f} seconds")
		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()


		print("Getting here to goals dict")
		tic = time.perf_counter()
		envir_cache[ENV_HERE_TO_GOALS] = get_dict_cost_here_to_goals_all(r, exp_settings)
		toc = time.perf_counter()
		print(f"\tCalculated here to goals in {toc - tic:0.4f} seconds")
		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()

		print("Getting visibility per obs dict")
		tic = time.perf_counter()
		envir_cache[ENV_VISIBILITY_PER_OBS] = get_dict_vis_per_obs_set(r, exp_settings, f_vis)
		toc = time.perf_counter()
		print(f"\tCalculated vis per obs in {toc - tic:0.4f} seconds")
		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()

		print("Done with pickle")



		dbfile = open(fn_pickle, 'wb')
		pickle.dump(envir_cache, dbfile)
		dbfile.close()

	return envir_cache

def get_dict_cost_here_to_goals_all(r, exp_settings):
	goals = r.get_goals_all()
	all_goals = {}

	for g_index in range(len(goals)):
		g = goals[g_index]
		all_goals[g] = get_dict_cost_here_to_goal(r, g, exp_settings)
		export_envir_cache_pic(r, all_goals[g], 'here-to-goal', g_index, exp_settings)

	return all_goals

# Get pre-calculated cost from here to goal
# returns (x,y) -> cost_here_to_goal
# for the entire environment
def get_dict_cost_here_to_goal(r, goal, exp_settings):
	dict_start_to_goal = np.zeros((r.get_width(), r.get_length()))
	pt_goal = resto.to_xy(goal)

	for i in range(r.get_width()): #r.get_sampling_width():
		# print(str(i) + "... ", end='')
		# if i % 15 ==0:
		# 	print()
		for j in range(r.get_length()): #r.get_sampling_length():
			pt = (i, j)
			val = get_min_direct_path_cost_between(r, pt, pt_goal, exp_settings)
			dict_start_to_goal[i, j] = val

	return dict_start_to_goal

# Get pre-calculated dict of cost from here to goal
# returns (x,y) -> cost_start_to_here
# for the entire environment
def get_dict_cost_start_to_here(r, exp_settings):
	dict_start_to_goal = np.zeros((r.get_width(), r.get_length()))
	start = resto.to_xy(r.get_start())
	# print(dict_start_to_goal.shape)

	for i in range(r.get_width()):
		# print(str(i) + "... ", end='')
		# if i % 15 ==0:
		# 	print()
		for j in range(r.get_length()): #r.get_sampling_length():
			pt = (i, j)
			# print(pt)
			val = get_min_direct_path_cost_between(r, start, resto.to_xy(pt), exp_settings)
			dict_start_to_goal[i, j] = val
	
	export_envir_cache_pic(r, dict_start_to_goal, 'start-to-here', "-", exp_settings)
	return dict_start_to_goal

def get_dict_vis_per_obs_set(r, exp_settings, f_vis):
	# f_vis = f_exp_single(t, pt, aud, path)
	
	obs_sets = r.get_obs_sets()
	all_vis_dict = {}
	for ok in obs_sets.keys():
		print("Getting obs vis for " + ok)
		os = obs_sets[ok]
		os_vis = np.zeros((r.get_width(), r.get_length()))
		# print(os_vis.shape)

		for i in range(r.get_width()): #r.get_sampling_length():
			# print(str(i) + "... ", end='')
			# if i % 15 ==0:
			# 	print()
			for j in range(r.get_length()): #r.get_sampling_width():
				# print(str(j) + "... ", end='')
				pt = (i, j)
				val = f_vis(None, pt, os, None)
				# if len(os) > 0:
				# if len(os) > 0 and (i % 50 == 0) and (j % 50 == 0):
				# 	print(str(pt) + " -> " + str(os[0].get_center()) + " = " + str(val))
				os_vis[i, j] = val

		all_vis_dict[ok] = os_vis
		print("\texporting " + ok)
		export_envir_cache_pic(r, os_vis, 'obs_angle', ok, exp_settings)

	return all_vis_dict

def title_from_exp_settings(exp_settings):
	title = exp_settings['title']
	sampling_type = exp_settings['sampling_type']
	eps = exp_settings['epsilon']
	lam = exp_settings['lambda']
	n_chunks 		= exp_settings['num_chunks']
	chunking_type 	= exp_settings['chunk_type']
	angle_strength = exp_settings['angle_strength']

	FLAG_is_denominator = exp_settings['is_denominator']
	rez 				= exp_settings['resolution']
	f_label 			= exp_settings['f_vis_label']
	fov 				= exp_settings['fov']
	prob_og 			= exp_settings['prob_og']
	right_bound 		= exp_settings['right-bound']

	eps = eps_to_str(eps)
	lam = lam_to_str(lam)

	cool_title = title + ": " + sampling_type + " ang_str=" + str(angle_strength)
	cool_title += "\nright_bound=" + str(right_bound) + " fov=" + str(fov)
	cool_title += "\nlam=" + lam + "     prob_og=" + str(prob_og)
	cool_title += "\nn=" + str(n_chunks) + " distr=" + str(chunking_type)

	return cool_title

# Convert sample points into actual useful paths
def get_paths_from_sample_set(r, exp_settings, goal_index):
	sampling_type = exp_settings['sampling_type']

	sample_pts = get_sample_points_sets(r, r.get_start(), r.get_goals_all()[goal_index], exp_settings)
	print("\tSampled " + str(len(sample_pts)) + " points using the sampling type {" + sampling_type + "}")

	target = r.get_goals_all()[goal_index]
	all_paths = []
	fn_pickle = fn_pathpickle_from_exp_settings(exp_settings, goal_index)

	print("\t Looking for import @ " + fn_pickle)

	if not FLAG_REDO_PATH_CREATION and os.path.isfile(fn_pickle):
		print("\tImporting preassembled paths")
		with open(fn_pickle, "rb") as f:
			try:
				path_dict = pickle.load(f)		
				print("\tImported pickle of combo (goal=" + str(goal_index) + ", sampling=" + str(sampling_type) + ")")
				print("imported " + str(len(all_paths)) + " paths")
				return path_dict

			except Exception: # so many things could go wrong, can't be more specific.
				pass

	if sampling_type not in premade_path_sampling_types:
		print("\tDidn't import; assembling set of paths")
		all_paths = []
		# If I don't yet have a path
		for point_set in sample_pts:
			
			# path_option = construct_single_path_with_angles(exp_settings, r.get_start(), target, point_set, fn_export_from_exp_settings(exp_settings))
			# path_option = chunkify.chunkify_path(exp_settings, path_option)
			# all_paths.append(path_option)

			try:
				path_option_2 = construct_single_path_with_angles_spline(exp_settings, r.get_start(), target, point_set, fn_export_from_exp_settings(exp_settings), is_weak=True)
				path_option_2 = chunkify.chunkify_path(exp_settings, path_option_2)
				all_paths.append(path_option_2)
			except Exception:
				all_paths.append([])
	else:
		all_paths = sample_pts

	path_dict = {'path': all_paths, 'sp': sample_pts}

	dbfile = open(fn_pickle, 'wb')
	pickle.dump(path_dict, dbfile)
	dbfile.close()
	print("\tSaved paths for faster future run on combo (goal=" + str(goal_index) + ", sampling=" + str(sampling_type) + ")")

	return path_dict

# TODO Ada
def create_systematic_path_options_for_goal(r, exp_settings, start, goal, img, num_paths=500):
	all_paths = []
	target = goal

	label = exp_settings['title']
	sampling_type = exp_settings['sampling_type']


	fn = FILENAME_PATH_ASSESS + label + "_sample_path" + ".png"
	goal_index = r.get_goal_index(goal)

	title = title_from_exp_settings(exp_settings)

	min_paths = [get_min_viable_path(r, goal, exp_settings)]
	resto.export_raw_paths(r, img, min_paths, title, fn_export_from_exp_settings(exp_settings)+ "_g" + str(goal_index) + "-min")

	all_paths_dict = get_paths_from_sample_set(r, exp_settings, goal_index)
	resto.export_raw_paths(r, img, all_paths_dict['path'], title, fn_export_from_exp_settings(exp_settings)+ "_g" + str(goal_index) + "-all")

	trimmed_paths, removed_paths, trimm_sp = trim_paths(r, all_paths_dict, goal, exp_settings)
	resto.export_raw_paths(r, img, trimmed_paths, title, fn_export_from_exp_settings(exp_settings) + "_g" + str(goal_index) + "-trimmed")
	resto.export_raw_paths(r, img, removed_paths, title, fn_export_from_exp_settings(exp_settings) + "_g" + str(goal_index) + "-rmvd")

	return trimmed_paths, trimm_sp


def experimental_scenario_single():
	generate_type = resto.TYPE_EXP_SINGLE

	# SETUP FROM SCRATCH, AND SAVE OPTIONS
	if FLAG_SAVE:
		# Create the restaurant scene from our saved description of it
		print("PLANNER: Creating layout of type " + str(generate_type))
		r 	= resto.Restaurant(generate_type)
		# print("PLANNER: get visibility info")

		if FLAG_VIS_GRID:
			# If we'd like to make a graph of what the visibility score is at different points
			# df_vis = r.get_visibility_of_pts_pandas(f_visibility)

			# dbfile = open(vis_pickle, 'ab') 
			# pickle.dump(df_vis, dbfile)					  
			# dbfile.close()
			# print("Saved visibility map")

			# df_vis.to_csv('visibility.csv')
			# print("Visibility point grid created")
			pass
		
		# pickle the map for future use
		dbfile = open(resto_pickle, 'ab') 
		pickle.dump(r, dbfile)					  
		dbfile.close()
		print("Saved restaurant pickle")

	# OR LOAD FROM FILE
	else:
		dbfile = open(resto_pickle, 'rb')
		r = pickle.load(dbfile)
		print("Imported pickle of restaurant")


		if FLAG_VIS_GRID:
			dbfile = open(vis_pickle, 'rb')
			df_vis = pickle.load(dbfile)
			print("Imported pickle of vis")


	return r


def image_to_planner(resto, pt):
	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	# planner always starts at (0,0)
	gx, gy, gtheta = resto.get_start()


	x = float(x) - gx
	y = float(y) - gy
	nx = x #(x / UNITY_SCALE_X) + UNITY_OFFSET_X
	ny = y #(y / UNITY_SCALE_Y) + UNITY_OFFSET_Y
	
	if len(pt) == 3:
		return (int(ny), int(nx), theta)
	return (int(ny), int(nx))

def planner_to_image(resto, pt):
	if len(pt) == 3:
		x, y, theta = pt
	else:
		x, y = pt

	# planner always starts at (0,0)
	gx, gy, gtheta = resto.get_start()


	x = float(x) + gx
	y = float(y) + gy
	nx = y #(x / UNITY_SCALE_X) + UNITY_OFFSET_X
	ny = x #(y / UNITY_SCALE_Y) + UNITY_OFFSET_Y
	
	if len(pt) == 3:
		return (int(ny), int(nx), np.deg2rad(theta))
	return (int(ny), int(nx))

def angle_between_points(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	angle = np.arctan2(y2 - y1, x2 - x1)

	# ang1 = np.arctan2(*p1[::-1])
	# ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg(angle)

def angle_between_lines(l1, l2):
	# l1_x1, l1_y1 = l1[0]
	# l1_x2, l1_y2 = l1[1]
	# l2_x1, l2_y1 = l2[0]
	# l2_x2, l2_y2 = l2[1]

	p1a, p1b = l1
	p2a, p2b = l2

	a1 = angle_between_points(p1a, p1b)
	a2 = angle_between_points(p2a, p2b)
	angle = (a1 - a2)

	# cosTh = np.dot(l1,l2)
	# sinTh = np.cross(l1,l2)
	# angle = np.rad2deg(np.arctan2(sinTh,cosTh))
	return angle

def angle_of_turn(l1, l2):
	return (angle_between_lines(l1, l2))

def print_states(resto, states, label):
	img = resto.get_img()

	cv2.circle(img, planner_to_image(resto, (0,0)), 5, (138,43,226), 5)
	for s in states:
		x, y, t = planner_to_image(resto, s)
		print((x,y, t))

		COLOR_RED = (138,43,226)
		cv2.circle(img, (x, y), 5, COLOR_RED, 5)

		angle = s[2];
		length = 20;
		
		ax =  int(x + length * np.cos(angle * np.pi / 180.0))
		ay =  int(y + length * np.sin(angle * np.pi / 180.0))
		cv2.arrowedLine(img, (x,y), (ax, ay), COLOR_RED, 2)
		# print((ax, ay))

	cv2.imwrite(FILENAME_PATH_ASSESS + 'samples-' + label + '.png', img)

def print_path(resto, xc, yc, yawc, label):
	img = resto.get_img()

	cv2.circle(img, planner_to_image(resto, (0,0)), 5, (138,43,226), 5)
	for i in range(len(xc)):
		x = int(xc[i])
		y = int(yc[i])
		yaw = yawc[i]

		COLOR_RED = (138,43,226)
		cv2.circle(img, (x, y), 5, COLOR_RED, 5)

		angle = yaw;
		length = 20;
		
		ax =  int(x + length * np.cos(angle * np.pi / 180.0))
		ay =  int(y + length * np.sin(angle * np.pi / 180.0))
		cv2.arrowedLine(img, (x,y), (ax, ay), COLOR_RED, 2)

	cv2.imwrite(FILENAME_PATH_ASSESS + 'path-' + label + '.png', img)

def make_path_libs(resto, goal):
	start = resto.get_start()
	sx, sy, stheta = image_to_planner(resto, start)
	gx, gy, gtheta = image_to_planner(resto, goal)
	print("FINDING ROUTE TO GOAL " + str(goal))
	show_animation = False

	min_distance = np.sqrt((sx-gx)**2 + (sy-gy)**2)
	target_states = [image_to_planner(resto, goal)]
	# :param goal_angle: goal orientation for biased sampling
	# :param ns: number of biased sampling
	# :param nxy: number of position sampling
	# :param nxy: number of position sampling
	# :param nh: number of heading sampleing
	# :param d: distance of terminal state
	# :param a_min: position sampling min angle
	# :param a_max: position sampling max angle
	# :param p_min: heading sampling min angle
	# :param p_max: heading sampling max angle
	# :return: states list

	k0 = 0.0
	nxy = 7
	nh = 9
	# verify d is reasonable
	d = min_distance
	print("D=" + str(d))
	a_min = - np.deg2rad(15.0)
	a_max = np.deg2rad(15.0)
	p_min = - np.deg2rad(15.0)
	p_max = np.deg2rad(15.0)
	print("calculating states")
	states = slp.calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
	print(states)

	print_states(resto, states, 'calc-unif')

	print("calculating results")
	result = slp.generate_path(states, k0)
	print(result)

	for table in result:
		xc, yc, yawc = slp.motion_model.generate_trajectory(
			table[3], table[4], table[5], k0)

		print("gen trajectory")
		print((xc, yc, yawc))

		if show_animation:
			plt.plot(xc, yc, "-r")
			print(xc, yc)

	if show_animation:
		plt.grid(True)
		plt.axis("equal")
		plt.show()

	print("Done")


def export_path_options_for_each_goal(restaurant, best_paths, exp_settings):
	# print(best_paths)
	img = restaurant.get_img()
	 #cv2.flip(img, 0)
	# cv2.imwrite(FILENAME_PATH_ASSESS + unique_key + 'empty.png', empty_img)

	fn = FILENAME_PATH_ASSESS
	title = title_from_exp_settings(exp_settings)

	# flip required for orientation
	font_size = 1
	y0, dy = 50, 50
	for i, line in enumerate(title.split('\n')):
	    y = y0 + i*dy
	    cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (209, 80, 0, 255), 3)
	
	empty_img = img
	all_img = img

	color_dict = restaurant.get_obs_sets_colors()

	goal_imgs = {}
	for pkey in best_paths.keys():
		goal 		= pkey[0]
		# audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		goal_imgs[goal_index] = copy.copy(empty_img)


	for pkey in best_paths.keys():
		path = best_paths[pkey]
		path = restaurant.path_to_printable_path(path)
		path_img = img.copy()
		
		goal 		= pkey[0]
		audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		goal_img = goal_imgs[goal_index]
		obs_key = pkey[1]
		solo_img = restaurant.get_obs_img(obs_key)
		
		color = color_dict[audience]

		# Draw the path  
		for i in range(len(path) - 1):
			a = path[i]
			b = path[i + 1]
			
			cv2.line(solo_img, a, b, color, thickness=3, lineType=8)
			cv2.circle(solo_img, a, 4, color, 4)

			if audience is not 'naked':
				cv2.line(goal_img, a, b, color, thickness=3, lineType=8)
				cv2.line(all_img, a, b, color, thickness=3, lineType=8)
				cv2.circle(goal_img, a, 4, color, 4)
				cv2.circle(all_img, a, 4, color, 4)
		
		title = exp_settings['title']

		sampling_type = exp_settings['sampling_type']
		cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_solo_path-g=' + str(goal_index)+ "-aud=" + str(audience) + '.png', solo_img) 
		print("exported image of " + str(pkey) + " for goal " + str(goal_index))


	for goal_index in goal_imgs.keys():
		goal_img = goal_imgs[goal_index]

		cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_goal_' + str(goal_index) + '.png', goal_img) 

	cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_overview_yay'+ '.png', all_img)

	# TODO: actually export pics for them

def get_columns_metric(r, df):
	columns = df.columns.tolist()
	for col in non_metric_columns:
		if col in columns:
			columns.remove(col)
	return columns

def get_columns_env(r, df):
	columns = df.columns.tolist()
	new_cols = []
	for col in columns:
		if 'env' in col:
			new_cols.append(col)
	return new_cols

def get_columns_legibility(r, df):
	columns = df.columns.tolist()
	new_cols = ['naked']
	for col in r.get_obs_sets().keys():
		new_cols.append(col)
	return new_cols

def dict_to_leg_df(r, data, exp_settings):
	df = pd.DataFrame.from_dict(data)
	columns = get_columns_metric(r, df)

	for col in columns:
		df = df.astype({col: float})

	export_legibility_df(r, df, exp_settings)

	return df

def rgb_to_hex(red, green, blue):
	"""Return color as #rrggbb for the given color values."""
	return '#%02x%02x%02x' % (red, green, blue)

def export_legibility_df(r, df, exp_settings):
	title = exp_settings['title']
	sampling_type = exp_settings['sampling_type']

	df.to_csv(fn_export_from_exp_settings(exp_settings) + "_legibilities.csv")

	df.describe().to_csv(fn_export_from_exp_settings(exp_settings) + "_description.csv")

	print(get_columns_metric(r, df))
	print(get_columns_env(r, df))
	print(get_columns_legibility(r, df))

	# columns_env = get_columns_pure_vis(r, df)
	# make_overview_plot(r, df, exp_settings, columns_env, 'env')

	columns_env = get_columns_env(r, df)
	make_overview_plot(r, df, exp_settings, columns_env, 'env')

	columns_legi = get_columns_legibility(r, df)
	make_overview_plot(r, df, exp_settings, columns_legi, 'legi')

def make_overview_plot(r, df, exp_settings, columns, label):
	all_goals = df["goal"].unique().tolist()
	df_array = []

	g_index = 0
	if False:
		for g in all_goals:
			df_new = df[df['goal'] == g]
			df_array.append(df_new)


			df_new.plot.box(vert=False) # , by=["goal"]
			# bp = df.boxplot(by="goal") #, column=columns)
			# bp = df.groupby('goal').boxplot()

			plt.tight_layout()
			#save the plot as a png file
			plt.savefig(fn_export_from_exp_settings(exp_settings) + "g="+ str(g_index) +  '-desc_plot_' + label  + '.png')
			plt.clf()

			g_index += 1


	obs_palette = r.get_obs_sets_hex()
	goal_labels = r.get_goal_labels()

	goals_list = r.get_goals_all()

	if FLAG_MIN_MODE:
		if 'omni' in columns:
			columns = ['omni', 'c']
		if 'omni-env' in columns:
			columns = ['omni-env', 'c-env']


	df_a = df[df['goal'] == goals_list[0]]
	df_b = df[df['goal'] == goals_list[1]]

	df_a.loc[:,"goal"] = df_a.loc[:, "goal"].map(goal_labels)
	df_b.loc[:,"goal"] = df_b.loc[:, "goal"].map(goal_labels)

	df_a = df_a[columns]
	df_b = df_b[columns]

	# make the total overview plot
	contents_a = np.round(df_a.describe(), 2)
	contents_b = np.round(df_b.describe(), 2)

	contents_a.loc['count'] = contents_a.loc['count'].astype(int).astype(str)
	contents_b.loc['count'] = contents_b.loc['count'].astype(int).astype(str)

	fig_grid = plt.figure(figsize=(10, 6), constrained_layout=True)
	gs = gridspec.GridSpec(ncols=2, nrows=2,
						 width_ratios=[1, 1], wspace=None,
						 hspace=None, height_ratios=[1, 2], figure=fig_grid)

	cool_title = title_from_exp_settings(exp_settings)
	plt.suptitle(cool_title)

	# gs.update(wspace=1)
	ax1 = plt.subplot(gs[0, :1], )
	ax2 = plt.subplot(gs[0, 1:])
	ax3 = plt.subplot(gs[1, 0:2])

	ax1.axis('off')
	ax2.axis('off')

	table_a = table(ax1, contents_a, loc="center")
	table_b = table(ax2, contents_b, loc="center")

	table_a.auto_set_font_size(False)
	table_b.auto_set_font_size(False)

	table_a.set_fontsize(6)
	table_b.set_fontsize(6)

	# plt.savefig(FILENAME_PATH_ASSESS + title + "_" + sampling_type+  '-table'  + '.png')

	key_cols = columns
	key_cols.append('goal')
	mdf = df[key_cols].melt(id_vars=['goal'])
	ax3 = sns.stripplot(x="goal", y="value", hue="variable", data=mdf, palette=obs_palette, split=True, linewidth=1, edgecolor='gray')	
	if label == 'env':
		ax3.set_ylabel('Size of maximum envelope of visibility')
	else:
		ax3.set_ylabel('Legibility with regard to goal')
	ax3.set_xlabel('Goal')

	ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
	
	# df_new.plot.box(vert=False) # , by=["goal"]
	plt.tight_layout()
	# fig.tight_layout()
	#save the plot as a png file
	plt.savefig(fn_export_from_exp_settings(exp_settings) +  '-desc_plot_' + label + '.png')
	plt.clf()
	

def export_table_all_viewers(r, best_paths, exp_settings):
	# if stat_type == 'env':
	# 	f_function = f_env
	# elif stat_type == 'leg':
	# 	f_function = f_legibility
	# else:
	# 	print("Problem")
	# 	exit()

	obs_sets = r.get_obs_sets()
	obs_keys = list(obs_sets.keys())

	# (target, observer) = value
	data = []

	for key in best_paths:
		path = best_paths[key]
		gkey, target_aud_key = key

		for aud_key in obs_keys:
			actual_audience = obs_sets[aud_key]
		
			f_leg_value = f_legibility(r, gkey, r.get_goals_all(), path, actual_audience, None, exp_settings)
			f_env_value = f_env(r, gkey, r.get_goals_all(), path, actual_audience, None, exp_settings)

			datum = {'goal':gkey, 'target_aud':target_aud_key, 'actual_aud':aud_key, 'legibility':f_leg_value, 'env':f_env_value}
			data.append(datum)

			if target_aud_key == 'omni' and aud_key == 'omni':
				print(gkey)
				print(f_leg_value)
				print(f_env_value)

	df = pd.DataFrame.from_dict(data)
	# print(data)


	fig_grid = plt.figure(figsize=(10, 6), constrained_layout=True)
	gs = gridspec.GridSpec(ncols=2, nrows=2,
						 width_ratios=[1, 1], wspace=None,
						 hspace=None, height_ratios=[1, 1], figure=fig_grid)

	cool_title = title_from_exp_settings(exp_settings)
	plt.suptitle(cool_title)

	goals_list = r.get_goals_all()

	df['legibility'] = df['legibility'].astype(float)
	df['env'] = df['env'].astype(int)

	df_a = df[df['goal'] == goals_list[0]]
	df_b = df[df['goal'] == goals_list[1]]

	df_a_rounded = np.round(df_a, 3)
	df_b_rounded = np.round(df_b, 3)

	# df_c = df[df['goal'] == goals_list[0]]
	# df_d = df[df['goal'] == goals_list[1]]

	# df_a.loc[:,"goal"] = df_a.loc[:, "goal"].map(goal_labels)
	# df_b.loc[:,"goal"] = df_b.loc[:, "goal"].map(goal_labels)

	cols_env = get_columns_legibility(r, df)
	cols_leg = get_columns_env(r, df)


	# df_a = df_a[cols_leg]
	# df_b = df_b[cols_leg]

	# df_c = df_c[cols_env]
	# df_d = df_d[cols_env]

	# print(df.pivot(index='target_aud', columns='actual_aud'))

	# gs.update(wspace=1)
	ax1 = plt.subplot(gs[0, :1], )
	ax2 = plt.subplot(gs[0, 1:])
	ax3 = plt.subplot(gs[1, :1], )
	ax4 = plt.subplot(gs[1, 1:])

	ax1.set_title("Legibility for paths to goal 0")
	ax2.set_title("Legibility for paths to goal 1")
	ax3.set_title("Theoretical Max\n Envelope of Readiness for paths to goal 0")
	ax4.set_title("Theoretical Max\n Envelope of Readiness for paths to goal 1")

	ax1.set_xlabel("Target Audience")
	ax2.set_xlabel("Target Audience")
	ax3.set_xlabel("Target Audience")
	ax4.set_xlabel("Target Audience")

	ax1.set_ylabel("Optimized for Audience")
	ax2.set_ylabel("Optimized for Audience")
	ax3.set_ylabel("Optimized for Audience")
	ax4.set_ylabel("Optimized for Audience")

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')

	# .pivot_table(values='value', index='label', columns='type')
	pt_a = df_a_rounded.pivot_table(values='legibility', index='target_aud', columns='actual_aud', aggfunc='first')
	pt_b = df_b_rounded.pivot_table(values='legibility', index='target_aud', columns='actual_aud', aggfunc='first')
	pt_c = df_a.pivot_table(values='env', index='target_aud', columns='actual_aud', aggfunc='first')
	pt_d = df_b.pivot_table(values='env', index='target_aud', columns='actual_aud', aggfunc='first')

	table_a = table(ax1, pt_a, loc="center")
	table_b = table(ax2, pt_b, loc="center")
	table_c = table(ax3, pt_c, loc="center")
	table_d = table(ax4, pt_d, loc="center")

	# table_a.auto_set_font_size(False)
	# table_b.auto_set_font_size(True)

	# table_a.set_fontsize(6)
	# table_b.set_fontsize(6)

	plt.tight_layout()
	plt.savefig(fn_export_from_exp_settings(exp_settings) +  '-desc_table_voila' + '.png')
	plt.clf()

	return None

# TODO: verify is indexing correctly and grabbing best overall, 
# not best in short zone
def get_best_paths_from_df(r, df, exp_settings):
	best_index = {}
	best_paths = {}
	best_lookup = {}
	best_sample_points = {}

	# symmetry check
	cached_omni_bottom = [(204, 437), (204, 436), (204, 436), (203, 435), (203, 433), (203, 430), (203, 425), (203, 420), (203, 412), (203, 403), (203, 393), (204, 380), (205, 365), (206, 350), (207, 332), (210, 313), (214, 292), (219, 270), (225, 246), (233, 221), (244, 197), (258, 172), (276, 150), (299, 130), (328, 115), (362, 107), (400, 107), (441, 115), (485, 129), (529, 145), (572, 163), (613, 182), (652, 201), (690, 219), (725, 237), (759, 254), (791, 268), (820, 282), (848, 293), (873, 303), (897, 310), (918, 316), (937, 318), (954, 319), (968, 317), (980, 312), (988, 305), (995, 298), (999, 290), (1002, 282), (1003, 274), (1004, 268), (1004, 263), (1004, 260), (1004, 258), (1004, 257)]
	cached_omni_top = get_mirrored_path(r, cached_omni_bottom)

	goals = df['goal'].unique()
	columns = get_columns_legibility(r, df)
	if FLAG_MIN_MODE:
		columns = ['omni', 'c']

	# print("GOALS")
	# print(goals)
	# print(columns)
	FLAG_USE_CACHED = False

	for goal in goals:
		is_goal =  df['goal']==goal
		for col in columns:
			df_goal 	= df[is_goal]
			column 		= df_goal[col]
			# print(column)
			max_index 	= pd.to_numeric(column).idxmax()

			if column is 'omni' and FLAG_USE_CACHED:
				if goal is goals[0]:
					df.index[df['path'] == cached_omni_top].tolist()[0]
				elif goal is goals[0]:
					df.index[df['path'] == cached_omni_bottom].tolist()[0]
				else:
					print("sad")
					exit()

			best_index[(goal, col)] = max_index
			best_paths[(goal, col)] = df.iloc[max_index]['path']
			best_sample_points[(goal, col)] = df.iloc[max_index]['sample_points']
			best_lookup[(goal, col)] = df.iloc[max_index]

	omni_case_goal_0 = best_lookup[(goals[0], 'omni')]
	omni_case_goal_1 = best_lookup[(goals[1], 'omni')]

	if omni_case_goal_0['omni'] > omni_case_goal_1['omni']:
		best_paths[(goals[1], 'omni')] = get_mirrored_path(r, best_paths[(goals[0], 'omni')])
	elif omni_case_goal_1['omni'] > omni_case_goal_0['omni']:
		best_paths[(goals[0], 'omni')] = get_mirrored_path(r, best_paths[(goals[1], 'omni')])


	export_table_all_viewers(r, best_paths, exp_settings)
	
	print("BEST SAMPLE POINTS")
	print(best_sample_points)
	return best_paths, best_index

def analyze_all_paths(r, paths_for_analysis_dict, exp_settings):
	paths 		= None
	goals 		= r.get_goals_all()
	if FLAG_MIN_MODE:
		obs_sets_old 	= r.get_obs_sets()
		obs_sets 	= {}
		obs_sets['omni'] 	= obs_sets_old['omni']
		obs_sets['c'] 		= obs_sets_old['c']
	else:
		obs_sets 	= r.get_obs_sets()

	all_entries = []
	key_index 	= 0

	df_all = []
	data = []

	for key in paths_for_analysis_dict:
		goal 	= key
		paths 	= paths_for_analysis_dict[key]['paths']
		sp 	= paths_for_analysis_dict[key]['sp']

		for pi in range(len(paths)):
			path = paths[pi]
			f_vis = exp_settings['f_vis']
			datum = get_legibilities(r, path, goal, goals, obs_sets, f_vis, exp_settings)
			datum['path'] = path
			datum['goal'] = goal
			datum['sample_points'] = sp[pi]
			datum['path_length'] = get_path_length(path)[1]
			datum['path_cost'] = f_path_cost(path)
			data.append(datum)
			# datum = [goal_index] + []


	# data_frame of all paths overall
	df = dict_to_leg_df(r, data, exp_settings)

	best_paths, best_index = get_best_paths_from_df(r, df, exp_settings)

	export_path_options_for_each_goal(r, best_paths, exp_settings)
	return best_paths

def do_exp(lam, km):
	# Run the scenario that aligns with our use case
	restaurant = experimental_scenario_single()
	unique_key = 'exp_single'
	start = restaurant.get_start()
	all_goals = restaurant.get_goals_all()

	sample_pts = []
	# sampling_type = SAMPLE_TYPE_CENTRAL
	# sampling_type = SAMPLE_TYPE_DEMO
	# sampling_type = SAMPLE_TYPE_CENTRAL_SPARSE
	# sampling_type = SAMPLE_TYPE_FUSION
	# sampling_type = SAMPLE_TYPE_SPARSE
	# sampling_type = SAMPLE_TYPE_SYSTEMATIC
	# sampling_type = SAMPLE_TYPE_HARDCODED
	# sampling_type = SAMPLE_TYPE_VISIBLE
	# sampling_type = SAMPLE_TYPE_INZONE
	# sampling_type = SAMPLE_TYPE_CURVE_TEST
	sampling_type = SAMPLE_TYPE_NEXUS_POINTS


	OPTION_DOING_STATE_LATTICE = False
	if OPTION_DOING_STATE_LATTICE:
		for i in range(len(all_goals)):
			goal = all_goals[i]
			# lane_state_sampling_test1(resto, goal, i)
			make_path_libs(resto, goal)

	exp_settings = {}
	exp_settings['title'] 			= unique_key
	exp_settings['sampling_type'] 	= sampling_type
	exp_settings['resolution']		= 15
	exp_settings['f_vis_label']		= 'fcut'
	exp_settings['epsilon'] 		= 0 #1e-12 #eps #decimal.Decimal(1e-12) # eps #.000000000001
	exp_settings['lambda'] 			= lam #decimal.Decimal(1e-12) #lam #.000000000001
	exp_settings['num_chunks']		= 50
	exp_settings['chunk-by-what']	= chunkify.CHUNK_BY_DURATION
	exp_settings['chunk_type']		= chunkify.CHUNKIFY_TRIANGULAR #LINEAR	# CHUNKIFY_LINEAR, CHUNKIFY_TRIANGULAR, CHUNKIFY_MINJERK
	exp_settings['angle_strength']	= 550 #40
	exp_settings['min_path_length'] = {}
	exp_settings['is_denominator']	= False
	exp_settings['f_vis']			= f_exp_single_normalized
	exp_settings['kill_1']			= km
	exp_settings['angle_cutoff']	= 70
	exp_settings['fov']	= 120
	exp_settings['prob_og']			= False
	exp_settings['right-bound']		= 20


	# Preload envir cache for faster calculations
	envir_cache = get_envir_cache(restaurant, exp_settings)
	restaurant.set_envir_cache(envir_cache)

	print("Prepped environment")
	# print(envir_cache)

	# SET UP THE IMAGES FOR FUTURE DRAWINGS
	img = restaurant.get_img()
	cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_empty.png', img)

	min_paths = []
	for g in restaurant.get_goals_all():
		print("Finding min path for goal " + str(g))
		min_path_length = get_min_viable_path_length(restaurant, g, exp_settings)
		exp_settings['min_path_length'][g] = min_path_length
		
		min_path = get_min_viable_path(restaurant, g, exp_settings)
		min_paths.append(min_path)

	title = title_from_exp_settings(exp_settings)
	resto.export_raw_paths(restaurant, img, min_paths, title, fn_export_from_exp_settings(exp_settings) + "_all" + "-min")

	paths_for_analysis = {}
	#  add permutations of goals with some final-angle-wiggle
	for goal in all_goals:
		print("Generating paths for goal " + str(goal))
		paths, sample_pts_that_generated = create_systematic_path_options_for_goal(restaurant, exp_settings, start, goal, img, num_paths=10)
		print("Made paths")
		paths_for_analysis[goal] = {'paths':paths, 'sp': sample_pts_that_generated}

	# debug curvatures
	# plt.clf()
	# print(curvatures)
	# sns.histplot(data=curvatures, bins=100)
	# # plt.hist(curvatures, bins=1000) 
	# plt.title("histogram of max angles")
	# plt.tight_layout()
	# plt.savefig("path_assessment/curvatures.png") 
	# plt.clf()

	# sns.histplot(data=max_curvatures, bins=100)
	# # plt.hist(curvatures, bins=1000) 
	# plt.title("histogram of max curvatures")
	# plt.tight_layout()
	# plt.savefig("path_assessment/max-curvatures.png") 
	# exit()

	print("~~~")
	best_paths = analyze_all_paths(restaurant, paths_for_analysis, exp_settings)
		# print(best_paths.keys())

	file1 = open(fn_export_from_exp_settings(exp_settings) + "_BEST_PATHS.txt","w+")
  
	file1.write(str(best_paths))
	file1.close()

	# print(best_paths)



	# # Set of functions for exporting easy paths
	# title = "pts_" + str(exp_settings['num_chunks']) + "_" + str(exp_settings['angle_strength']) + "_" + str(exp_settings['chunk_type']) + " = "
	# path_a = best_paths[((1035, 307, 180), 'omniscient')]
	# path_b = best_paths[((1035, 567, 0), 'omniscient')]

	# path_a = str(path_a)
	# path_b = str(path_b)

	# print(title + path_a)
	# print(title + path_b)
	print("Number of bugs per calculation:")
	print(bug_counter)

	print("Done with experiment")

def exp_determine_lam_eps():
	lam_vals = []
	eps = 1e-7
	# eps_vals = []
	# # exit()
	# for i in range(-5, -10, -1):
	for i in np.arange(1.1, 2, .1):
		new_val = i * 1e-6
	# 	eps_vals.append(new_val)
		lam_vals.append(new_val)
	# 	# lam_vals.append(new_val)

	# print("REMIX TIME")
	# for eps in eps_vals:
	
	for lam in lam_vals:
		do_exp(lam, False)
	# pass

def main():
	# lam = 1e-6
	lam = 0 #(10.0 / 23610)#1e-6 #0 #1.5e-8 #1e-16
	kill_mode = True
	# eps_start = decimal.Decimal(.000000000001)
	# lam_start = decimal.Decimal(.00000000001)
	do_exp(lam, kill_mode)	
	# exp_determine_lam_eps()

if __name__ == "__main__":

	# con = decimal.getcontext()
	# con.prec = 35
	# con.Emax = 1000
	# con.Emin = -1000
	main()






# result = df_vis.pivot(index='x', columns='y', values=resto.VIS_MULTI)
# inspect_heatmap(df_vis)

# print("pivoted")
# heatmap = sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
# print("made heatmap")

# fig = heatmap.get_figure()
# fig.savefig("multi-vis.png")
# print("Graphs")

# resto.draw_paths(r, paths_dict)
# resto.export(r, paths_dict)
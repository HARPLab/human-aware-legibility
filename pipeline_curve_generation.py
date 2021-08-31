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
FLAG_REDO_PATH_CREATION = False
FLAG_REDO_ENVIR_CACHE 	= True

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
SAMPLE_TYPE_DEMO 		= 'demo'
SAMPLE_TYPE_SPARSE		= 'sparse'
SAMPLE_TYPE_SYSTEMATIC 	= 'systematic'
SAMPLE_TYPE_HARDCODED 	= 'hardcoded'
SAMPLE_TYPE_VISIBLE 	= 'visible'
SAMPLE_TYPE_INZONE 		= 'in_zone'
SAMPLE_TYPE_SHORTEST	= 'minpaths'

ENV_START_TO_HERE 		= 'start_to_here'
ENV_HERE_TO_GOALS 		= 'here_to_goals'
ENV_VISIBILITY_PER_OBS 	= 'vis_per_obs'

premade_path_sampling_types = [SAMPLE_TYPE_DEMO, SAMPLE_TYPE_SHORTEST]
non_metric_columns = ["path", "goal", 'path_length', 'path_cost']


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

# Ada final equation
# f_VIS TODO VERIFY
def f_exp_single(t, pt, aud, path):
	# if this is the omniscient case, return the original equation
	if len(aud) == 0 and path is not None:
		return len(path) - t
	elif len(aud) == 0:
		return 0

	# if in the (x, y) OR (x, y, t) case we can totally 
	# still run this equation
	val = get_visibility_of_pt_w_observers(pt, aud)
	return val

def get_visibility_of_pt_w_observers(pt, aud):
	observers = []
	score = 0

	MAX_DISTANCE = 500
	for observer in aud:
		obs_orient 	= observer.get_orientation()
		# if obs_orient != 300:
		# 	print(obs_orient)
		# 	exit()
		obs_FOV 	= observer.get_FOV()

		angle 		= angle_between_points(pt, observer.get_center())
		distance 	= resto.dist(pt, observer.get_center())
		# print("~~~")
		# print(observer.get_center())
		# print(distance)
		# print(pt)
		
		# print(ang)

		angle_diff = abs(obs_orient + angle) % 360

		# if (pt[0] % 50 == 0) and (pt[1] % 50 == 0):
		# 	print(str(pt) + " -> " + str(observer.get_center()) + " = angle " + str(angle))
		# 	print("observer looking at... " + str(obs_orient))
		# 	print("angle diff = " + str(angle_diff))

		# print(angle, distance)
		# observation = (pt, angle, distance)
		# observers.append(observation)

		half_fov = (obs_FOV / 2.0)
		if angle_diff < half_fov:
			score += half_fov - angle_diff

		# 	# full credit at the center of view
		# 	offset_multiplier = np.abs(angle_diff) / obs_FOV

		# 	# # 1 if very close
		# 	# distance_bonus = (MAX_DISTANCE - distance) / MAX_DISTANCE
		# 	# score += (distance_bonus*offset_multiplier)
		# 	score = offset_multiplier
		# 	score = distance

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
		return 0

	return g_target / (sum(g_array))

# Ada: final equation
def unnormalized_prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):
	decimal.getcontext().prec = 20

	start = r.get_start()

	c1 = decimal.Decimal(cost_path_to_here)
	c2 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(pt), resto.to_xy(goal), exp_settings))
	c3 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(start), resto.to_xy(goal), exp_settings))

	a = np.exp((-c1 + -c2))
	b = np.exp(-c3)
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
	path_option = construct_single_path_with_angles(exp_settings, r.get_start(), goal, [], fn_export_from_exp_settings(exp_settings))
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
def f_legibility(resto, goal, goals, path, aud, f_function, exp_settings):
	min_realistic_path_length = exp_settings['min_path_length'][goal]
	print("min_realistic_path_length -> " + str(min_realistic_path_length))
	
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
	delta_x = length_of_total_path / len(aug_path)

	t = 1
	p_n = path[0]
	divisor = epsilon
	numerator = decimal.Decimal(0.0)

	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))
	
		prob_goal_given = prob_goal_given_path(resto, p_n, pt, goal, goals, cost_to_here, exp_settings)

		numerator += ((prob_goal_given * f) * delta_x)
		divisor += delta_x*f

		t = t + 1
		total_cost += decimal.Decimal(f_cost(p_n, pt))
		p_n = pt


	if divisor == 0:
		legibility = 0
	else:
		legibility = (numerator / divisor)

	total_cost =  - LAMBDA*total_cost
	overall = legibility + total_cost

	print("get_path_length output")
	print(get_path_length(path)[1])
	return overall

def get_costs(path, target, obs_sets):
	vals = []

	for aud in obs_sets:
		new_val = f_cost()

	return vals

def get_legibilities(resto, path, target, goals, obs_sets, f_vis, exp_settings):
	vals = {}
	f_vis = exp_settings['f_vis']

	for key in obs_sets.keys():
		aud = obs_sets[key]
		# goal, goals, path, df_obs
		new_val = f_legibility(resto, target, goals, path, aud, f_vis, exp_settings)

		vals[key] = new_val

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

# TODO: add test methods for this
def is_valid_path(restaurant, path):
	# return True
	tables = restaurant.get_tables()
	# print(len(tables))

	for t in tables:
		# print("TABLE MID: " + str(t.get_center()))
		for i in range(len(path) - 1):
			pt1 = path[i]
			pt2 = path[i + 1]
			
			# print((pt1, pt2))

			if t.intersects_line(pt1, pt2):
				# print("Intersection!")

				# print(t.get_center())
				# print(path)
				# print((pt1, pt2))
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
	xs = [int(x) for x in xs]
	ys = [int(y) for y in ys]
	return list(zip(xs, ys))

# https://hal.archives-ouvertes.fr/hal-03017566/document
def construct_single_path_with_angles(exp_settings, start, goal, sample_pts, fn):
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
	# print("x=")
	# print(x)
	# print("y=")
	# print(y)

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
		plt.plot(x_t,y_t,'k',linewidth=2.0,color='orange')

		x_all.extend(x_t)
		y_all.extend(y_t)

	if FLAG_EXPORT_SPLINE_DEBUG:
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
def trim_paths(r, all_paths, goal):
	trimmed_paths = []
	for p in all_paths:
		if is_valid_path(r, p):
			trimmed_paths.append(p)

	print("Paths trimmed: " + str(len(all_paths)) + " -> " + str(len(trimmed_paths)))
	return trimmed_paths

def get_sample_points_sets(r, start, goal, exp_settings):
	# sampling_type = 'systematic'
	# sampling_type = 'visible'
	# sampling_type = 'in_zone'

	sample_sets = []
	resolution = 10

	sampling_type = exp_settings['sampling_type']

	if sampling_type == SAMPLE_TYPE_SYSTEMATIC:
		width = r.get_width()
		length = r.get_length()

		for xi in range(int(width / resolution)):
			for yi in range(int(length / resolution)):
				x = int(resolution * xi)
				y = int(resolution * yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

	elif sampling_type == SAMPLE_TYPE_DEMO:
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

	elif sampling_type == SAMPLE_TYPE_SHORTEST:
		goals = r.get_goals_all()
		sample_sets = []

		for g in goals:
			min_path = get_min_viable_path(r, goal, exp_settings)
			sample_sets.append(min_path)

	elif sampling_type == SAMPLE_TYPE_CENTRAL:
		sx, sy, stheta = start
		gx, gy, gt = goal
		print("sampling central")
		
		low_x, hi_x = sx, gx
		low_y, hi_y = sy, gy

		if sx > gx:
			low_x, hi_x = gx, sx

		if sy > gy:
			low_y, hi_y = gy, sy

		for xi in range(low_x, hi_x, resolution):
			for yi in range(low_y, hi_y, resolution):
				# print(xi, yi)
				x = int(xi)
				y = int(yi)

				point_set = [(x, y)]
				sample_sets.append(point_set)

		# print(point_set)

	elif sampling_type == SAMPLE_TYPE_HARDCODED:
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

def fn_pathpickle_from_exp_settings(exp_settings, goal_index):
	sampling_type = exp_settings['sampling_type']
	fn_pickle = FILENAME_PATH_ASSESS + "export-" + sampling_type + "-" + str(goal_index) + ".pickle"
	print("{" + fn_pickle + "}")
	return fn_pickle


def fn_pathpickle_envir_cache(exp_settings):
	sampling_type 	= exp_settings['sampling_type']
	f_vis_label 	= exp_settings['f_vis_label']

	fn_pickle = FILENAME_PATH_ASSESS + "export-envir-cache-" + sampling_type + "-" + f_vis_label + ".pickle"
	# print("{" + fn_pickle + "}")
	return fn_pickle

def numpy_to_image(data):
	pretty_data = copy.copy(data.T)
	pretty_data = cv2.flip(pretty_data, 0)
	return pretty_data


def export_envir_cache_pic(r, data, label, g_index, exp_settings):
	pretty_data = numpy_to_image(data)	

	fig, ax = plt.subplots()
	ax.imshow(pretty_data, interpolation='nearest')
	obs_sets = r.get_obs_sets()
	xs, ys = [], []

	for ok in obs_sets.keys():
		obs_xy = obs_sets[ok]
		if len(obs_xy) > 0:
			obs_xy = obs_xy[0].get_center()
			x = obs_xy[0]
			y = 1000 - obs_xy[1]
			xs.append(x)
			ys.append(y)
	
	ax.plot(xs, ys, 'ro')
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
	print(dict_start_to_goal.shape)

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
		print(os_vis.shape)

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

	eps = eps_to_str(eps)
	lam = lam_to_str(lam)

	cool_title = title + ": " + sampling_type + " ang_str=" + str(angle_strength)
	cool_title += "\n eps=" + eps + " lam=" + lam
	cool_title += "\n n=" + str(n_chunks) + " distr=" + str(chunking_type)

	return cool_title

def fn_export_from_exp_settings(exp_settings):
	title = exp_settings['title']
	sampling_type = exp_settings['sampling_type']
	eps = exp_settings['epsilon']
	lam = exp_settings['lambda']
	n_chunks 		= exp_settings['num_chunks']
	chunking_type 	= exp_settings['chunk_type']
	astr 				= exp_settings['angle_strength']

	eps = eps_to_str(eps)
	lam = lam_to_str(lam)

	fn = FILENAME_PATH_ASSESS + title + "_" 
	fn += sampling_type + "-ep" + eps + "-lam" + lam + "_" + str(chunking_type) + "-" + str(n_chunks) + "-as-" + str(astr)
	return fn

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
				all_paths = pickle.load(f)		
				print("\tImported pickle of combo (goal=" + str(goal_index) + ", sampling=" + str(sampling_type) + ")")
				print("imported " + str(len(all_paths)) + " paths")
				return all_paths

			except Exception: # so many things could go wrong, can't be more specific.
				pass

	if sampling_type not in premade_path_sampling_types:
		print("\tDidn't import; assembling set of paths")
		# If I don't yet have a path
		for point_set in sample_pts:
			path_option = construct_single_path_with_angles(exp_settings, r.get_start(), target, point_set, fn_export_from_exp_settings(exp_settings))
			path_option = chunkify.chunkify_path(exp_settings, path_option)
			all_paths.append(path_option)
	else:
		all_paths = sample_pts

	dbfile = open(fn_pickle, 'wb')
	pickle.dump(all_paths, dbfile)
	dbfile.close()
	print("\tSaved paths for faster future run on combo (goal=" + str(goal_index) + ", sampling=" + str(sampling_type) + ")")

	return all_paths

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
	resto.export_raw_paths(img, min_paths, title, fn_export_from_exp_settings(exp_settings)+ "_g" + str(goal_index) + "-min")

	all_paths = get_paths_from_sample_set(r, exp_settings, goal_index)
	resto.export_raw_paths(img, all_paths, title, fn_export_from_exp_settings(exp_settings)+ "_g" + str(goal_index) + "-all")

	trimmed_paths = trim_paths(r, all_paths, goal)
	resto.export_raw_paths(img, trimmed_paths, title, fn_export_from_exp_settings(exp_settings) + "_g" + str(goal_index) + "-trimmed")

	return all_paths


def experimental_scenario_single():
	generate_type = resto.TYPE_EXP_SINGLE

	# SETUP FROM SCRATCH, AND SAVE OPTIONS
	if FLAG_SAVE:
		# Create the restaurant scene from our saved description of it
		print("PLANNER: Creating layout of type " + str(generate_type))
		r 	= resto.Restaurant(generate_type)
		print("PLANNER: get visibility info")

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


def print_states(resto, states, label):
	img = resto.get_img()
	img = cv2.flip(img, 0)

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
	img = cv2.flip(img, 0)

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
	empty_img = img #cv2.flip(img, 0)
	# cv2.imwrite(FILENAME_PATH_ASSESS + unique_key + 'empty.png', empty_img)

	fn = FILENAME_PATH_ASSESS

	color_dict = restaurant.get_obs_sets_colors()

	goal_imgs = {}
	for pkey in best_paths.keys():
		goal 		= pkey[0]
		# audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		goal_imgs[goal_index] = copy.copy(empty_img)


	for pkey in best_paths.keys():
		path = best_paths[pkey]
		path_img = img.copy()
		
		goal 		= pkey[0]
		audience 	= pkey[1]
		goal_index 	= restaurant.get_goal_index(goal)

		goal_img = goal_imgs[goal_index]
		solo_img = copy.copy(empty_img)
		
		color = color_dict[audience]

		# Draw the path  
		for i in range(len(path) - 1):
			a = path[i]
			b = path[i + 1]
			
			cv2.line(solo_img, a, b, color, thickness=3, lineType=8)
			cv2.line(goal_img, a, b, color, thickness=3, lineType=8)

			cv2.circle(solo_img, a, 4, color, 4)
			cv2.circle(goal_img, a, 4, color, 4)
		
		solo_img = cv2.flip(solo_img, 0)
		title = exp_settings['title']
		sampling_type = exp_settings['sampling_type']
		cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_solo_path-g=' + str(goal_index)+ "-aud=" + str(audience) + '.png', solo_img) 
		print("exported image of " + str(pkey) + " for goal " + str(goal_index))


	for goal_index in goal_imgs.keys():
		goal_img = goal_imgs[goal_index]

		goal_img = cv2.flip(goal_img, 0)
		cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_goal_' + str(goal_index) + '.png', goal_img) 

	# TODO: actually export pics for them

def get_metric_columns(df):
	columns = df.columns.tolist()
	for col in non_metric_columns:
		columns.remove(col)
	return columns

def dict_to_leg_df(r, data, exp_settings):
	df = pd.DataFrame.from_dict(data)
	columns = get_metric_columns(df)

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

	columns = get_metric_columns(df)

	all_goals = df["goal"].unique().tolist()
	df_array = []

	g_index = 0
	for g in all_goals:
		df_new = df[df['goal'] == g]
		df_array.append(df_new)
		# df.plot(ax=ax, ylim=(0, 2), legend=None);
		# df.plot.hist(orientation="horizontal", cumulative=True);
		
		df_new.plot.box(vert=False) # , by=["goal"]
		# bp = df.boxplot(by="goal") #, column=columns)
		# bp = df.groupby('goal').boxplot()

		plt.tight_layout()
		#save the plot as a png file
		plt.savefig(fn_export_from_exp_settings(exp_settings) + "g="+ str(g_index) +  '-desc_plot'  + '.png')
		plt.clf()

		g_index += 1


	obs_palette = r.get_obs_sets_hex()
	goal_labels = r.get_goal_labels()

	goals_list = r.get_goals_all()

	df_a = df[df['goal'] == goals_list[0]]
	df_b = df[df['goal'] == goals_list[1]]

	df_a.loc[:,"goal"] = df_a.loc[:, "goal"].map(goal_labels)
	df_b.loc[:,"goal"] = df_b.loc[:, "goal"].map(goal_labels)

	# make the total overview plot
	contents_a = np.round(df_a.describe(), 2)
	contents_b = np.round(df_b.describe(), 2)

	contents_a.loc['count'] = contents_a.loc['count'].astype(int).astype(str)
	contents_b.loc['count'] = contents_b.loc['count'].astype(int).astype(str)

	fig_grid = plt.figure(figsize=(9, 6), constrained_layout=True)
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
	ax3 = sns.boxplot(x="goal", y="value", hue="variable", data=mdf, palette=obs_palette)	
	ax3.set_ylabel('Legibility with regard to goal')
	ax3.set_xlabel('Goal')

	ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
	
	# df_new.plot.box(vert=False) # , by=["goal"]
	plt.tight_layout()
	# fig.tight_layout()
	#save the plot as a png file
	plt.savefig(fn_export_from_exp_settings(exp_settings) +  '-desc_plot'  + '.png')
	plt.clf()

# TODO: verify is indexing correctly and grabbing best overall, 
# not best in short zone
def get_best_paths_from_df(df):
	best_paths = {}
	best_index = {}

	# print(df)

	goals = df['goal'].unique()
	columns = get_metric_columns(df)

	# print("GOALS")
	# print(goals)

	for goal in goals:
		is_goal =  df['goal']==goal
		for col in columns:
			df_goal 	= df[is_goal]
			column 		= df_goal[col]
			# print(column)
			max_index 	= pd.to_numeric(column).idxmax()

			best_paths[(goal, col)] = df.iloc[max_index]['path']
			best_index[(goal, col)] = max_index

	# print(best_index)
	return best_paths, best_index

def analyze_all_paths(resto, paths_for_analysis, exp_settings):
	paths 		= None
	goals 		= resto.get_goals_all()
	obs_sets 	= resto.get_obs_sets()

	all_entries = []
	key_index 	= 0

	df_all = []

	data = []

	for key in paths_for_analysis:
		goal 	= key
		paths 	= paths_for_analysis[key]


		for path in paths:
			f_vis = exp_settings['f_vis']
			datum = get_legibilities(resto, path, goal, goals, obs_sets, f_vis, exp_settings)
			datum['path'] = path
			datum['goal'] = goal
			datum['path_length'] = get_path_length(path)[1]
			datum['path_cost'] = f_path_cost(path)
			data.append(datum)
			# datum = [goal_index] + []


	# data_frame of all paths overall
	df = dict_to_leg_df(resto, data, exp_settings)

	best_paths, best_index = get_best_paths_from_df(df)

	export_path_options_for_each_goal(resto, best_paths, exp_settings)
	return best_paths

def main():
	# Run the scenario that aligns with our use case
	restaurant = experimental_scenario_single()
	unique_key = 'exp_single'
	start = restaurant.get_start()
	all_goals = restaurant.get_goals_all()

	sample_pts = []
	# sampling_type = SAMPLE_TYPE_CENTRAL
	sampling_type = SAMPLE_TYPE_DEMO
	# sampling_type = SAMPLE_TYPE_SPARSE
	# sampling_type = SAMPLE_TYPE_SYSTEMATIC
	# sampling_type = SAMPLE_TYPE_HARDCODED
	# sampling_type = SAMPLE_TYPE_VISIBLE
	# sampling_type = SAMPLE_TYPE_INZONE


	OPTION_DOING_STATE_LATTICE = False
	if OPTION_DOING_STATE_LATTICE:
		for i in range(len(all_goals)):
			goal = all_goals[i]
			# lane_state_sampling_test1(resto, goal, i)
			make_path_libs(resto, goal)

	exp_settings = {}
	exp_settings['title'] 			= unique_key
	exp_settings['sampling_type'] 	= SAMPLE_TYPE_DEMO
	exp_settings['f_vis_label']		= 'f_cred_central'
	exp_settings['epsilon'] 		= .000000001
	exp_settings['lambda'] 			= .0000011
	exp_settings['num_chunks']		= 50
	exp_settings['chunk-by-what']	= chunkify.CHUNK_BY_DURATION
	exp_settings['chunk_type']		= chunkify.CHUNKIFY_TRIANGULAR	# CHUNKIFY_LINEAR, CHUNKIFY_TRIANGULAR, CHUNKIFY_MINJERK
	exp_settings['angle_strength']	= 450
	exp_settings['min_path_length'] = {}
	exp_settings['f_vis']			= f_exp_single

	# Preload envir cache for faster calculations
	envir_cache = get_envir_cache(restaurant, exp_settings)
	restaurant.set_envir_cache(envir_cache)

	print("Prepped environment")
	print(envir_cache)

	# SET UP THE IMAGES FOR FUTURE DRAWINGS
	img = restaurant.get_img()
	empty_img = cv2.flip(img, 0)
	cv2.imwrite(fn_export_from_exp_settings(exp_settings) + '_empty.png', empty_img)

	min_paths = []
	for g in restaurant.get_goals_all():
		print("Finding min path for goal " + str(g))
		min_path_length = get_min_viable_path_length(restaurant, g, exp_settings)
		exp_settings['min_path_length'][g] = min_path_length
		
		min_path = get_min_viable_path(restaurant, g, exp_settings)
		min_paths.append(min_path)

	print(exp_settings)

	title = title_from_exp_settings(exp_settings)
	resto.export_raw_paths(img, min_paths, title, fn_export_from_exp_settings(exp_settings) + "_all" + "-min")

	paths_for_analysis = {}
	# TODO add permutations of goals with some final-angle-wiggle
	for goal in all_goals:
		print("Generating paths for goal " + str(goal))
		paths = create_systematic_path_options_for_goal(restaurant, exp_settings, start, goal, img, num_paths=10)
		print("Made paths")
		paths_for_analysis[goal] = paths


	print("~~~")
	best_paths = analyze_all_paths(restaurant, paths_for_analysis, exp_settings)
		# print(best_paths.keys())

	# # Set of functions for exporting easy paths
	# title = "pts_" + str(exp_settings['num_chunks']) + "_" + str(exp_settings['angle_strength']) + "_" + str(exp_settings['chunk_type']) + " = "
	# path_a = best_paths[((1035, 307, 180), 'omniscient')]
	# path_b = best_paths[((1035, 567, 0), 'omniscient')]

	# path_a = str(path_a)
	# path_b = str(path_b)

	# print(title + path_a)
	# print(title + path_b)

	print("Done")



if __name__ == "__main__":
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
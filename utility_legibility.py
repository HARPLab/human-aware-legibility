import copy
import decimal
import numpy as np
import math
import pandas as pd
import matplotlib.pylab as plt

import utility_environ_descrip 		as resto
import utility_path_segmentation 	as chunkify


# FUNCTIONS FOR CALCULATING FEATURES OF PATHS
# SUCH AS VISIBLIITY, LEGIBILITY, PATH_LENGTH, and ENVELOPE

F_JDIST 				= 'JDIST'
F_JHEADING_EXPONENTIAL 	= 'JHEAD_EXPON'
F_JHEADING_QUADRATIC 	= 'JHEAD_QUADR'
F_JHEADING 				= 'JHEAD'

LEGIBILITY_METHOD		= 'l_method'

PROB_INDEX_DIST = 0
PROB_INDEX_HEADING = 1


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

def f_og(t, path):
	# len(path)
	return NUMBER_STEPS - t


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

# uniform weighting function
def f_naked(t, pt, aud, path):
	return decimal.Decimal(1.0)

# Ada final equation
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

# OBSERVER-AWARE LEGIBILITY PAPER ROMAN VERSION
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


# ADA MASTER VISIBILITY EQUATION
# OBSERVER-AWARE EQUATION
def get_visibility_of_pt_w_observers(pt, aud, normalized=True):
	observers = []
	score = []

	reasonable_set_sizes = [0, 1, 5]
	if len(aud) not in reasonable_set_sizes:
		print(len(aud))

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



def prob_overall_fuse_signals(probs_array_goal_given_signals, r, p_n, pt, goal, goals, cost_to_here, exp_settings):
	COMPONENT_DIST 			= decimal.Decimal(probs_array_goal_given_signals[PROB_INDEX_DIST])
	COMPONENT_HEADING 		= decimal.Decimal(probs_array_goal_given_signals[PROB_INDEX_HEADING])

	return COMPONENT_DIST + COMPONENT_HEADING


# Ada: Final equation
# TODO Cache this result for a given path so far and set of goals
def prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings, unnorm_prob_function):
	entry = []

	start = r.get_start()
	g_array = []
	g_target = 0
	for g in goals:
		p_raw = unnorm_prob_function(r, p_n1, pt, g, goals, cost_path_to_here, exp_settings)
		g_array.append(p_raw)
		if g is goal:
			g_target = p_raw

	if(sum(g_array) == 0):
		print("weird g_array")
		return decimal.Decimal(1.0)

	return decimal.Decimal(g_target / (sum(g_array)))


# Ada: Heading-aware version of legibility
def unnormalized_prob_goal_given_path_use_heading(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):

	prob_val = prob_goal_given_heading(r.get_start(), p_n1, pt, goal, goals, cost_path_to_here, exp_settings)

	prob_val = decimal.Decimal(prob_val)
	if prob_val.is_nan():
		prob_val = decimal.Decimal(1.0)

	return prob_val

	# decimal.getcontext().prec = 60
	# is_og = exp_settings['prob_og']

	# start = r.get_start()

	# if is_og:
	# 	c1 = decimal.Decimal(cost_path_to_here)
	# else:
	# 	c1 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(r.get_start()), resto.to_xy(pt), exp_settings))	

	
	# c2 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(pt), resto.to_xy(goal), exp_settings))
	# c3 = decimal.Decimal(get_min_direct_path_cost_between(r, resto.to_xy(start), resto.to_xy(goal), exp_settings))

	# # print(c2)
	# # print(c3)
	# a = np.exp((-c1 + -c2))
	# b = np.exp(-c3)
	# # print(a)
	# # print(b)

	# ratio 		= a / b

	# if math.isnan(ratio):
	# 	ratio = 0

	# return ratio



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

# Ada: Final equation
# TODO Cache this result for a given path so far and set of goals
def prob_array_goal_given_signals(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings):
	val_0, val_1 = 0.0, 0.0

	# only add the value to the array if it's going to be relevant
	if exp_settings[LEGIBILITY_METHOD] in get_set_legibility_method_uses_dist():
		val_0 = prob_goal_given_path(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings, unnormalized_prob_goal_given_path)
	if exp_settings[LEGIBILITY_METHOD] in get_set_legibility_method_uses_heading():
		val_1 = prob_goal_given_heading(r, p_n1, pt, goal, goals, cost_path_to_here, exp_settings, unnormalized_prob_goal_given_path_use_heading)

	return [val_0, val_1]


def prob_goal_given_heading(start, pn, pt, goal, goals, cost_path_to_here, exp_settings):
	g_probs = prob_goals_given_heading(pn, pt, goals, exp_settings)
	g_index = goals.index(goal)

	return g_probs[g_index]


def f_angle_prob(heading, goal_theta, exp_settings):
	diff = np.abs(1.0 / (heading - goal_theta))

	if exp_settings[LEGIBILITY_METHOD] == F_JHEADING_QUADRATIC:
		return diff * diff

	if exp_settings[LEGIBILITY_METHOD] == F_JHEADING_EXPONENTIAL:
		return np.exp(diff)

	if exp_settings[LEGIBILITY_METHOD] in [F_JDIST]:
		print("ERR, wrong legibility function consulting with angle probability function")
		return 0

	return diff


def prob_goals_given_heading(p0, p1, goals, exp_settings):
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

		goal_theta = resto.angle_between(p1, goal[:2])
		prob = f_angle_prob(heading, goal_theta, exp_settings)
		probs.append(prob)


	divisor = sum(probs)
	# divisor = 1.0

	return decimal.Decimal(np.true_divide(probs, divisor))
	# return ratio


def get_costs_along_path(path):
	output = []
	ci = 0
	csf = 0
	for pi in range(len(path)):
		# print(pi, ci)
		# print(path[ci], path[pi])
		
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

def get_dist(p0, p1):
	p0_x, p0_y = p0
	p1_x, p1_y = p1

	min_distance = np.sqrt((p0_x-p1_x)**2 + (p0_y-p1_y)**2)
	return min_distance

def get_min_direct_path_cost_angle_between(r, p0, p1, exp_settings):
	# TODO CURRENT ADA

	cost = (num_chunks * cost_chunk) + (leftover*leftover)

	return cost
	
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
	
	legibility = decimal.Decimal(0)
	divisor = decimal.Decimal(0)
	total_dist = decimal.Decimal(0)

	if 'lambda' in exp_settings and exp_settings['lambda'] != '':
		LAMBDA = decimal.Decimal(exp_settings['lambda'])
		epsilon = decimal.Decimal(exp_settings['epsilon'])
	else:
		# TODO verify this
		LAMBDA = 1.0
		epsilon = 1.0

	start = path[0]
	total_cost = decimal.Decimal(0)
	aug_path = get_costs_along_path(path)

	path_length_list, length_of_total_path = get_path_length(path)
	length_of_total_path = decimal.Decimal(length_of_total_path)

	delta_x = decimal.Decimal(1.0) #length_of_total_path / len(aug_path)

	t = 1
	p_n = path[0]
	divisor = epsilon
	numerator = decimal.Decimal(0.0)

	f_log = []
	p_log = []
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))

		# Get this probability from all the available signals
		probs_array_goal_given_signals = prob_array_goal_given_signals(r, p_n, pt, goal, goals, cost_to_here, exp_settings)

		# combine them according to the exp settings
		prob_goal_signals_fused = prob_overall_fuse_signals(probs_array_goal_given_signals, r, p_n, pt, goal, goals, cost_to_here, exp_settings)


		# Then do the normal methods of combining them
		f_log.append(float(f))
		p_log.append(prob_goal_signals_fused)


		if len(aud) == 0: # FLAG_is_denominator or 
			numerator += (prob_goal_signals_fused * f) # * delta_x)
			divisor += f #* delta_x
		else:
			numerator += (prob_goal_signals_fused * f) # * delta_x)
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

# Old version used for RO-MAN paper 2022
def f_legibility_single_input(r, goal, goals, path, aud, f_function, exp_settings):

	if path is None or len(path) == 0:
		return 0
	
	legibility = decimal.Decimal(0)
	divisor = decimal.Decimal(0)
	total_dist = decimal.Decimal(0)

	if 'lambda' in exp_settings and exp_settings['lambda'] != '':
		LAMBDA = decimal.Decimal(exp_settings['lambda'])
		epsilon = decimal.Decimal(exp_settings['epsilon'])
	else:
		# TODO verify this
		LAMBDA = 1.0
		epsilon = 1.0

	start = path[0]
	total_cost = decimal.Decimal(0)
	aug_path = get_costs_along_path(path)

	path_length_list, length_of_total_path = get_path_length(path)
	length_of_total_path = decimal.Decimal(length_of_total_path)

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

		if len(aud) == 0: # FLAG_is_denominator or 
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

# Given a path, count how long it's in sight
def f_env(r, goal, goals, path, aud, f_function, exp_settings):
	fov = exp_settings['fov']
	FLAG_is_denominator = exp_settings['is_denominator']
	if path is None or len(path) == 0:
		return 0, 0, 0

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

	env_readiness = -1
	t = 1
	p_n = path[0]
	for pt, cost_to_here in aug_path:
		f = decimal.Decimal(f_function(t, pt, aud, path))
	
		# if f is greater than 0, this indicates being in-view
		if f > vis_cutoff:
			count += 1
			if env_readiness == -1:
				env_readiness = (len(aug_path) - t + 1)

		# if it's not at least 0, then out of sight, not part of calc
		else:
			count = 0.0

		t += 1

	return count, env_readiness, len(aug_path)


def get_costs(path, target, obs_sets):
	vals = []

	for aud in obs_sets:
		new_val = f_cost()

	return vals


def angle_between_points(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	angle = np.arctan2(y2 - y1, x2 - x1)

	# ang1 = np.arctan2(*p1[::-1])
	# ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg(angle)

def angle_between_lines(l1, l2):
	p1a, p1b = l1
	p2a, p2b = l2

	a1 = angle_between_points(p1a, p1b)
	a2 = angle_between_points(p2a, p2b)
	angle = (a1 - a2)

	return angle

def angle_of_turn(l1, l2):
	return (angle_between_lines(l1, l2))

# TODO ada update
def inspect_legibility_of_paths(options, restaurant, exp_settings, fn):
	# options = options[0]
	print("Inspecting overall legibility")

	for pkey in options.keys():
		print(pkey)
		path = options[pkey]
		# print('saving fig')


		t = range(len(path))
		v = get_legib_graph_info(path, restaurant, exp_settings)
		# vo, va, vb, vm = v

		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		
		for key in v.keys():
			print("key combo is")
			print(key)
			# print(len(t))
			print(len(v[key]))
			t = range(len(v[key]))

			ax1.scatter(t, v[key], s=10, marker="o", label=key)

		# ax1.scatter(t, va, s=10, c='b', marker="o", label="Vis A")
		# ax1.scatter(t, vb, s=10, c='y', marker="o", label="Vis B")
		# ax1.scatter(t, vm, s=10, c='g', marker="o", label="Vis Multi")

		pkey_label = str(pkey)
		pkey_label.replace(" ", "")
		ax1.set_title('legibility of ' + str(pkey_label))
		# plt.get_legend().remove()
		# plt.legend(loc='upper left');
		
		plt.savefig(fn + "-" + pkey_label + '-legib' + '.png')
		plt.clf()
			
def get_legib_graph_info(path, restaurant, exp_settings):
	vals_dict = {}

	obs_sets = restaurant.get_obs_sets()

	costs_along = get_costs_along_path(path)

	for aud_i in obs_sets.keys():
		for goal in restaurant.get_goals_all():
			vals = []
			for t in range(1, len(path)):
				# with reference to which goal?
				cost_path_to_here = costs_along[t]
				# goal, goals, path, df_obs
				# new_val = legib.f_legibility(resto, target, goals, path, [], legib.f_naked, exp_settings)
				new_val = prob_goal_given_path(restaurant, path[t - 1], path[t], goal, restaurant.get_goals_all(), cost_path_to_here, exp_settings)
				# new_val = f_legibi(t, path[t], obs_sets[aud_i], path)
				# print(new_val)
				# exit()

				vals.append(new_val)

			vals_dict[aud_i, goal] = vals

	return vals_dict
	# return vo, va, vb, vm


def get_set_legibility_method_uses_heading():
	return [F_JHEADING, F_JHEADING_QUADRATIC, F_JHEADING_EXPONENTIAL]

def get_set_legibility_method_uses_dist():
	return [F_JDIST]


def get_legibility_options():
	options = [F_JDIST, F_JHEADING_EXPONENTIAL, F_JHEADING, F_JHEADING_QUADRATIC] #F_JDIST

	return options

def lookup_legibility_label_of(f):
	if f == unnormalized_prob_goal_given_path:
		return F_JDIST
	elif f == unnormalized_prob_goal_given_path_use_heading:
		return F_JHEADING

	return "LABELERR"

def lookup_legibility_unnormalized_function(exp_settings):
	l = exp_settings[LEGIBILITY_METHOD] # l_method defined in pipeline_generate_path

	if l in [F_JDIST]:
		return unnormalized_prob_goal_given_path
	elif l in [F_JHEADING, F_JHEADING_QUADRATIC, F_JHEADING_EXPONENTIAL]:
		return unnormalized_prob_goal_given_path_use_heading

	print("ERR, label not found")
	return "LABELERR"



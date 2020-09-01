import table_path_code as resto
import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import cv2


# start 		= r.get_start()
# goals 		= r.get_goals_all()
# goal 		= r.get_current_goal()
# observers 	= r.get_observers()
# tables 		= r.get_tables()
# waypoints 	= r.get_waypoints()
# SCENARIO_IDENTIFIER = r.get_scenario_identifier()

FLAG_SAVE 			= True
FLAG_VIS_GRID 		= False
VISIBILITY_TYPES 	= resto.VIS_CHECKLIST
NUM_CONTROL_PTS 	= 3

PATH_TIMESTEPS = 15

resto_pickle = 'pickle_vis'
vis_pickle = 'pickle_resto'
FILENAME_PATH_ASSESS = 'path_assessment/'


def f_cost(t1, t2):
	return resto.dist(t1, t2)

def f_cost(t1, t2):
	return resto.dist(t1, t2)

def f_path_cost(path):
	cost = 0
	for i in range(len(path) - 1):
		cost = cost + f_cost(path[i], path[i + 1])

	return cost


def f_audience_agnostic():
	return f_leg_personalized()
	pass

def f_leg_personalized():

	pass


# Given the observers of a given location, in terms of distance and relative heading
def f_visibility(df_obs):
	dist_units = 100
	angle_cone = 400
	distance_cutoff = 500

	# Given a list of entries in the format 
	# ((obsx, obsy), angle, distance)
	if len(df_obs) == 0:
		return 0
	
	vis = 0
	for obs in df_obs:	
		pt, angle, dist = obs
		if angle < angle_cone and dist < distance_cutoff:
			vis += (distance_cutoff - dist)

	return vis

def f(t):
	return PATH_TIMESTEPS - t

	# return (PATH_TIMESTEPS - f(t)) * visibility(pt(t))
	# non-vanilla version


def prob_goal_given_path(start, pt, goal, goals):
	c1 = f_cost(start, pt)
	c2 = f_cost(pt, goal)
	c3 = f_cost(start, goal)

	return np.exp(-c1 -c2) / np.exp(c3)

# Given a 
def f_legibility(goal, goals, path, df_obs):
	legibility = 0
	divisor = 0
	total_dist = 0
	LAMBDA = 1

	start = path[0]
	total_cost = 0

	t = 1
	p_n = path[0]
	for pt in path:
		legibility += prob_goal_given_path(start, pt, goal, goals) * f(t)
		
		total_cost += f_cost(p_n, pt)
		p_n = pt

		divisor += f(t)
		t = t + 1
	
	overall = (legibility / divisor) - LAMBDA*total_cost
	return overall

def get_costs(path, target, obs_sets):
	vals = []

	for aud in obs_sets:
		new_val = f_cost()

	return vals


def get_legibilities(path, target, goals, obs_sets):
	vals = []

	for aud in obs_sets:
		# goal, goals, path, df_obs
		new_val = f_legibility(target, goals, path, aud)
		vals.append(new_val)

	return vals


def generate_visibility(df, vis_function):
	pass


def generate_single_path(restaurant, target, vis_type, n_control):
	sample_pts 	= r.sample_points(n_control, target, vis_type)
	path 		= construct_single_path(r.get_start(), target, sample_pts)
	return path

def construct_single_path(start, end, sample_pts):
	STEPSIZE = 15
	points = []
	
	xys = [start] + sample_pts + [end]

	ts = [t/STEPSIZE for t in range(STEPSIZE + 1)]
	bezier = resto.make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points

def generate_n_paths(restaurant, num_paths, target, n_control):
	path_list = []
	vis_type = resto.VIS_OMNI
	for i in range(num_paths):
		path_option = generate_single_path(restaurant, target, vis_type, n_control)
		path_list.append(path_option)

	return path_list

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

def get_path_analysis(all_paths, r, target):	
	obs_none 	= []
	obs_a 		= [r.get_observer_back()]
	obs_b 		= [r.get_observer_towards()]
	obs_multi 	= [r.get_observer_back(), r.get_observer_towards()]

	obs_sets = [obs_none, obs_a, obs_b, obs_multi]

	goals = r.get_goals_all()
	col_labels = ['path', 'cost', 'l_agnostic', 'l_a', 'l_b', 'l_multi', 'target']

	data = []
	for p in all_paths:
		cost = f_path_cost(p)
		l_o, l_a, l_b, l_m = get_legibilities(p, target, goals, obs_sets)

		entry = [p, cost, l_o, l_a, l_b, l_m, target]
		data.append(entry)

	df = pd.DataFrame(data, columns = col_labels) 

	return df

def assess_paths(all_paths, r, target):
	df = get_path_analysis(all_paths, r, target)

	return df

def iterate_on_paths():
	path_options 		= generate_paths(NUM_PATHS, r, VISIBILITY_TYPES)
	path_assessments 	= assess_paths(path_options)


def inspect_heatmap(df):
	# print(df)

	length 		= df['x'].max()
	width 		= df['y'].max()
	max_multi 	= df['VIS_MULTI'].max()
	max_a 		= df['VIS_A'].max()
	max_b 		= df['VIS_B'].max()
	max_omni 	= df['VIS_OMNI'].max()

	print((length, width))
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

NUM_PATHS = 200
generate_type = resto.TYPE_UNITY_ALIGNED

# SETUP FROM SCRATCH AND SAVE
if FLAG_SAVE:
	r 	= resto.Restaurant(generate_type)
	# vis = r.get_visibility_of_pts_pandas()
	print("PLANNER: get visibility info")

	if FLAG_VIS_GRID:
		df_vis = r.get_visibility_of_pts_pandas(f_visibility)

		dbfile = open(vis_pickle, 'ab') 
		pickle.dump(df_vis, dbfile)					  
		dbfile.close()
		print("Saved visibility map")
		df_vis.to_csv('visibility.csv')
		print("Visibility point grid created")
	
	dbfile = open(resto_pickle, 'ab') 
	pickle.dump(r, dbfile)					  
	dbfile.close()
	print("Saved restaurant maps")

# OR LOAD FROM FILE
else:
	dbfile = open(resto_pickle, 'rb')
	r = pickle.load(dbfile)
	print("Imported pickle of restaurant")

	if FLAG_VIS_GRID:
		dbfile = open(vis_pickle, 'rb')
		df_vis = pickle.load(dbfile)
		print("Imported pickle of vis")


img = r.get_img()
empty_img = cv2.flip(img, 0)
cv2.imwrite(FILENAME_PATH_ASSESS + 'empty.png', empty_img) 
goals = r.get_goals_all()

# Decide how many control points to provide
for ti in range(len(goals)):
	all_paths = []
	target = goals[ti]
	for n_control in range(3):

		paths = generate_n_paths(r, 20, target, n_control)
		fn = FILENAME_PATH_ASSESS + str(ti) + "-" + str(n_control) + "-cp" + "-all.png"
		resto.export_raw_paths(img, paths, fn)
		all_paths.extend(paths)

	options = assess_paths(all_paths, r, target)
	print(options)
	# options.













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

print("Done")
import table_path_code as resto
import pandas as pd
import pickle
import seaborn as sns


# start 		= r.get_start()
# goals 		= r.get_goals_all()
# goal 		= r.get_current_goal()
# observers 	= r.get_observers()
# tables 		= r.get_tables()
# waypoints 	= r.get_waypoints()
# SCENARIO_IDENTIFIER = r.get_scenario_identifier()

FLAG_SAVE = False
VISIBILITY_TYPES 	= resto.VIS_CHECKLIST
NUM_CONTROL_PTS 	= 3

resto_pickle = 'pickle_vis'
vis_pickle = 'pickle_resto'

def f_cost(df_row):

	return 0

def f_visibility(df_obs):
	dist_units = 100
	angle_cone = 60
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

def f_legibility(df_row):

	return 0

def generate_visibility(df, vis_function):
	pass


def create_path_options():
	path_list = []
	for i in range(num_paths):
		pts 	= r.sample_points()
		path 	= make_bezier(pts)

	return path_list

def generate_paths(num_paths, restaurant, vis_types):
	path_options = {}
	for vis_type in vis_types:
		path_options[vis_type] = create_path_options(num_paths, restaurant, vis_type)
	return path_options

def assess_paths():
	pass


def iterate_on_paths():
	path_options 		= generate_paths(NUM_PATHS, r, VISIBILITY_TYPES)
	path_assessments 	= assess_paths(path_options)


# df.at[i,COL_PATHING] = get_pm_label(row)

NUM_PATHS = 200

generate_type = resto.TYPE_UNITY_ALIGNED

if FLAG_SAVE:
	r 	= resto.Restaurant(generate_type)
	# vis = r.get_visibility_of_pts_pandas()
	print("PLANNER: get visibility info")
	df_vis = r.get_visibility_of_pts_pandas(f_visibility)

	dbfile = open(vis_pickle, 'ab') 
	pickle.dump(df_vis, dbfile)					  
	dbfile.close()
	print("Saved visibility map")

	dbfile = open(resto_pickle, 'ab') 
	pickle.dump(r, dbfile)					  
	dbfile.close()
	print("Saved restaurant maps")

	print("Visibility point grid created")
	df_vis.to_csv('visibility.csv')



dbfile = open(resto_pickle, 'rb')
r = pickle.load(dbfile)
print("Imported pickle of restaurant")

dbfile = open(vis_pickle, 'rb')
df_vis = pickle.load(dbfile)
print("Imported pickle of vis")

result = df_vis.pivot(index='x', columns='y', values=resto.VIS_MULTI)
print("pivoted")
heatmap = sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
print("made heatmap")

fig = heatmap.get_figure()
fig.savefig("multi-vis.png")
print("Graphs")



# resto.draw_paths(r, paths_dict)
# resto.export(r, paths_dict)

print("Done")
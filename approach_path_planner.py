import table_path_code

# start 		= r.get_start()
# goals 		= r.get_goals_all()
# goal 		= r.get_current_goal()
# observers 	= r.get_observers()
# tables 		= r.get_tables()
# waypoints 	= r.get_waypoints()
# SCENARIO_IDENTIFIER = r.get_scenario_identifier()

VISIBILITY_TYPES = VIS_CHECKLIST

def generate_paths(num_paths, r):
	pass


def assess_paths():
	pass


NUM_PATHS = 200

generate_type = TYPE_UNITY_ALIGNED
r 	= Restaurant(generate_type)
vis = r.get_visibilities()
v 	= r.get_visibility_maps()

path_options 		= generate_paths()
path_assessments 	= assess_paths()

draw_paths(r, paths_dict)
export(r, paths_dict)

print("Done")
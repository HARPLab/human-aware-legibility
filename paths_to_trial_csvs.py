### ABC
### DEF

# ab = cb horizontal flip
# cd = fd horizontal flip

goal_a  = [1.0, -1.0]
goal_b  = [3.0, -1.0] 
goal_c  = [5.0, -1.0]

goal_d  = [1.0, -3.0]
goal_e  = [3.0, -3.0]
goal_f  = [5.0, -3.0]

state_dict = {}
state_dict['A'] = goal_a
state_dict['B'] = goal_b
state_dict['C'] = goal_c
state_dict['D'] = goal_d
state_dict['E'] = goal_e
state_dict['F'] = goal_f

# GENERATED PATHS
# generate_vanilla_straight_line_paths_for_testing(goal_a, [goal_b, goal_c, goal_d, goal_e, goal_f])
path_ab = [[1.0, -1.0], [1.25, -1.0], [1.5, -1.0], [1.75, -1.0], [2.0, -1.0], [2.25, -1.0], [2.5, -1.0], [2.75, -1.0], [3.0, -1.0]]
path_ac = [[1.0, -1.0], [1.5, -1.0], [2.0, -1.0], [2.5, -1.0], [3.0, -1.0], [3.5, -1.0], [4.0, -1.0], [4.5, -1.0], [5.0, -1.0]]
path_ad = [[1.0, -1.0], [1.0, -1.25], [1.0, -1.5], [1.0, -1.75], [1.0, -2.0], [1.0, -2.25], [1.0, -2.5], [1.0, -2.75], [1.0, -3.0]]
path_ae = [[1.0, -1.0], [1.25, -1.25], [1.5, -1.5], [1.75, -1.75], [2.0, -2.0], [2.25, -2.25], [2.5, -2.5], [2.75, -2.75], [3.0, -3.0]]
path_af = [[1.0, -1.0], [1.5, -1.25], [2.0, -1.5], [2.5, -1.75], [3.0, -2.0], [3.5, -2.25], [4.0, -2.5], [4.5, -2.75], [5.0, -3.0]]

# generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
path_ba = [[3.0, -1.0], [2.75, -1.0], [2.5, -1.0], [2.25, -1.0], [2.0, -1.0], [1.75, -1.0], [1.5, -1.0], [1.25, -1.0], [1.0, -1.0]]
path_bc = [[3.0, -1.0], [3.25, -1.0], [3.5, -1.0], [3.75, -1.0], [4.0, -1.0], [4.25, -1.0], [4.5, -1.0], [4.75, -1.0], [5.0, -1.0]]
path_bd = [[3.0, -1.0], [2.75, -1.25], [2.5, -1.5], [2.25, -1.75], [2.0, -2.0], [1.75, -2.25], [1.5, -2.5], [1.25, -2.75], [1.0, -3.0]]
path_be = [[3.0, -1.0], [3.0, -1.25], [3.0, -1.5], [3.0, -1.75], [3.0, -2.0], [3.0, -2.25], [3.0, -2.5], [3.0, -2.75], [3.0, -3.0]]
path_bf = [[3.0, -1.0], [3.25, -1.25], [3.5, -1.5], [3.75, -1.75], [4.0, -2.0], [4.25, -2.25], [4.5, -2.5], [4.75, -2.75], [5.0, -3.0]]

def get_all_possible_path_names():
	all_paths = []
	targets = ['A', 'B', 'C', 'D', 'E', 'F']

	for i in targets:
		for j in targets:
			link = i + j

			if link not in all_paths and i != j:
				all_paths.append(link)

	return all_paths

def generate_vanilla_straight_line_paths_for_testing(start, goal_list):
	N = 8
	for end_state in goal_list:
		start_state = start
		crow_flies_vector = [end_state[0] - start_state[0], end_state[1] - start_state[1]]
		step_vector = [1.0 * crow_flies_vector[0] / N, 1.0 * crow_flies_vector[1] / N]

		path = [start_state]
		print("~~")
		prev_pt = path[0]
		for i in range(N):
			pt = [prev_pt[0] + step_vector[0], prev_pt[1] + step_vector[1]]
			path.append(pt)
			prev_pt = pt

		print(path)


def horizontal_flip(path):
	center_horiz = goal_b[0]

	new_path = []
	for p in path:
		offset = (center_horiz - p[0])
		new_x = center_horiz + (offset)

		new_p = [new_x, p[1]]
		new_path.append(new_p)


	return new_path


def vertical_flip(path):
	center_horiz = (goal_a[1] + goal_d[1]) / 2.0

	new_path = []
	for p in path:
		offset = (center_horiz - p[1])
		new_y = center_horiz + (offset)
		
		new_p = [p[0], new_y]
		new_path.append(new_p)


	return new_path


# generate_vanilla_straight_line_paths_for_testing(goal_a, [goal_b, goal_c, goal_d, goal_e, goal_f])
# generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])

def setup_path_dict():
	all_paths = get_all_possible_path_names()
	# print(all_paths)
	path_dict = {}
	for name in all_paths:
		path_dict[name] = None

	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ac
	path_dict['AD'] = path_ad
	path_dict['AE'] = path_ae
	path_dict['AF'] = path_af

	path_dict['BA'] = path_ba
	path_dict['BC'] = path_bc
	path_dict['BD'] = path_bd
	path_dict['BE'] = path_be
	path_dict['BF'] = path_bf

	path_dict['CB'] = horizontal_flip(path_ab)
	path_dict['CA'] = horizontal_flip(path_ac)
	path_dict['EA'] = horizontal_flip(path_ae)
	path_dict['CD'] = horizontal_flip(path_af)
	path_dict['CE'] = horizontal_flip(path_ae)
	path_dict['CF'] = horizontal_flip(path_ad)


	path_dict['DA'] = vertical_flip(path_ad)
	path_dict['DB'] = vertical_flip(path_ae)
	path_dict['DC'] = vertical_flip(path_af)
	path_dict['DE'] = vertical_flip(path_ab)
	path_dict['DF'] = vertical_flip(path_ac)

	path_dict['EA'] = vertical_flip(path_bd)
	path_dict['EB'] = vertical_flip(path_be)
	path_dict['EC'] = vertical_flip(path_bf)
	path_dict['ED'] = vertical_flip(path_ba)
	path_dict['EF'] = vertical_flip(path_bc)

	path_dict['FA'] = horizontal_flip(vertical_flip(path_af))
	path_dict['FB'] = horizontal_flip(vertical_flip(path_ae))
	path_dict['FC'] = horizontal_flip(vertical_flip(path_ad))
	path_dict['FD'] = horizontal_flip(vertical_flip(path_ac))
	path_dict['FE'] = horizontal_flip(vertical_flip(path_ab))
	
	### VERIFY THAT ALL PATHS ARE COVERED
	todo = []
	for key in path_dict.keys():
		if path_dict[key] == None:
			todo.append(key)
	# print(todo)

	is_problem = False
	#### VERIFY ALL HAVE CORRECT START AND END
	for key in path_dict.keys():
		path = path_dict[key]

		start 	= state_dict[key[0]]
		end 	= state_dict[key[1]]

		if path[0] != start:
			print("Broken in " + key + " bad start")
			is_problem = True

		if path[-1] != end:
			print("Broken in " + key + " bad end")
			is_problem = True

	if is_problem:
		print("Problem in path transformations")
	else:
		print("All paths added and checked for reasonableness!")

setup_path_dict()
# augment_to_find_all_paths()





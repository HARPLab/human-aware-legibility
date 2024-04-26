import random
import copy 
import numpy as np

### A B C
### D E F

targets = ['A', 'B', 'C', 'D', 'E', 'F']
participant = 'E'

states = targets


# name the routes for easier understanding
titles = {}
titles['AB'] = 'back_left'
titles['BA'] = 'back_left'
titles['BC'] = 'back_right'
titles['CB'] = 'back_right'

titles['DE'] = 'front_left'
titles['ED'] = 'front_left'
titles['EF'] = 'front_right'
titles['FD'] = 'front_right'

titles['AC'] = 'back_full'
titles['CA'] = 'back_full'
titles['DF'] = 'front_full'
titles['FD'] = 'front_full'

titles['DC'] = 'diag_full'
titles['CD'] = 'diag_full'
titles['AF'] = 'diag_full'
titles['FA'] = 'diag_full'

titles['AE'] = 'diag_short'
titles['EA'] = 'diag_short'
titles['CE'] = 'diag_short'
titles['EC'] = 'diag_short'

titles['BE'] = 'strt_mid_away'
titles['EB'] = 'strt_mid_to'

titles['AD'] = 'strt_left_to'
titles['DA'] = 'strt_left_away'
titles['CF'] = 'strt_right_to'
titles['FC'] = 'strt_right_away'

# AC-early-obs.csv
# AF-late-obs.csv
# AF-even-obs.csv
# AF-early-obs.csv
# AC-late-obs.csv
# AC-even-obs.csv

label_dict = {}
for key in titles.keys():
	name_label 				= titles[key]
	label_dict[name_label] 	= key



# Show given paths in shuffled order overall
def get_hitlist():
	hitlist = []

	# just the ones coming to me
	# crucial paths
	to_target = ['DE', 'FE']
	hitlist.extend(to_target)

	past_front = ['DF', 'FD']
	hitlist.extend(past_front)

	past_back = ['AC', 'CA']
	hitlist.extend(past_back)

	more = ['AB', 'AD']
	hitlist.extend(more)

	# Also good
	diag_long = ['AF', 'DC']
	hitlist.extend(to_target)

	diag_long = ['AC', 'CA', '']
	hitlist.extend(to_target)

	diag_long = ['AF', 'DC']
	hitlist.extend(to_target)

	# past_back = ['AD', 'DA']
	# hitlist.extend(to_target)

	# to_hit = diag_full
	full_hitlist = []

	for h in hitlist:
		full_hitlist.append(h + "-HI")
		full_hitlist.append(h + "-MID")
		full_hitlist.append(h + "-LOW")


	return hitlist, full_hitlist


def dist_between(x1, x2):
	distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
	return distance

# def get_total_path_time(path):
# 	total_time = 0
# 	resolution = 1 / .2

# 	square_size = 2
# 	for segment in path:

# 		##### SHORT EDGES
# 		if dist_between(path[0], path[-1]) == dist_between(goal_a, goal_b):
# 			total_time += resolution * square_size

# 		##### LONG EDGES
# 		if dist_between(path[0], path[-1]) == dist_between(goal_a, goal_c):
# 			total_time += (resolution * square_size * 2)

# 		##### SHORT DIAG
# 		if dist_between(path[0], path[-1]) == dist_between(goal_a, goal_e):
# 			total_time += (resolution * square_size * np.sqrt(2))

# 		##### LONG DIAG
# 		if dist_between(path[0], path[-1]) == dist_between(goal_a, goal_f):
# 			total_time += (resolution * square_size * np.sqrt(3))

# 	return total_time



all_paths = []

for i in targets:
	for j in targets:
		link = i + j

		if link not in all_paths and i != j:
			all_paths.append(link)

print("All possible paths")
print(all_paths)


checklist, full_hitlist = get_hitlist()

checklist = all_paths

print()
print("Paths to hit")
print(checklist)

# print(full_hitlist)

start = random.choice([y for y in targets if y != participant])
print("start at: ")
print(start)

path = []
checked_off = []

current_state = start
remaining = copy.copy(checklist)

i = 0
while len(remaining) > 0 and i < 100:
	i += 1

	# Could go to any other state
	next_state_options =  [y for y in states if y != current_state]

	# Or, could complete a link that's not checked off yet
	useful_options = [y[1] for y in remaining if y[0] == current_state]
	
	# OR could move to a place where I can check something off next
	setup_for_useful_options = [y[0] for y in remaining if y[0] != current_state]

	if len(useful_options) > 0:
		next_state = random.choice(useful_options)
	elif len(setup_for_useful_options) > 0:
		next_state = random.choice(setup_for_useful_options)
	else:
		next_state = random.choice(next_state_options)

	link_name = current_state + next_state
	path.append(link_name)

	if link_name in remaining:
		remaining.remove(link_name)

	if link_name in checklist and link_name not in checked_off:
		checked_off.append(link_name)


	current_state = next_state

print()
print(path)
print(len(path))
print(len(remaining))
print(remaining)

final_route = []

for p in path:
	final_route.append(p + '-even')
	# final_route.append(p + '-toptwo')

print(final_route)

# print("estimated total path time")
# print(get_total_path_time(path))

# saved = ['AC', 'CD', 'DF', 'FD', 'DC', 'CE', 'EF', 'FC', 'CD', 'DE', 'ED', 'DE', 'EF', 'FA', 'AB', 'BE', 'EA', 'AB', 'BA', 'AE', 'ED', 'DE', 'ED', 'DC', 'CF', 'FD', 'DC', 'CB', 'BE', 'EA', 'AE', 'EC', 'CF', 'FA', 'AF', 'FE']

#### CALCULATE THE PROJECTED TIME ALSO





























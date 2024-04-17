import random
import copy 
import numpy as np

### A B C
### D E F

goal_points = ['A', 'B', 'C', 'D', 'E', 'F']
participant = 'E'

states = goal_points


FRONT_SHORT_TO 		= 'front_short_to'
BACK_SHORT_TO 		= 'back_short_to'
FRONT_SHORT_FROM 	= 'front_short_from'
BACK_SHORT_FROM 	= 'back_short_from'

SIDE_SHORT_TO			= 'side_short_to'
SIDE_SHORT_FROM			= 'side_short_from'
BACK_LONG 				= 'back_long'
FRONT_LONG 				= 'front_long'
DIAG_LONG_TO 			= 'diag_long_to'
DIAG_LONG_FROM 			= 'diag_long_from'
DIAG_SHORT_TO 			= 'diag_short_to'
DIAG_SHORT_FROM 		= 'diag_short_from'
MID_SHORT 				= 'mid_short'
BACK_DIAG_SHORT_TO		= 'back_diag_short_to'
BACK_DIAG_SHORT_FROM	= 'back_diag_short_from'

BACK_LONG_OBS 				= 'back_long_obs'
FRONT_LONG_OBS 				= 'front_long_obs'
DIAG_OBS_LONG_TO			= 'diag_obs_long_to'
DIAG_OBS_LONG_FROM			= 'diag_obs_long_from'

targets = [FRONT_SHORT_TO, BACK_SHORT_TO, SIDE_SHORT_TO, SIDE_SHORT_FROM, BACK_LONG, FRONT_LONG, DIAG_LONG_TO, DIAG_LONG_FROM, DIAG_SHORT_TO]

all_options = [FRONT_SHORT_TO, BACK_SHORT_TO, FRONT_SHORT_FROM, BACK_SHORT_FROM, SIDE_SHORT_TO, SIDE_SHORT_FROM, BACK_LONG, FRONT_LONG, DIAG_LONG_TO, DIAG_LONG_FROM, DIAG_SHORT_TO, DIAG_SHORT_FROM, MID_SHORT, BACK_DIAG_SHORT_TO, BACK_DIAG_SHORT_FROM, BACK_LONG_OBS, FRONT_LONG_OBS, DIAG_OBS_LONG_TO, DIAG_OBS_LONG_FROM]

# name the routes for easier understanding
titles = {}
titles['AB'] = BACK_SHORT_TO
titles['BA'] = BACK_SHORT_FROM
titles['BC'] = BACK_SHORT_FROM
titles['CB'] = BACK_SHORT_TO

titles['DE'] = FRONT_SHORT_TO
titles['ED'] = FRONT_SHORT_FROM
titles['EF'] = FRONT_SHORT_FROM
titles['FE'] = FRONT_SHORT_TO

titles['AC'] = BACK_LONG
titles['CA'] = BACK_LONG
titles['DF'] = FRONT_LONG
titles['FD'] = FRONT_LONG

titles['DC'] = DIAG_LONG_TO
titles['CD'] = DIAG_LONG_FROM
titles['AF'] = DIAG_LONG_TO
titles['FA'] = DIAG_LONG_FROM

titles['AE'] = DIAG_SHORT_TO
titles['EA'] = DIAG_SHORT_FROM
titles['CE'] = DIAG_SHORT_TO
titles['EC'] = DIAG_SHORT_FROM

titles['BE'] = MID_SHORT
titles['EB'] = MID_SHORT

titles['AD'] = SIDE_SHORT_TO
titles['DA'] = SIDE_SHORT_FROM
titles['CF'] = SIDE_SHORT_TO
titles['FC'] = SIDE_SHORT_FROM

titles['BD'] = BACK_DIAG_SHORT_FROM
titles['BF'] = BACK_DIAG_SHORT_FROM
titles['DB'] = BACK_DIAG_SHORT_TO
titles['FB'] = BACK_DIAG_SHORT_TO

titles['DC_OBS'] = DIAG_OBS_LONG_TO
titles['CD_OBS'] = DIAG_OBS_LONG_FROM
titles['AF_OBS'] = DIAG_OBS_LONG_TO
titles['FA_OBS'] = DIAG_OBS_LONG_FROM

titles['AC_OBS'] = BACK_LONG_OBS
titles['CA_OBS'] = BACK_LONG_OBS
titles['DF_OBS'] = FRONT_LONG_OBS
titles['FD_OBS'] = FRONT_LONG_OBS

# AC-early-obs.csv
# AF-late-obs.csv
# AF-even-obs.csv
# AF-early-obs.csv
# AC-late-obs.csv
# AC-even-obs.csv

label_dict = {}
for key in titles.keys():
	name_label 				= titles[key]
	label_dict[name_label] 	= []

for key in titles.keys():
	name_label 				= titles[key]
	label_dict[name_label].append(key)

# Show given paths in shuffled order overall
def get_hitlist():
	hitlist = []
	hitlist.append(SIDE_SHORT_TO)
	hitlist.append(SIDE_SHORT_FROM)
	hitlist.append(FRONT_SHORT_TO)
	hitlist.append(BACK_SHORT_TO)
	hitlist.append(FRONT_LONG)
	hitlist.append(BACK_LONG)

	hitlist.append(DIAG_OBS_LONG_TO)
	hitlist.append(DIAG_OBS_LONG_FROM)
	hitlist.append(BACK_LONG_OBS)
	hitlist.append(FRONT_LONG_OBS)



	# # just the ones coming to me
	# # crucial paths
	# to_target = ['DE', 'FE']
	# hitlist.extend(to_target)

	# past_front = ['DF', 'FD']
	# hitlist.extend(past_front)

	# past_back = ['AC', 'CA']
	# hitlist.extend(past_back)

	# more = ['AB', 'AD']
	# hitlist.extend(more)

	# # Also good
	# diag_long = ['AF', 'DC']
	# hitlist.extend(to_target)

	# diag_long = ['AC', 'CA', '']
	# hitlist.extend(to_target)

	# diag_long = ['AF', 'DC']
	# hitlist.extend(to_target)

	# # past_back = ['AD', 'DA']
	# # hitlist.extend(to_target)

	# to_hit = diag_full
	full_hitlist = []

	# for h in hitlist:
	# 	full_hitlist.append(h + "-HI")
	# 	full_hitlist.append(h + "-MID")
	# 	full_hitlist.append(h + "-LOW")


	return hitlist


def dist_between(x1, x2):
	distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
	return distance

all_paths = []

for i in goal_points:
	for j in goal_points:
		link = i + j

		if link not in all_paths and i != j:
			all_paths.append(link)

print("All possible paths")
print(all_paths)

checklist = get_hitlist()

print()
print("Paths to hit")
print(checklist)

# print(full_hitlist)


def get_setup_for_useful_options_after(current_state, remaining):

	# Make a lil histogram and go for the most likely option
	options = []

	best_setups = []
	# try to set up for elements still in the list
	for key in label_dict.keys():
		if key in remaining: # if it remains to be done
			for item in label_dict[key]:
				if item[0] != current_state:
					# print(item)
					# where would we like to be in order to follow the next best path?
					best_setups.append(current_state + item[0])

	# options = [y[0][1] for y in options if y[0] == current_state]

	# returns a list of connector options with a frequency 
	# related to how many list items it could possibly check off
	return best_setups


def get_useful_options_after(current_state, remaining):
	options = []

	for key in label_dict.keys():
		if key in remaining:
			for item in label_dict[key]:
				options.append((item, key))

	options = [y[0] for y in options if y[0][0] == current_state]

	return options


def get_path(start_link=None):
	if start_link == None:
		start = random.choice([y for y in goal_points if y != participant])
	else:
		start = start_link[1]

	print()
	print("start at: ")
	print(start)

	path = []
	checked_off = []

	current_state = start
	remaining = copy.copy(checklist)

	i = 0
	while len(remaining) > 0 and i < 30:
		i += 1

		# Could go to any other state
		next_state_options =  [y for y in states if y != current_state]

		# Or, could complete a link that's not checked off yet
		useful_options 		= get_useful_options_after(current_state, remaining)
		
		# OR could move to a place where I can check something off next
		setup_for_useful_options 	= get_setup_for_useful_options_after(current_state, remaining)

		# print("Useful next steps")
		# print(useful_options)
		# print("Setup steps")
		# print(setup_for_useful_options)

		if len(useful_options) > 0:
			next_link = random.choice(useful_options)
		elif len(setup_for_useful_options) > 0:
			next_link = random.choice(setup_for_useful_options)
		else:
			print("Eeek")
			exit()
			# next_link = random.choice(next_state_options)

		link_name = next_link
		path.append(link_name)

		checkoff_name = titles[link_name]

		if checkoff_name in remaining:
			remaining.remove(checkoff_name)
			# print("Completed: " + checkoff_name)

		if checkoff_name in checklist and checkoff_name not in checked_off:
			checked_off.append(checkoff_name)

		current_state = link_name[1]

		# print("remaining")
		# print(remaining)

		# print("check off")
		# print(checked_off)

	return path


def latin_squares_ify(path1, path2, path3):
	hitlist = get_hitlist()

	hitlist1 = {}
	hitlist2 = {}
	hitlist3 = {}

	for target in all_options:
		# Latin Squares is:
		# ABC 
		# BCA 
		# CAB

		options = [['early', 'even', 'late'], ['even', 'late', 'early'], ['late', 'early', 'even']]
		random.shuffle(options)

		hitlist1[target] = options[0]
		hitlist2[target] = options[1]
		hitlist3[target] = options[2]

	# for each relevant link, assign it a label per person
	# walk through the specific path and paint it at that point with the label

	### apply those labels to the relevant paths
	augpath1 = []
	augpath2 = []
	augpath3 = []

	for link in path1:
		checkoff_name = titles[link]

		if checkoff_name in hitlist:
			person1_exp = hitlist1[checkoff_name][0]
			person2_exp = hitlist2[checkoff_name][0]
			person3_exp = hitlist3[checkoff_name][0]
		else:
			person1_exp = 'null'
			person2_exp = 'null'
			person3_exp = 'null'

		augpath1.append(link + "-" + person1_exp)
		augpath2.append(link + "-" + person2_exp)
		augpath3.append(link + "-" + person3_exp)

	for link in path2:
		checkoff_name = titles[link]

		if checkoff_name in hitlist:
			person1_exp = hitlist1[checkoff_name][1]
			person2_exp = hitlist2[checkoff_name][1]
			person3_exp = hitlist3[checkoff_name][1]
		else:
			person1_exp = 'null'
			person2_exp = 'null'
			person3_exp = 'null'

		augpath1.append(link + "-" + person1_exp)
		augpath2.append(link + "-" + person2_exp)
		augpath3.append(link + "-" + person3_exp)

	for link in path3:
		checkoff_name = titles[link]

		if checkoff_name in hitlist:
			person1_exp = hitlist1[checkoff_name][2]
			person2_exp = hitlist2[checkoff_name][2]
			person3_exp = hitlist3[checkoff_name][2]
		else:
			person1_exp = 'null'
			person2_exp = 'null'
			person3_exp = 'null'

		augpath1.append(link + "-" + person1_exp)
		augpath2.append(link + "-" + person2_exp)
		augpath3.append(link + "-" + person3_exp)

	return augpath1, augpath2, augpath3

def setup_exp_set():
	path1 = get_path()
	path2 = get_path(start_link=path1[-1])
	path3 = get_path(start_link=path2[-1])

	print()
	print(path1)
	print(len(path1))
	print(path2)
	print(len(path2))
	print(path3)
	print(len(path3))

	exp1, exp2, exp3 = latin_squares_ify(path1, path2, path3)
	return exp1, exp2, exp3





# print("estimated total path time")
# print(get_total_path_time(path))

# saved = ['AC', 'CD', 'DF', 'FD', 'DC', 'CE', 'EF', 'FC', 'CD', 'DE', 'ED', 'DE', 'EF', 'FA', 'AB', 'BE', 'EA', 'AB', 'BA', 'AE', 'ED', 'DE', 'ED', 'DC', 'CF', 'FD', 'DC', 'CB', 'BE', 'EA', 'AE', 'EC', 'CF', 'FA', 'AF', 'FE']

### good ones
# C=11 ['CB', 'BD', 'DC_OBS', 'CA', 'AC_OBS', 'CD_OBS', 'DA', 'AD', 'DF', 'FD_OBS', 'DE']
# F=12 ['FD', 'DA', 'AB', 'BD', 'DC_OBS', 'CA', 'AC_OBS', 'CF', 'FA_OBS', 'AD', 'DF_OBS', 'FE']


exp1, exp2, exp3 = setup_exp_set()

print()
print("Final orderings")
print(exp1)
print("~~~")
print(exp2)
print("~~~")
print(exp3)
print("~~~")
























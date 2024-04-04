import random
import copy 

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

	# Also good
	diag_long = ['AF', 'DC']
	hitlist.extend(to_target)

	# past_back = ['AC', 'CA']
	# hitlist.extend(to_target)


	# to_hit = diag_full



	full_hitlist = []

	for h in hitlist:
		full_hitlist.append(h + "-HI")
		full_hitlist.append(h + "-MID")
		full_hitlist.append(h + "-LOW")


	return hitlist, full_hitlist


all_paths = []

for i in targets:
	for j in targets:
		link = i + j

		if link not in all_paths and i != j:
			all_paths.append(link)

print("All possible paths")
print(all_paths)


checklist, full_hitlist = get_hitlist()

print("Paths to hit")
print(checklist)

print(full_hitlist)

start = "A"

path = []
checked_off = []

current_state = start
remaining = copy.copy(checklist)

i = 0
while len(checked_off) < len(checklist) and i < 100:
	i += 1

	next_state_options =  [y for y in states if y != current_state]
	
	if False: # could choose randomly
		next_state = random.choice(next_state_options)
	else: # could prefer those on the list
		useful_options = [y for y in remaining if y[0] == current_state]

		if len(useful_options) > 0:
			next_state = random.choice(useful_options)[1]
		else:
			next_state = random.choice(next_state_options)


	link_name = current_state + next_state
	path.append(link_name)

	if link_name in checklist and link_name not in checked_off:
		checked_off.append(link_name)
		remaining.remove(link_name)


	current_state = next_state


print(path)
print(len(path))

saved = ['AC', 'CD', 'DF', 'FD', 'DC', 'CE', 'EF', 'FC', 'CD', 'DE', 'ED', 'DE', 'EF', 'FA', 'AB', 'BE', 'EA', 'AB', 'BA', 'AE', 'ED', 'DE', 'ED', 'DC', 'CF', 'FD', 'DC', 'CB', 'BE', 'EA', 'AE', 'EC', 'CF', 'FA', 'AF', 'FE']

#### CALCULATE THE PROJECTED TIME ALSO





























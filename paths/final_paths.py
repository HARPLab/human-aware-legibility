from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

goal_list 	= [goal_a, goal_b, goal_c, goal_d, goal_e, goal_f]
goal_colors = ['red', 'blue', 'purple', 'green', 'orange', 'pink']

# GENERATED PATHS
export_name = 'toptwo' #'null' #'toptwo'

inspection_save_path = "study_paths/"

def inspect_path_set():

	print("\nEarly dict")
	early_dict 	= setup_path_dict('early')
	print("\nLate dict")
	late_dict 	= setup_path_dict('late')
	print("\nEven dict")
	even_dict	= setup_path_dict('even')

	print("Obstacle path")
	obstacle_dict = setup_path_dict('obs')

	export_path_dict('early', early_dict)
	export_path_dict('late', late_dict)
	export_path_dict('even', even_dict)
	export_path_dict('obs', obstacle_dict)

	##### Calculate the path lengths, also
	count_path_lengths(inspection_save_path, early_dict, late_dict, even_dict, obstacle_dict)

	##### draw them in groups by the path segment
	draw_paths_by_segment(inspection_save_path, early_dict, late_dict, even_dict)

	##### draw them in groups by the early/late/etc/group
	draw_paths_by_dict(inspection_save_path, early_dict, late_dict, even_dict, obstacle_dict)

	##### special drawings for obstacle avoidance
	draw_obstacle_sets(inspection_save_path, early_dict, late_dict, even_dict, obstacle_dict)

def get_xy_from_path(path):
	if len(path) == 0:
		return [], []

	x_list, y_list = list(map(list, zip(*path)))

	return x_list, y_list

def draw_paths_by_segment(inspection_save_path, early_dict, late_dict, even_dict):

	for key in early_dict.keys():
		path_early 	= early_dict[key]
		path_late 	= late_dict[key]
		path_even 	= even_dict[key]

		early_x, early_y 	= get_xy_from_path(path_early)
		late_x, late_y 		= get_xy_from_path(path_late)
		even_x, even_y 		= get_xy_from_path(path_even)


		plt.figure(figsize=(5, 4))
		f, ax = plt.subplots()

		buffer = 2
		ax = plt.gca()
		ax.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
		ax.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
		ax.set_aspect('equal')

		plt.plot(early_x, 	early_y, 	label = "early", color='red')
		plt.plot(late_x, 	late_y, 	label = "late", color='green')
		plt.plot(even_x, 	even_y, 	label = "even", color='blue')
		 
		plt.title('Path options for ' + key)
			 
		for j in range(len(goal_list)):
			goal 	= goal_list[j]
			color = goal_colors[j]
			circle = plt.Circle(goal, .1, color=color)
			ax.add_patch(circle)

		# show a legend on the plot
		plt.legend() #loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
		 
		# function to show the plot
		plt.savefig(inspection_save_path + key + '.png')
		plt.clf()
		plt.close()

	
	print("Exported images of all paths")		


def draw_paths_by_dict(inspection_save_path, early_dict, late_dict, even_dict, obstacle_dict):
	fig, axes = plt.subplot_mosaic("ABCD;EFGH;IJKL", figsize=(8, 6), gridspec_kw={'width_ratios':[1, 1, 1, 1], 'height_ratios':[1, 1, 1]})

	ax_mappings = {}
	ax_early 	= axes['A']
	ax_even 	= axes['E']
	ax_late 	= axes['I']

	ax_early2 	= axes['B']
	ax_even2 	= axes['F']
	ax_late2 	= axes['J']

	ax_early3 	= axes['C']
	ax_even3 	= axes['G']
	ax_late3 	= axes['K']

	ax_early4 	= axes['D']
	ax_even4 	= axes['H']
	ax_late4 	= axes['L']

	buffer = 1
	ax_early.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_early.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_early.set_aspect('equal')

	ax_even.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_even.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_even.set_aspect('equal')

	ax_late.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_late.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_late.set_aspect('equal')

	ax_early2.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_early2.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_early2.set_aspect('equal')

	ax_even2.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_even2.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_even2.set_aspect('equal')

	ax_late2.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_late2.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_late2.set_aspect('equal')

	ax_early3.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_early3.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_early3.set_aspect('equal')

	ax_even3.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_even3.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_even3.set_aspect('equal')

	ax_late3.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_late3.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_late3.set_aspect('equal')

	ax_early4.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_early4.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_early4.set_aspect('equal')

	ax_even4.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_even4.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_even4.set_aspect('equal')

	ax_late4.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	ax_late4.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax_late4.set_aspect('equal')

	### -AB -AC -AD -AE -AF -BA -BC -BD -BE -BF -CA -CB -CD -CE -CF -DA DB DC -DE -DF -EA -EB -EC -ED -EF -FA -FB -FC -FD -FE

	title1 = "Inwards / Up"
	group1 = ['AB', 'CB', 'DA', 'FC', 'EA', 'EC', 'FE', 'DE', ]

	title2 = "Outwards / Down"
	group2 = ['BA', 'BC', 'BD', 'BF', 'EF', 'ED', "AD", 'CF']

	title3 = "Long / Diagonal"
	group3 = ['AF', 'AC'] # FA, DC CD CA DF FD

	# title4 = "Short Diagonal"
	# group4 = ['AE', 'CE', 'DB', 'FB']

	title4 = "Obstacle Repulsion"
	group4 = ['AC-obs', 'AF-obs']

	### boringly flat
	# 'BE', EB

	ax_early.set_title("Early\n " + title1, fontweight="bold")
	ax_even.set_title("Even\n " + title1, fontweight="bold")
	ax_late.set_title("Late\n " + title1, fontweight="bold")

	ax_early2.set_title("Early\n " + title2, fontweight="bold")
	ax_even2.set_title("Even\n " + title2, fontweight="bold")
	ax_late2.set_title("Late\n " + title2, fontweight="bold")

	ax_early3.set_title("Early\n " + title3, fontweight="bold")
	ax_even3.set_title("Even\n " + title3, fontweight="bold")
	ax_late3.set_title("Late\n " + title3, fontweight="bold")

	ax_early4.set_title("Early\n " + title4, fontweight="bold")
	ax_even4.set_title("Even\n " + title4, fontweight="bold")
	ax_late4.set_title("Late\n " + title4, fontweight="bold")

	for j in range(len(goal_list)):
		goal 	= goal_list[j]
		color = goal_colors[j]

		circle1 = plt.Circle(goal, .1, color=color)
		ax_early.add_patch(circle1)

		circle2 = plt.Circle(goal, .1, color=color)
		ax_late.add_patch(circle2)
		
		circle3 = plt.Circle(goal, .1, color=color)
		ax_even.add_patch(circle3)

		circle4 = plt.Circle(goal, .1, color=color)
		ax_early2.add_patch(circle4)

		circle5 = plt.Circle(goal, .1, color=color)
		ax_late2.add_patch(circle5)
		
		circle6 = plt.Circle(goal, .1, color=color)
		ax_even2.add_patch(circle6)

		circle7 = plt.Circle(goal, .1, color=color)
		ax_early3.add_patch(circle7)

		circle8 = plt.Circle(goal, .1, color=color)
		ax_late3.add_patch(circle8)
		
		circle9 = plt.Circle(goal, .1, color=color)
		ax_even3.add_patch(circle9)

		circle10 = plt.Circle(goal, .1, color=color)
		ax_early4.add_patch(circle10)

		circle11 = plt.Circle(goal, .1, color=color)
		ax_late4.add_patch(circle11)
		
		circle12 = plt.Circle(goal, .1, color=color)
		ax_even4.add_patch(circle12)


	for key in early_dict.keys():
		path_early 	= early_dict[key]
		path_late 	= late_dict[key]
		path_even 	= even_dict[key]

		early_x, early_y 	= get_xy_from_path(path_early)
		late_x, late_y 		= get_xy_from_path(path_late)
		even_x, even_y 		= get_xy_from_path(path_even)


		if key in group1:
			ax_early.plot(early_x, 	early_y, 	label = "early", color='red')
			ax_late.plot(late_x, 	late_y, 	label = "late", color='green')
			ax_even.plot(even_x, 	even_y, 	label = "even", color='blue')

		elif key in group2:
			ax_early2.plot(early_x, 	early_y, 	label = "early", color='red')
			ax_late2.plot(late_x, 	late_y, 	label = "late", color='green')
			ax_even2.plot(even_x, 	even_y, 	label = "even", color='blue')

		elif key in group3:
			ax_early3.plot(early_x, 	early_y, 	label = "early", color='red')
			ax_late3.plot(late_x, 	late_y, 	label = "late", color='green')
			ax_even3.plot(even_x, 	even_y, 	label = "even", color='blue')

		elif key in group4:
			ax_early4.plot(early_x, 	early_y, 	label = "early", color='red')
			ax_late4.plot(late_x, 	late_y, 	label = "late", color='green')
			ax_even4.plot(even_x, 	even_y, 	label = "even", color='blue')
		 
		# plt.title('Path options for ' + key)
			 
	##### Add the obstacle paths
	AC_early 	= obstacle_dict['AC_OBS-early']
	AC_late 	= obstacle_dict['AC_OBS-late']
	AC_even 	= obstacle_dict['AC_OBS-even']

	AF_early 	= obstacle_dict['AF_OBS-early']
	AF_late 	= obstacle_dict['AF_OBS-late']
	AF_even 	= obstacle_dict['AF_OBS-even']


	AC_early_x, AC_early_y 		= get_xy_from_path(AC_early)
	AC_late_x, AC_late_y 		= get_xy_from_path(AC_late)
	AC_even_x, AC_even_y 		= get_xy_from_path(AC_even)

	AF_early_x, AF_early_y 		= get_xy_from_path(AF_early)
	AF_late_x, AF_late_y 		= get_xy_from_path(AF_late)
	AF_even_x, AF_even_y 		= get_xy_from_path(AF_even)


	ax_early4.plot(AC_early_x, 	AC_early_y, 	label = "early", color='red')
	ax_late4.plot(AC_late_x, 	AC_late_y, 	label = "late", color='green')
	ax_even4.plot(AC_even_x, 	AC_even_y, 	label = "even", color='blue')


	ax_early4.plot(AF_early_x, 	AF_early_y, 	label = "early", color='red')
	ax_late4.plot(AF_late_x, 	AF_late_y, 	label = "late", color='green')
	ax_even4.plot(AF_even_x, 	AF_even_y, 	label = "even", color='blue')


	# show a legend on the plot
	# plt.legend() #loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
	 
	# function to show the plot
	plt.tight_layout()
	plt.savefig(inspection_save_path + "overview" + '.png')
	plt.clf()
	plt.close()

	
	print("Exported images of all paths")


def draw_obstacle_sets(inspection_save_path, early_dict, late_dict, even_dict, obstacle_paths):
	pass

def count_path_lengths(inspection_save_path, early_dict, late_dict, even_dict, obstacle_paths):
	# Create a csv of the lengths for each path

	length_rows = []
	for key in early_dict.keys():
		early_length 	= get_path_length(early_dict[key])
		late_length 	= get_path_length(late_dict[key])
		even_length 	= get_path_length(even_dict[key])


		row = [key, early_length, late_length, even_length]

		length_rows.append(row)

	df = pd.DataFrame(length_rows, columns=['PATH', 'early', 'late', 'even'])
	df.to_csv(inspection_save_path + "lengths.csv")

	
def dist_between(x1, x2):
        # print(x1)
        # print(x2)
        # print(x1[0], x2[0], x1[1], x2[1])

        distance = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        return distance

def get_path_length(path):
	total_length = 0
	for pi in range(1, len(path)):
		p0 = path[pi - 1]
		p1 = path[pi]

		total_length += dist_between(p0, p1)
	return total_length


ramp_a = [goal_a, [goal_a[0], goal_a[1] + .05], [goal_a[0], goal_a[1] + .1], [goal_a[0], goal_a[1] + .15], [goal_a[0], goal_a[1] + .2], [goal_a[0], goal_a[1] + .25], [goal_a[0], goal_a[1] + .3]]
ramp_b = [goal_b, [goal_b[0], goal_b[1] + .05], [goal_b[0], goal_b[1] + .1], [goal_b[0], goal_b[1] + .15], [goal_b[0], goal_b[1] + .2], [goal_b[0], goal_b[1] + .25], [goal_b[0], goal_b[1] + .3]]
ramp_c = [goal_c, [goal_c[0], goal_c[1] + .05], [goal_c[0], goal_c[1] + .1], [goal_c[0], goal_c[1] + .15], [goal_c[0], goal_c[1] + .2], [goal_c[0], goal_c[1] + .25], [goal_c[0], goal_c[1] + .3]]
ramp_d = [goal_d, [goal_d[0], goal_d[1] - .05], [goal_d[0], goal_d[1] - .1], [goal_d[0], goal_d[1] - .15], [goal_d[0], goal_d[1] - .2], [goal_d[0], goal_d[1] - .25], [goal_d[0], goal_d[1] - .3]]
ramp_e = [goal_e, [goal_e[0], goal_e[1] - .05], [goal_e[0], goal_e[1] - .1], [goal_e[0], goal_e[1] - .15], [goal_e[0], goal_e[1] - .2], [goal_e[0], goal_e[1] - .25], [goal_e[0], goal_e[1] - .3]]
ramp_f = [goal_f, [goal_f[0], goal_f[1] - .05], [goal_f[0], goal_f[1] - .1], [goal_f[0], goal_f[1] - .15], [goal_f[0], goal_f[1] - .2], [goal_f[0], goal_f[1] - .25], [goal_f[0], goal_f[1] - .3]]

ramps = {}
ramps['A'] = ramp_a
ramps['B'] = ramp_b
ramps['C'] = ramp_c
ramps['D'] = ramp_d
ramps['E'] = ramp_e
ramps['F'] = ramp_f

def add_offramps(path_dict):
	new_path_dict = {}

	for key in path_dict.keys():
		end_point = key[1]
		new_path_dict[key] = path_dict[key] + ramps[end_point]

		dist = dist_between(path_dict[key][-1], ramps[end_point][0])
		if dist > .51:
			print(key)
			print(dist)


	return new_path_dict


def get_early_paths():
	tag = 'early'

	path_ab = []
	path_ac = []
	path_ad = []
	path_ae = []
	path_af = []

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	path_ba = []
	path_bc = None
	path_bd = []
	path_be = []
	path_bf = None

	path_dict = {}
	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ac
	path_dict['AD'] = path_ad
	path_dict['AE'] = path_ae
	path_dict['AF'] = path_af

	path_dict['BA'] = path_ba
	path_dict['BC'] = horizontal_flip(path_ba)
	path_dict['BD'] = path_bd
	path_dict['BE'] = path_be
	path_dict['BF'] = horizontal_flip(path_bd)

	path_dict = add_offramps(path_dict)

	# Return the name and the list
	return path_dict

def get_late_paths():
	path_ab = []
	path_ac = []
	path_ad = []
	path_ae = []
	path_af = []

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	path_ba = []
	path_bc = None
	path_bd = []
	path_be = []
	path_bf = None

	path_dict = {}
	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ac
	path_dict['AD'] = path_ad
	path_dict['AE'] = path_ae
	path_dict['AF'] = path_af

	path_dict['BA'] = path_ba
	path_dict['BC'] = horizontal_flip(path_ba)
	path_dict['BD'] = path_bd
	path_dict['BE'] = path_be
	path_dict['BF'] = horizontal_flip(path_bd)

	path_dict = add_offramps(path_dict)

	# Return the name and the list
	return path_dict

def get_even_paths():
	path_ab = []
	path_ac = []
	path_ad = []
	path_ae = []
	path_af = []

	path_ba = []
	path_bc = None
	path_bd = []
	path_be = []
	path_bf = None

	path_dict = {}
	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ac
	path_dict['AD'] = path_ad
	path_dict['AE'] = path_ae
	path_dict['AF'] = path_af

	path_dict['BA'] = path_ba
	path_dict['BC'] = horizontal_flip(path_ba)
	path_dict['BD'] = path_bd
	path_dict['BE'] = path_be
	path_dict['BF'] = horizontal_flip(path_bd)

	path_dict = add_offramps(path_dict)

	# Return the name and the list
	return path_dict

def get_obstacle_paths():
	path_ac_early 	= [[1.0, -1.0], [1.04065768459355, -0.8025269589536208], [1.0832820306943416, -0.6428797381248794], [1.1289671228738964, -0.520291416533252], [1.178297200741314, -0.43463636687906226], [1.2317275172872229, -0.3860285587022211], [1.2899242335737382, -0.37457777669235376], [1.354073492123451, -0.4001353434456025], [1.4261565961136102, -0.46194243336133384], [1.5093078882123554, -0.5583418730791333], [1.6083265900046784, -0.6862238827870129], [1.7302703112864044, -0.8399812701548578], [1.8815774859518668, -1.0109144416391176], [2.0638327496633893, -1.185564825071889], [2.324015164619231, -1.3487475016256674], [2.620565009283986, -1.4950484644766342], [2.928805660621607, -1.5929197458787965], [3.2036970182689366, -1.6157328305102625], [3.412249052071795, -1.59291055140455], [3.5677396553885448, -1.5557374590877289], [3.6943844884166372, -1.5158174111676148], [3.807271734569141, -1.4760576897013469], [3.914319738581377, -1.4367844563674157], [4.018986217513451, -1.3978392570240676], [4.122864068930434, -1.3590717974386204], [4.226428865679596, -1.3203942512526454], [4.329868588194759, -1.2817585230696944], [4.433257922902051, -1.2431413632867923], [4.536626792037579, -1.2045322350607006], [4.639987301979826, -1.1659265454572667], [4.743344406136944, -1.1273223454306298], [4.846700155040498, -1.0887188248365482], [4.950055375561886, -1.0501156447204016], [5.0, -1.0]]
	path_ac_late 	= [[1.0, -1.0], [1.154159016877842, -1.0545450400944207], [1.3083180337436946, -1.109090081075342], [1.4624770500511748, -1.163635124186041], [1.6166360606404029, -1.218180172338831], [1.7707950281765428, -1.2727252314293394], [1.9249537142078483, -1.3272704187135165], [2.0791101326978962, -1.3818166788701567], [2.233258244430597, -1.4363666612307069], [2.3874066004214343, -1.490919914052634], [2.5415575308927947, -1.5454582748567487], [2.695709641901434, -1.5999556532078456], [2.8498601649566773, -1.654365541522785], [3.0040072753922336, -1.708593426801302], [3.1581595177169732, -1.7624153080156626], [3.3123411317498013, -1.815276595230516], [3.4665690997656307, -1.8658794863485373], [3.620714297306663, -1.9114083235536599], [3.7739482384033574, -1.9466434614605599], [3.9193358646094354, -1.9642667536647005], [4.050996107349441, -1.957513184663464], [4.176096613946983, -1.9216648825428588], [4.291277601785897, -1.857900187913299], [4.393105511963645, -1.7723991933299625], [4.506265721287969, -1.6735797020469885], [4.607734480189492, -1.5694013297628648], [4.701369669409918, -1.4632441108385046], [4.785087071000749, -1.3551232803198547], [4.858756430224998, -1.2398453728213488], [4.922530428142207, -1.1107872809799482], [4.958486203908188, -0.9883682722280825], [5.050917204483773, -0.859339805798577], [4.9504499239577395, -1.0496321360312346], [5.0, -1.0]] #[]
	path_ac_even 	= [[1.0, -1.0], [1.1149193732425284, -1.019512234305543], [1.2298482475724262, -1.0455038124936946], [1.3448792285107045, -1.0777087331384585], [1.460204869585528, -1.1157602733415954], [1.576136427173082, -1.1591607384223177], [1.6931232219120265, -1.2072401206360495], [1.811769570531599, -1.2590999044764555], [1.9328423920811015, -1.313537359332425], [2.05725775049522, -1.3689451969586257], [2.190507045793057, -1.4235708446119641], [2.332882700750063, -1.4755129794944204], [2.4841597626421943, -1.522135959744047], [2.6430637616021753, -1.559999150119519], [2.8065166848892096, -1.5851666136101414], [2.969092638391727, -1.5943079816834484], [3.1238297293279405, -1.586565756848071], [3.2653532899070736, -1.564885755053261], [3.3934184746594234, -1.5348347417783232], [3.512177643759454, -1.501286296490134], [3.6260434323143045, -1.4667168124120271], [3.7377031991224174, -1.4319622670592749], [3.8484394750116597, -1.3972199060689783], [3.958801852761657, -1.3625112286963437], [4.069015024772571, -1.3278251611401264], [4.179176458915717, -1.2931516200428637], [4.289317384064062, -1.2584843373425758], [4.399450089692621, -1.2238199671143124], [4.509579462745524, -1.189156901561467], [4.619707469864797, -1.1544944092646194], [4.729834912152086, -1.1198321678910577], [4.839962119880994, -1.085170038260407], [4.950089231370053, -1.0505079618178723], [5.0, -1.0]]
	
	path_af_early 	= [[1.0, -1.0], [1.1142689509417216, -1.0616224752808818], [1.2296995678798395, -1.126732098352018], [1.346215660655739, -1.194982916209246], [1.4637478877096195, -1.2658859457116192], [1.5822392425172398, -1.3387733440366605], [1.7016399262137856, -1.4128198067721525], [1.8204625819400093, -1.4871016214275388], [1.9394408962289984, -1.5610723974760328], [2.062083391594216, -1.6425122260419163], [2.196102291769474, -1.7242007768626835], [2.3448364123150536, -1.8031455882129666], [2.5060266098600295, -1.8789260496763487], [2.6747142467769045, -1.9513283457205306], [2.8349342645089806, -2.0020005467702346], [2.9879085339078064, -2.0281071481547], [3.1461995130515286, -2.061294081932079], [3.308627584321639, -2.0972124887282724], [3.4742410971873197, -2.160507768835285], [3.6402313985691137, -2.246595952230326], [3.8038529860524144, -2.3469593188571003], [3.967474536216126, -2.447322682096602], [4.131096074385998, -2.5476860424650045], [4.294717607096866, -2.6480494007087905], [4.4583391373932315, -2.748412757851101], [4.62196066659396, -2.8487761144870603], [4.78558219527053, -2.9491394709177947], [4.949203723691698, -3.049502827284859], [5.0, -3.0]]
	path_af_late 	= [[1.0, -1.0], [1.1711863158318148, -1.0539570329455135], [1.342372631665987, -1.1079140658966393], [1.5135589475086193, -1.1618710988595262], [1.6847452633842013, -1.21582813184259], [1.8559315793888347, -1.2697851648394847], [2.0271178959286327, -1.3237421977397217], [2.198304215204164, -1.377699230017211], [2.3679162304172165, -1.4310689451334135], [2.5302446879150646, -1.4902192413901783], [2.6941948596251866, -1.5492458142017373], [2.8925803359305586, -1.6505146853144554], [3.121176430485129, -1.8114790490150086], [3.3325384505836984, -1.934565822157622], [3.523931001274881, -2.0311655405547597], [3.7127758928616315, -2.1586633497857313], [3.902522618387213, -2.3028664757309794], [4.092147626777105, -2.4470382729902207], [4.281526954802849, -2.590955569286537], [4.4871789492884435, -2.73033495769785], [4.700070802347392, -2.8619858430064853], [4.85734920484218, -2.9521457743026622], [4.954085250660903, -2.9863214941120404], [4.984280043585036, -3.006414348651466], [5.000578618238872, -3.0060527138200617], [5.011625737340479, -2.994101721964911], [5.010121330454056, -3.000727965134355], [4.949830173878053, -3.049213844330472], [5.0, -3.0]]
	path_af_even 	= [[1.0, -1.0], [1.0994182516690971, -1.0210545007212048], [1.2013159130599536, -1.047257977539884], [1.2961531831140682, -1.0835022091299862], [1.3924522592503212, -1.1236175495205203], [1.4896417891428386, -1.1682105953922965], [1.5834582698638036, -1.2177989010227215], [1.6741968270131902, -1.2944691299360713], [1.7917910950433982, -1.3895769381944956], [1.936121903146585, -1.511855785062103], [2.1207854671437234, -1.6392128447053567], [2.3418030655210367, -1.765644939362415], [2.57024658158403, -1.8587796118042281], [2.8075400494559353, -1.9309007201295587], [3.0487312865774334, -1.981939568811921], [3.286589188594533, -2.0292075974418964], [3.5153774279719396, -2.0674900822978097], [3.7336299630258796, -2.1110137218513776], [3.9387869227957104, -2.1806368217455936], [4.124392708620047, -2.278557658805892], [4.2778242515957166, -2.417358128967342], [4.395816262310789, -2.5570412707265615], [4.525690113505102, -2.6972256101250727], [4.6652624373355165, -2.827727830566381], [4.782389420283344, -2.9316372748950794], [4.8708686883209955, -3.0052330151446753], [4.914800820749876, -3.0419612996324985], [4.949291021582027, -3.048492673049541]]

	# AC_OBS-even
	obstacle_paths = {}
	obstacle_paths['AC_OBS-early'] 	= path_ac_early
	obstacle_paths['AC_OBS-even']	= path_ac_even
	obstacle_paths['AC_OBS-late']	= path_ac_late
	obstacle_paths['AF_OBS-early'] 	= path_af_early
	obstacle_paths['AF_OBS-even'] 	= path_af_even
	obstacle_paths['AF_OBS-late'] 	= path_af_late


	obstacle_paths['FD_OBS-early'] 	= horizontal_flip(vertical_flip(path_ac_early))
	obstacle_paths['FD_OBS-even']	= horizontal_flip(vertical_flip(path_ac_even))
	obstacle_paths['FD_OBS-late']	= horizontal_flip(vertical_flip(path_ac_late))

	obstacle_paths['DF_OBS-early'] 	= vertical_flip(path_ac_early)
	obstacle_paths['DF_OBS-even'] 	= vertical_flip(path_ac_even)
	obstacle_paths['DF_OBS-late'] 	= vertical_flip(path_ac_late)

	obstacle_paths['CA_OBS-early'] 	= horizontal_flip(path_ac_early)
	obstacle_paths['CA_OBS-even']	= horizontal_flip(path_ac_even)
	obstacle_paths['CA_OBS-late']	= horizontal_flip(path_ac_late)

	obstacle_paths['CD_OBS-early'] 	= horizontal_flip(path_af_early)
	obstacle_paths['CD_OBS-even'] 	= horizontal_flip(path_af_even)
	obstacle_paths['CD_OBS-late'] 	= horizontal_flip(path_af_late)

	obstacle_paths['FA_OBS-early'] 	= horizontal_flip(vertical_flip(path_af_early))
	obstacle_paths['FA_OBS-even']	= horizontal_flip(vertical_flip(path_af_even))
	obstacle_paths['FA_OBS-late']	= horizontal_flip(vertical_flip(path_af_late))
	
	obstacle_paths['DC_OBS-early'] 	= vertical_flip(path_af_early)
	obstacle_paths['DC_OBS-even'] 	= vertical_flip(path_af_even)
	obstacle_paths['DC_OBS-late'] 	= vertical_flip(path_af_late)

	obstacle_paths = add_offramps(obstacle_paths)

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	# Return the name and the list
	return obstacle_paths

def get_straight_line_paths():
	# VANILLA PATHS
	# generate_vanilla_straight_line_paths_for_testing(goal_a, [goal_b, goal_c, goal_d, goal_e, goal_f])

	path_ab = [[1.0, -1.0], [1.25, -1.0], [1.5, -1.0], [1.75, -1.0], [2.0, -1.0], [2.25, -1.0], [2.5, -1.0], [2.75, -1.0], [3.0, -1.0]]
	path_ac = [[1.0, -1.0], [1.5, -1.0], [2.0, -1.0], [2.5, -1.0], [3.0, -1.0], [3.5, -1.0], [4.0, -1.0], [4.5, -1.0], [5.0, -1.0]]
	path_ad = [[1.0, -1.0], [1.0, -1.25], [1.0, -1.5], [1.0, -1.75], [1.0, -2.0], [1.0, -2.25], [1.0, -2.5], [1.0, -2.75], [1.0, -3.0]]
	path_ae = [[1.0, -1.0], [1.25, -1.25], [1.5, -1.5], [1.75, -1.75], [2.0, -2.0], [2.25, -2.25], [2.5, -2.5], [2.75, -2.75], [3.0, -3.0]]
	path_af = [[1.0, -1.0], [1.5, -1.25], [2.0, -1.5], [2.5, -1.75], [3.0, -2.0], [3.5, -2.25], [4.0, -2.5], [4.5, -2.75], [5.0, -3.0]]

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	path_ba = [[3.0, -1.0], [2.75, -1.0], [2.5, -1.0], [2.25, -1.0], [2.0, -1.0], [1.75, -1.0], [1.5, -1.0], [1.25, -1.0], [1.0, -1.0]]
	path_bc = [[3.0, -1.0], [3.25, -1.0], [3.5, -1.0], [3.75, -1.0], [4.0, -1.0], [4.25, -1.0], [4.5, -1.0], [4.75, -1.0], [5.0, -1.0]]
	path_bd = [[3.0, -1.0], [2.75, -1.25], [2.5, -1.5], [2.25, -1.75], [2.0, -2.0], [1.75, -2.25], [1.5, -2.5], [1.25, -2.75], [1.0, -3.0]]
	path_be = [[3.0, -1.0], [3.0, -1.25], [3.0, -1.5], [3.0, -1.75], [3.0, -2.0], [3.0, -2.25], [3.0, -2.5], [3.0, -2.75], [3.0, -3.0]]
	path_bf = [[3.0, -1.0], [3.25, -1.25], [3.5, -1.5], [3.75, -1.75], [4.0, -2.0], [4.25, -2.25], [4.5, -2.5], [4.75, -2.75], [5.0, -3.0]]

	path_dict = {}
	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ac
	path_dict['AD'] = path_ad
	path_dict['AE'] = path_ae
	path_dict['AF'] = path_af

	path_dict['BA'] = path_ba
	path_dict['BC'] = horizontal_flip(path_ba)
	path_dict['BD'] = path_bd
	path_dict['BE'] = path_be
	path_dict['BF'] = horizontal_flip(path_bd)


	path_dict = add_offramps(path_dict)

	# Return the name and the list
	return path_dict

def get_bigger_path_rectangle_1():
	path_ab = [[-1.0, -1.0], [-0.625, -1.0], [-0.25, -1.0], [0.125, -1.0], [0.5, -1.0], [0.875, -1.0], [1.25, -1.0], [1.625, -1.0], [2.0, -1.0]]
	path_ac = [[-1.0, -1.0], [-0.25, -1.0], [0.5, -1.0], [1.25, -1.0], [2.0, -1.0], [2.75, -1.0], [3.5, -1.0], [4.25, -1.0], [5.0, -1.0]]
	path_ad = [[-1.0, -1.0], [-1.0, -1.375], [-1.0, -1.75], [-1.0, -2.125], [-1.0, -2.5], [-1.0, -2.875], [-1.0, -3.25], [-1.0, -3.625], [-1.0, -4.0]]
	path_ae = [[-1.0, -1.0], [-0.625, -1.375], [-0.25, -1.75], [0.125, -2.125], [0.5, -2.5], [0.875, -2.875], [1.25, -3.25], [1.625, -3.625], [2.0, -4.0]]
	path_af = [[-1.0, -1.0], [-0.25, -1.375], [0.5, -1.75], [1.25, -2.125], [2.0, -2.5], [2.75, -2.875], [3.5, -3.25], [4.25, -3.625], [5.0, -4.0]]


	path_ba = [[-1.0, -1.0], [-0.625, -1.0], [-0.25, -1.0], [0.125, -1.0], [0.5, -1.0], [0.875, -1.0], [1.25, -1.0], [1.625, -1.0], [2.0, -1.0]]
	path_bc = [[-1.0, -1.0], [-0.25, -1.0], [0.5, -1.0], [1.25, -1.0], [2.0, -1.0], [2.75, -1.0], [3.5, -1.0], [4.25, -1.0], [5.0, -1.0]]
	path_bd = [[-1.0, -1.0], [-1.0, -1.375], [-1.0, -1.75], [-1.0, -2.125], [-1.0, -2.5], [-1.0, -2.875], [-1.0, -3.25], [-1.0, -3.625], [-1.0, -4.0]]
	path_be = [[-1.0, -1.0], [-0.625, -1.375], [-0.25, -1.75], [0.125, -2.125], [0.5, -2.5], [0.875, -2.875], [1.25, -3.25], [1.625, -3.625], [2.0, -4.0]]
	path_bf = [[-1.0, -1.0], [-0.25, -1.375], [0.5, -1.75], [1.25, -2.125], [2.0, -2.5], [2.75, -2.875], [3.5, -3.25], [4.25, -3.625], [5.0, -4.0]]

	path_dict = {}
	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ab
	path_dict['AD'] = path_ab
	path_dict['AE'] = path_ab
	path_dict['AF'] = path_ab

	path_dict['BA'] = path_ab
	path_dict['BC'] = path_ab
	path_dict['BD'] = path_ab
	path_dict['BE'] = path_ab
	path_dict['BF'] = path_ab

	path_dict = add_offramps(path_dict)

	# Return the name and the list
	return path_dict


def get_curvey_line_paths_1():
	# # Curvier paths from my code!
	path_ab = [[ 1., -1.], [ 1.19634799, -0.9675439], [ 1.38696757, -0.9396737], [ 1.5707822,  -0.91637407], [ 1.74675563, -0.89764527], [ 1.91389779, -0.88350265], [ 2.07127019, -0.87397608], [ 2.21799127, -0.8691097], [ 2.35324126, -0.86896153], [ 2.47626681, -0.87360334], [ 2.58638522, -0.88312052], [ 2.68298828, -0.89761207], [ 2.76554573, -0.91719071], [ 2.83360823, -0.94198308], [ 2.88680999, -0.97212998], [ 2.92487083, -1.00778683], [ 2.94759786, -1.04912413], [3, -1]]
	path_ac = [[ 1., -1.], [ 1.11368137, -0.98140284], [ 1.22742351, -0.96473847 ], [ 1.34128619, -0.94994051 ], [ 1.45532829, -0.93694329 ], [ 1.56960782, -0.92568186 ], [ 1.68418199, -0.91609189 ], [ 1.79910729, -0.90810962 ], [ 1.91443952, -0.90167181 ], [ 2.03023386, -0.89671569 ], [ 2.14654492, -0.89317886 ], [ 2.26342682, -0.89099931 ], [ 2.38093322, -0.89011529 ], [ 2.4991174,  -0.8904653  ], [ 2.61803227, -0.89198803 ], [ 2.7377305,  -0.89462229 ], [ 2.8582645,  -0.89830697 ], [ 2.97968654, -0.90298098 ], [ 3.10204875, -0.9085832  ], [ 3.22540321, -0.91505241 ], [ 3.34980198, -0.92232728 ], [ 3.47529721, -0.93034626 ], [ 3.6019411,  -0.93904756 ], [ 3.72978607, -0.9483691  ], [ 3.8588847, -0.95824843 ], [ 3.98928989, -0.9686227 ], [ 4.12105485, -0.97942859], [ 4.25423316, -0.99060226], [ 4.38887889, -1.00207931], [ 4.52504655, -1.01379468], [ 4.66279127, -1.02568264], [ 4.80216877, -1.03767672], [ 4.94323543, -1.04970963], [ 5, -1.0]]
	path_ad = [[ 1., -1. ], [ 0.95770111, -1.19513019], [ 0.92008508, -1.38473945], [ 0.88723032, -1.56795431], [ 0.85922746, -1.74393511], [ 0.83617874, -1.91187987], [ 0.81819738, -2.07102781], [ 0.80540715, -2.2206629], [ 0.79794194, -2.36011707], [ 0.79594556, -2.48877333], [ 0.79957152, -2.60606861], [ 0.80898299, -2.71149643], [ 0.82435288, -2.80460929], [ 0.84586399, -2.88502087], [ 0.8737093,  -2.9524079 ], [ 0.90809241, -3.00651185], [ 0.94922806, -3.04714029], [1.0, -3.0]]
	path_ae = [[ 1., -1.], [ 1.09347342, -1.10510911], [ 1.18688026, -1.21006585], [ 1.28015412, -1.31471796], [ 1.37322898, -1.41891336], [ 1.46603937, -1.52250028], [ 1.55852057, -1.6253273], [ 1.65060881, -1.72724353], [ 1.74224141, -1.8280986], [ 1.83335703, -1.92774283], [ 1.92389582, -2.0260273], [ 2.0137996,  -2.12280391], [ 2.10301208, -2.21792547], [ 2.19147905, -2.31124581], [ 2.27914853, -2.40261982], [ 2.36597101, -2.49190354], [ 2.45189964, -2.57895424], [ 2.53689037, -2.66363043], [ 2.62090223, -2.74579199], [ 2.70389745, -2.82530018], [ 2.78584171, -2.90201769], [ 2.86670433, -2.97580869], [ 2.94645846, -3.04653886], [3.0, -3.0] ]
	path_af = [[ 1., -1.], [ 1.13715194, -1.07246145], [ 1.27471105, -1.14595233], [ 1.41271057, -1.22038189], [ 1.55118327, -1.29565933], [ 1.69016156, -1.37169368], [ 1.82967757, -1.44839377], [ 1.96976325, -1.52566822], [ 2.11045048, -1.60342532], [ 2.25177114, -1.68157301], [ 2.39375723, -1.76001884], [ 2.53644095, -1.83866991], [ 2.67985482, -1.91743279], [ 2.82403174, -1.9962135], [ 2.96900513, -2.07491745], [ 3.11480901, -2.15344937], [ 3.26147807, -2.23171329], [ 3.40904785, -2.30961243], [ 3.55755476, -2.38704919], [ 3.70703623, -2.46392508], [ 3.85753081, -2.54014065], [ 4.00907825, -2.61559544], [ 4.16171966, -2.69018793], [ 4.31549756, -2.76381544], [ 4.47045604, -2.83637411], [ 4.62664085, -2.90775882], [ 4.78409952, -2.9778631], [ 4.94288148, -3.0465791], [5.0, -3]]

	# Some paths from my code!
	path_ba = [[ 3.0, -1.], [ 2.78503369, -0.97566994], [ 2.57910723, -0.95427874], [ 2.38289537, -0.93598619], [ 2.1970388,  -0.92095047], [ 2.02214191, -0.90932796], [ 1.85877079, -0.90127304], [ 1.70745117, -0.89693786], [ 1.56866669, -0.89647229], [ 1.44285713, -0.90002378], [ 1.33041692, -0.90773734], [ 1.23169368, -0.91975548], [ 1.14698704, -0.9362183], [ 1.07654746, -0.95726354], [ 1.02057534, -0.98302671], [ 0.97922023, -1.01364127], [ 0.95258022, -1.04923881], [1.0, -1.0]]
	path_bc = [[ 3.0, -1.], [ 3.22435566, -0.97780816], [ 3.43920927, -0.9584987],  [ 3.64344306, -0.94217732], [ 3.83599603, -0.9289503],  [ 4.01586948, -0.918924],   [ 4.18213222, -0.91220442], [ 4.33392543, -0.9088968],  [ 4.47046715, -0.90910529], [ 4.59105635, -0.91293264], [ 4.69507665, -0.92047998], [ 4.78199952, -0.93184666], [ 4.85138711, -0.94713013], [ 4.90289453, -0.96642586], [ 4.93627171, -0.9898274], [ 4.95136477, -1.01742644], [ 4.94811684, -1.04931296], [5.0, -1.0]]
	path_bd = [[ 3.0, -1.], [ 2.88602503, -1.18844841], [ 2.77447803, -1.37347181], [ 2.6653771,  -1.55407742], [ 2.55873833, -1.72929519], [ 2.4545714,  -1.89818036], [ 2.35287536, -2.05981599], [ 2.25363443, -2.21331531], [ 2.15681376, -2.35782397], [ 2.06235533, -2.49252214], [ 1.97017368, -2.6166263], [ 1.88015175, -2.72939094], [ 1.79213653, -2.83010996], [ 1.70593474, -2.91811779], [ 1.62130831, -2.99279023], [ 1.53796972, -3.053545], [ 1.45557725, -3.09984193], [ 1.37372991, -3.13118278], [ 1.2919622,  -3.14711075], [ 1.20973854, -3.14720945], [ 1.12644735, -3.13110161], [ 1.04139482, -3.0984472], [ 0.95379814, -3.04894118], [1.0, -3.0]]
	path_be = [[ 3.0, -1.], [ 2.99576749, -1.21327589], [ 2.99156891, -1.41977882], [ 2.98743628, -1.61833005], [ 2.98339959, -1.80779615], [ 2.9794867,  -1.98709555], [ 2.97572318, -2.15520468], [ 2.97213222, -2.31116383], [ 2.96873454, -2.45408261], [ 2.9655482,  -2.58314505], [ 2.96258854, -2.69761425], [ 2.95986808, -2.79683658], [ 2.95739636, -2.88024544], [ 2.95517987, -2.94736445], [ 2.95322195, -2.99781021], [ 2.95152268, -3.03129447], [ 2.9500788,  -3.04762575], [3.0, -3.0]]
	path_bf = [[ 3.0, -1.], [ 3.05262313, -1.13295336], [ 3.10357956, -1.26325768], [ 3.15359442, -1.39067955], [ 3.20337847, -1.51499112], [ 3.25363395, -1.63597215], [ 3.30506041, -1.75341204], [ 3.35836053, -1.86711181], [ 3.41424601, -1.97688612], [ 3.47344346, -2.08256518], [ 3.53670065, -2.1839967], [ 3.60479279, -2.28104788], [ 3.67852922, -2.37360729], [ 3.7587604,  -2.46158692], [ 3.84638532, -2.54492408], [ 3.94235941, -2.6235835], [ 4.04770301, -2.69755936], [ 4.16351046, -2.76687743], [ 4.29095991, -2.83159729], [ 4.43132403, -2.89181456], [ 4.58598152, -2.94766335], [ 4.75642971, -2.99931873], [ 4.94429833, -3.04699939], [5.0, -3.0]]

	path_dict = {}
	path_dict['AB'] = path_ab
	path_dict['AC'] = path_ab
	path_dict['AD'] = path_ab
	path_dict['AE'] = path_ab
	path_dict['AF'] = path_ab

	path_dict['BA'] = path_ab
	path_dict['BC'] = path_ab
	path_dict['BD'] = path_ab
	path_dict['BE'] = path_ab
	path_dict['BF'] = path_ab

	path_dict = add_offramps(path_dict)

	# Return the name and the list
	return path_dict

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


def quaternion_multiply(q0, q1):
    """
    Multiplies two quaternions.
    https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Quaternion-Fundamentals.html

    Input
    :param q0: A 4 element array containing the first quaternion (q01, q11, q21, q31)
    :param q1: A 4 element array containing the second quaternion (q02, q12, q22, q32)

    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

    """
    # Extract the values from q0
    w0 = q0[0]
    x0 = q0[1]
    y0 = q0[2]
    z0 = q0[3]

    # Extract the values from q1
    w1 = q1[0]
    x1 = q1[1]
    y1 = q1[2]
    z1 = q1[3]

    # Computer the product of the two quaternions, term by term
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
    return final_quaternion

# generate_vanilla_straight_line_paths_for_testing(goal_a, [goal_b, goal_c, goal_d, goal_e, goal_f])
# generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])

def setup_path_dict(path_title):
	all_paths = get_all_possible_path_names()
	# print(all_paths)
	# path_dict = {}
	# for name in all_paths:
	# 	path_dict[name] = None

	if path_title == 'null':
		path_dict = get_straight_line_paths()

	elif path_title == 'toptwo':
		path_dict = get_curvey_line_paths_1()

	elif path_title == 'bigger':
		path_dict = get_curvey_line_paths_1()

	elif path_title == 'early':
		path_dict = get_early_paths()

	elif path_title == 'late':
		path_dict = get_late_paths()

	elif path_title == 'even':
		path_dict = get_even_paths()
	
	elif path_title == 'obstacles_special':
		path_dict = get_obstacle_paths()

	elif path_title == 'obs':
		path_dict = get_obstacle_paths()
		return path_dict

	path_ab = path_dict['AB']
	path_ac = path_dict['AC']
	path_ad = path_dict['AD']
	path_ae = path_dict['AE']
	path_af = path_dict['AF']

	path_ba = path_dict['BA']
	path_bc = path_dict['BC']
	path_bd = path_dict['BD']
	path_be = path_dict['BE']
	path_bf = path_dict['BF']

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

		if len(path) > 0:
			if path[0] != start:
				print("Broken in " + key + " bad start")
				is_problem = True

			if path[-1] != end:
				print("Broken in " + key + " bad end")
				is_problem = True
		else:
			print("No path yet for \n\t" + path_title + " -> " + key)

	if is_problem:
		print("Problem in path transformations")
	else:
		print("All paths added and checked for reasonableness!")

	return path_dict

def export_path_dict(export_name, path_dict):
	directory_name = "paths/"

	for key in path_dict.keys():
		path = path_dict[key]

		csv_content = ""
		# line format for ROS is 

		for i in range(1, len(path)):
			p0 = path[i - 1]
			p1 = path[i]

			prev_x, prev_y 	= p0[:2]
			curr_x, curr_y 	= p1[:2]
			prev_z, curr_z 	= 0, 	0

			q1_inv 	= [0, 0, 0, 0]
			q2 		= [0, 0, 0, 0]

			# Here's an example to get the relative rotation 
			# from the previous robot pose to the current robot pose: 
			# http://wiki.ros.org/tf2/Tutorials/Quaternions
			q1_inv[0] = prev_x
			q1_inv[1] = prev_y
			q1_inv[2] = 0 #prev_pose.pose.orientation.z
			q1_inv[3] = -1 #-prev_pose.pose.orientation.w # Negate for inverse

			q2[0] = curr_x
			q2[1] = curr_y
			q2[2] = 0 #current_pose.pose.orientation.z
			q2[3] = 1 #current_pose.pose.orientation.w
			
			qr = quaternion_multiply(q2, q1_inv)


			dX = curr_x - prev_x 
			dY = curr_y - prev_y
			dZ = 0 # since the robot doesn't float

			roll 	= 0
			yaw 	= np.arctan2(dY, dX)
			pitch 	= 0 #np.arctan2(np.sqrt(dZ * dZ + dX * dX), dY) + np.pi;

			# Create a rotation object from Euler angles specifying axes of rotation
			# (roll about an X-axis) / (subsequent pitch about the Y-axis) / (subsequent yaw about the Z-axis), 
			# rot = Rotation.from_euler('xyz', [0, 0, yaw], degrees=False)

			# # Convert to quaternions and print
			# rot_quat = rot.as_quat()

			x, y, z = prev_x, prev_y, 0
			qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
			qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
			qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
			qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
			# qx, qy, qz, qw = rot_quat

			csv_content += str(x) + ", " + str(y) + ", " + str(z) + ", " + str(qx) + ", " + str(qy) + ", " + str(qz) + ", " + str(qw) + "\n"

		x, y, z = curr_x, curr_y, 0
		csv_content += str(x) + ", " + str(y) + ", " + str(z) + ", " + str(qx) + ", " + str(qy) + ", " + str(qz) + ", " + str(qw) + "\n"

		filename = directory_name + key + "-" + export_name + ".csv"
		if export_name == 'obs':
			filename = directory_name + key + ".csv"

		f = open(filename, "w")
		f.write(csv_content)
		f.close()
		# print("wrote out " + filename)


path_dict = setup_path_dict("null")
export_path_dict('null', path_dict)



# generate_vanilla_straight_line_paths_for_testing(goal_a, [goal_b, goal_c, goal_d, goal_e, goal_f])

# path_dict = setup_path_dict('bigger')
# export_name = 'bigger'

# path_dict1 = setup_path_dict('null')
# path_dict2 = setup_path_dict('toptwo')

if False:
	export_path_dict(export_name, path_dict)
	print("All exported to paths/")


inspect_path_set()




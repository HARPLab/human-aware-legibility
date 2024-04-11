from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

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
export_name = 'toptwo' #'null' #'toptwo'

def inspect_path_set():
	inspection_save_path = "study_paths/"

	early_dict 	= setup_path_dict('early')
	late_dict 	= setup_path_dict('late')
	even_dict	= setup_path_dict('even')

	obstacle_paths = get_obstacle_paths()

	##### Calculate the path lengths, also
	count_path_lengths(inspection_save_path, early_dict, late_dict, even_dict, obstacle_paths)

	##### draw them in groups by the path segment
	draw_paths_by_segment(inspection_save_path, early_dict, late_dict, even_dict)

	##### draw them in groups by the early/late/etc/group
	draw_paths_by_dict(inspection_save_path, early_dict, late_dict, even_dict)

	##### special drawings for obstacle avoidance
	draw_obstacle_sets(inspection_save_path, early_dict, late_dict, even_dict, obstacle_paths)

def get_xy_from_path(path):
	if len(path) == 0:
		return [], []

	x_list, y_list = list(map(list, zip(*path)))

	return x_list, y_list

def draw_paths_by_segment(inspection_save_path, early_dict, late_dict, even_dict):
	goal_list = [goal_a, goal_b, goal_c, goal_d, goal_e, goal_f]
	goal_colors = ['red', 'blue', 'purple', 'green', 'orange', 'pink']

	plt.figure(figsize=(5, 4))

	buffer = 2
	ax = plt.gca()
	# ax.set_xlim([goal_a[0] - buffer, goal_c[0] + buffer])
	# ax.set_ylim([goal_d[1] - buffer, goal_a[1] + buffer])
	ax.set_aspect('equal')

	for key in ['AD']: #early_dict.keys():
		path_early 	= early_dict[key]
		path_late 	= late_dict[key]
		path_even 	= even_dict[key]

		early_x, early_y 	= get_xy_from_path(path_early)
		late_x, late_y 		= get_xy_from_path(path_late)
		even_x, even_y 		= get_xy_from_path(path_even)

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


def draw_paths_by_dict(inspection_save_path, early_dict, late_dict, even_dict):
	pass

def draw_obstacle_sets(inspection_save_path, early_dict, late_dict, even_dict, obstacle_paths):
	pass

def count_path_lengths(inspection_save_path, early_dict, late_dict, even_dict, obstacle_paths):
	# Create a csv of the lengths for each path
	pass


def get_early_paths():
	tag = 'early'

	path_ab = []
	path_ac = []
	path_ad = []
	path_ae = []
	path_af = []

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	path_ba = []
	path_bc = []
	path_bd = []
	path_be = []
	path_bf = []

	path_dict = {}
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

	# Return the name and the list
	return path_dict

def get_late_paths():
	path_ab = []
	path_ac = []
	path_ad = [[ 1., -1.], [ 0.43814402, -1.49705433], [-0.01379929, -1.87088852], [-0.35609953, -2.16668864], [-0.59500778, -2.39667103], [-0.74460785, -2.58158456], [-0.81344605, -2.73129475], [-0.80554059, -2.84918306], [-0.72294926, -2.93450258], [-0.57059295, -2.98493332], [-0.36167342, -3.00075196], [-0.16288852, -3.03459423], [ 0.05345099, -3.05153597], [ 0.27634019, -3.05724956], [ 0.50081657, -3.05673042], [ 0.72551991, -3.05297824], [ 0.95003615, -3.04740445], [1.0, -3]]
	path_ae = []
	path_af = []

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	path_ba = []
	path_bc = []
	path_bd = []
	path_be = []
	path_bf = []

	path_dict = {}
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

	# Return the name and the list
	return path_dict

def get_even_paths():
	path_ab = []
	path_ac = []
	path_ad = [[ 1., -1.], [ 0.88768633, -1.15630132], [ 0.80524659, -1.29759949], [ 0.74658819, -1.43105059], [ 0.70474532, -1.5614607], [ 0.67549602, -1.69076969], [ 0.65620035, -1.81981969], [ 0.64518372, -1.94896821], [ 0.6414097,  -2.07834285], [ 0.64430734, -2.20795136], [ 0.65369887, -2.33772252], [ 0.66981571, -2.46750169], [ 0.69343503, -2.59698916], [ 0.72626452, -2.72553795], [ 0.77202397, -2.85143547], [ 0.84019242, -2.96828326], [ 0.94936774, -3.04021608], [1.0, -3]]
	path_ae = []
	path_af = []

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	path_ba = []
	path_bc = []
	path_bd = []
	path_be = []
	path_bf = []

	path_dict = {}
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

	# Return the name and the list
	return path_dict

def get_obstacle_paths():
	path_ac = []
	path_af = []

	# # generate_vanilla_straight_line_paths_for_testing(goal_b, [goal_a, goal_c, goal_d, goal_e, goal_f])
	# Return the name and the list
	return path_ac, path_af

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
	path_dict['AC'] = path_ab
	path_dict['AD'] = path_ab
	path_dict['AE'] = path_ab
	path_dict['AF'] = path_ab

	path_dict['BA'] = path_ab
	path_dict['BC'] = path_ab
	path_dict['BD'] = path_ab
	path_dict['BE'] = path_ab
	path_dict['BF'] = path_ab

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
		print(path)
		print(p)


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

			x, y, z = curr_x, curr_y, 0
			qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
			qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
			qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
			qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
			# qx, qy, qz, qw = rot_quat

			csv_content += str(x) + ", " + str(y) + ", " + str(z) + ", " + str(qx) + ", " + str(qy) + ", " + str(qz) + ", " + str(qw) + "\n"


		filename = directory_name + key + "-" + export_name + ".csv"
		f = open(filename, "w")
		f.write(csv_content)
		f.close()
		print("wrote out " + filename)





# generate_vanilla_straight_line_paths_for_testing(goal_a, [goal_b, goal_c, goal_d, goal_e, goal_f])

# path_dict = setup_path_dict('bigger')
# export_name = 'bigger'

# path_dict1 = setup_path_dict('null')
# path_dict2 = setup_path_dict('toptwo')

if False:
	export_path_dict(export_name, path_dict)
	print("All exported to paths/")

if True:
	inspect_path_set()




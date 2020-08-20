import numpy as np
import math
import random
import copy
import cv2
import pickle
import seaborn as sns
import matplotlib.pylab as plt
import sys
from PIL import Image
from PIL import ImageDraw

from shapely.geometry import Point as fancyPoint
from shapely.geometry import Polygon as fancyPolygon

# import custom libraries from PythonRobotics
sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/ModelPredictiveTrajectoryGenerator/')
sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/StateLatticePlanner/')

# print(sys.path)

import state_lattice_planner as slp
import model_predictive_trajectory_generator as mptj
from collections import defaultdict

OPTION_SHOW_VISIBILITY = True
OPTION_FORCE_GENERATE_VISIBILITY = False
OPTION_FORCE_GENERATE_OBSTACLE_MAP = True
OPTION_EXPORT = True

EXPORT_MODE_JSON = 1
EXPORT_MODE_CSV = 2
OPTION_EXPORT_MODE = EXPORT_MODE_CSV


# window dimensions
length = 409
width = 451
pixels_to_foot = 20

resolution_visibility = 1
resolution_planning = 30

# divide by two because these are radiuses
DIM_TABLE_RADIUS = int(1 * pixels_to_foot / 2.0)
DIM_OBSERVER_RADIUS = int(1.5 * pixels_to_foot / 2.0)
DIM_ROBOT_RADIUS = int(3 * pixels_to_foot / 2.0)
DIM_NAVIGATION_BUFFER = int(2.5 * pixels_to_foot)

num_tables = 6
num_observers = 6

# Choose the table layout for the scene
TYPE_PLOTTED = 0
TYPE_RANDOM = 1
TYPE_UNITY_ALIGNED = 2

generate_type = TYPE_UNITY_ALIGNED

# Color options for visualization
COLOR_TABLE = (32, 85, 230) #(235, 64, 52) 		# dark blue
COLOR_OBSERVER = (32, 85, 230) 		# dark orange
COLOR_FOCUS_AWAY = (52, 192, 235) 		# dark yellow
COLOR_PERIPHERAL_AWAY = (178, 221, 235) 	# light yellow
COLOR_FOCUS_TOWARDS = (235, 64, 52)		# dark yellow
COLOR_PERIPHERAL_TOWARDS = (55, 120, 191) 	# light yellow
COLOR_GOAL = (255, 255, 255) # (50, 168, 82) 			# green
COLOR_START = (100, 100, 100) 		# white

COLOR_OBSTACLE_CLEAR = (0, 0, 0)
COLOR_OBSTACLE_BUFFER = (100, 100, 100)
COLOR_OBSTACLE_FULL = (255, 255, 255)

VIS_INFO_RESOLUTION = -1
VIS_ALL = "VIS_ALL"
VIS_OMNI = "VIS_OMNI"
VIS_MULTI = "VIS_MULTI"
VIS_A = "VIS_A"
VIS_B = "VIS_B"


VIS_CHECKLIST = [VIS_OMNI, VIS_A, VIS_B, VIS_MULTI]
PATH_COLORS = [(138,43,226), (0,255,255), (255,64,64), (0,201,87)]
PATH_LABELS = ['red', 'yellow', 'blue', 'green']
# PATH_COLORS = [(130, 95, 135), (254, 179, 8), (55, 120, 191), (123, 178, 116)]

VIS_COLOR_MAP = {}
for i in range(len(PATH_COLORS)):
	VIS_COLOR_MAP[VIS_CHECKLIST[i]] = PATH_COLORS[i]

COLOR_P_BACK 	= VIS_COLOR_MAP[VIS_A]
COLOR_P_FACING 	= VIS_COLOR_MAP[VIS_B]

goals = []
tables = []
observers = []
start = []
path = []


#lookup tables linking related objects 
goal_observers = {}
goal_obj_set = {}

visibility_maps = {}

SCENARIO_IDENTIFIER = "new_scenario"

FILENAME_OUTPUTS = 'generated/'
FILENAME_PICKLE_VIS = FILENAME_OUTPUTS + 'pickled_visibility'
FILENAME_PICKLE_OBSTACLES = FILENAME_OUTPUTS + 'pickled_obstacles'
FILENAME_VIS_PREFIX = FILENAME_OUTPUTS + "fine_fig_vis_"
FILENAME_OVERVIEW_PREFIX = FILENAME_OUTPUTS + "overview_"
FILENAME_OBSTACLE_PREFIX = FILENAME_OUTPUTS + "fig_obstacles"

FILENAME_TO_UNITY = "export/"
FILENAME_EXPORT_IMGS_PREFIX = FILENAME_TO_UNITY + "imgs/"
FILENAME_EXPORT_CSV_PREFIX = FILENAME_TO_UNITY + "csv/"

# visibility = np.zeros((r_width, r_length))
# for x in range(r_width):
# 	for y in range(r_length):
# 		rx = x*resolution_visibility
# 		ry = y*resolution_visibility
# 		score = 0
# 		for obs in observers:
# 			score += obs.get_visibility((rx,ry))

# 		visibility[x,y] = score

# visibility = visibility.T

def dist(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def bresenham_line(xy1, xy2):
	x0, y0 = xy1
	x1, y1 = xy2

	steep = abs(y1 - y0) > abs(x1 - x0)
	if steep:
		x0, y0 = y0, x0  
		x1, y1 = y1, x1

	switched = False
	if x0 > x1:
		switched = True
		x0, x1 = x1, x0
		y0, y1 = y1, y0

	if y0 < y1: 
		ystep = 1
	else:
		ystep = -1

	deltax = x1 - x0
	deltay = abs(y1 - y0)
	error = -deltax / 2
	y = y0

	line = []	
	for x in range(x0, x1 + 1):
		if steep:
			line.append((y,x))
		else:
			line.append((x,y))

		error = error + deltay
		if error > 0:
			y = y + ystep
			error = error - deltax
	if switched:
		line.reverse()
	return line

def in_bounds(point):
	x, y = point

	if x > width or x < 0:
		return False

	if y > length or y < 0:
		return False

	return True

def get_cost_of_segment(pos1, pos2, obstacle_map, visibility_map):
	# time in visibility
	
	return cost

def get_cost_of_move(pos1, pos2, obstacle_map, visibility_map):
	cost = 0
	most_vis = np.amax(visibility_map)
	obstacle_max = np.amax(obstacle_map)
	obstacle_min = np.amin(obstacle_map)

	# print(obstacle_map[0][0])

	# print(most_vis)
	# print(visibility_map[0][0])
	# print(obstacle_min)
	# print(obstacle_max)
	# exit()

	line = bresenham_line(pos1, pos2)
	# print(line) 

	for point in line:
		# infinite cost to go through obstacles
		min_val = 1
		if in_bounds(point):
			# cost += 1 + (obstacle_map[point] * 500 * 0)
			vis_x = int(point[0] / resolution_visibility)
			vis_y = int(point[1] / resolution_visibility)
			# print(visibility_map.shape)
			# print(vis_x, vis_y)
			vis = visibility_map[vis_y][vis_x]
			cost_increase = 1
			# cost_increase = ((most_vis - vis) / most_vis)
			cost += cost_increase
			
			if cost_increase < min_val:
				min_val = cost_increase


			# print((vis_x, vis_y, vis, obstacle_map[point]))

			# print("point info")
			# print((vis, obstacle_map[point], cost))

			if math.isnan(cost):
				exit()

		else:
			cost += np.Inf

			# Give a bonus for being at least partially well observed in this segment
	# cost += len(line)*min_val

	return cost


def get_children_astar(node, obstacle_map, visibility_map):
	xyz, parent_node, cost_so_far = node
	children = []
	directions = [(0, resolution_planning), (resolution_planning, 0), (0, -1*resolution_planning), (-1*resolution_planning, 0)]

	for direction in directions:
		new_xyz = tuple_plus(xyz, direction)
		cost_to_add = get_cost_of_move(xyz, new_xyz, obstacle_map, visibility_map)

		# print(cost_to_add)

		new_node = (new_xyz, node, cost_so_far + cost_to_add)
		children.append(new_node)


	return children


def heuristic(node, goal, visibility_map):
	pos1 = node[0]
	pos2 = goal

	most_vis = np.amax(visibility_map)
	ratio = (1 / most_vis)
	len(bresenham_line(pos1, pos2)) * ratio

	return len(bresenham_line(pos1, pos2))

def get_paths_astar(start, goal, obstacle_map, visibility_maps):
	paths = []

	obstacle_map = obstacle_map
	vis_maps = []
	# vis_maps.append(visibility_maps[VIS_ALL][0])
	vis_maps.append(visibility_maps[VIS_OMNI][0])
	vis_maps.append(visibility_maps[VIS_MULTI][0])
	vis_maps.append(visibility_maps[VIS_INDIVIDUALS][0])
	# vis_maps.append(visibility_maps[VIS_INDIVIDUALS][1])

	print(len(vis_maps))
	
	for visibility_map in vis_maps:
		print("math path being added")

		# node = ((x,y), parent_node, cost-so-far)

		n_0 = (start, None, 0)

		parent_grid = {}

		openset = [n_0]
		closed_set = defaultdict(lambda: float('inf'))

		i = 0
		goal_found = False
		while i < 600000 and not goal_found:
			i += 1
			openset = sorted(openset, key=lambda x: x[2] + heuristic(x, goal, visibility_map))
			# print(openset)
			# Get best bet node
			node = openset.pop(0)
			xyz, parent, cost = node
			closed_set[xyz] = (cost, parent)

			# print(len(closed_set))
			
			kids = get_children_astar(node, obstacle_map, visibility_map)
			openset.extend(kids)
			# print(len(openset))

			print(dist(xyz, goal))

			if dist(xyz, goal) < resolution_planning:
				print("Found the goal!")
				goal_found = True

				path = []
				n_path = node
				while n_path[1] is not None:
					n_path = n_path[1]
					path.append(n_path[0])
				
				print(path)
				paths.append(path[::-1])
		
	return paths

# Bezier helpers
def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result

def get_path_spoof(start, end, goal_pts, table_pts, vis_type, visibility_map):
	STEPSIZE = 15
	points = []

	goal_index = goal_pts.index(end)
	helper = goal_helper_pts[goal_index]	

	if vis_type == VIS_OMNI:
		xys = [start, end]

	elif vis_type == VIS_MULTI:
		xys = [start, (1008, 420), end]

	elif vis_type == VIS_A:
		xys = [start, (1008, 350), (1008, 350), end]

	elif vis_type == VIS_B:
		xys = [start, (822, 370), (822, 370), end]

	ts = [t/STEPSIZE for t in range(STEPSIZE + 1)]
	bezier = make_bezier(xys)
	points = bezier(ts)

	points = [(int(px), int(py)) for px, py in points]

	return points


def get_path(start, end, obs=[]):
	path = [start, end]
	x_start, y_start = start
	x_end, y_end = end

	return path

def get_path_2(start, end, obs=[]):
	path = [start, end]
	x_start, y_start = start
	x_end, y_end = end

	x_start = x_start / 10.0
	y_start = y_start / 10.0

	x_end = x_end / 10.0
	y_end = y_end / 10.0

	target = mptj.motion_model.State(x=x_end, y=x_end, yaw=np.deg2rad(90.0))
	k0 = 0.0

	init_p = np.array([x_start, y_start, 0.0]).reshape(3, 1)
	x, y, yaw, p = mptj.optimize_trajectory(target, k0, init_p)

	# print(x)
	# print(y)
	# print(yaw)
	# print(p)
	return path


def tuple_plus(a, b):
	return (int(a[0] + b[0]), int(a[1] + b[1]))

def angle_between(p1, p2):
	ang1 = np.arctan2(*p1[::-1])
	ang2 = np.arctan2(*p2[::-1])
	return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	angle = math.radians(angle)

	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return (qx, qy)

def get_random_point_in_room(length, width):
	new_goal = (random.randrange(width), random.randrange(length))
	# new_goal = Point(new_goal)

	return new_goal

class Point: 
	def __init__(self, xyz):
		self.xyz = xyz

	def get_tuple(self):
		return self.xyz

	def get_x(self):
		return self.xyz[0]

	def get_y(self):
		return self.xyz[1]

class Polygon:
	def __init__(self, pt_list):
		self.points = pt_list

class Table:
	radius = DIM_TABLE_RADIUS

	def __init__(self, pt, gen_type):
		self.location = pt

		if gen_type == TYPE_UNITY_ALIGNED:
			self.radius = 1
		
		half_radius = self.radius / 2.0

		pts = [pt, pt, pt, pt]
		pts[0] = tuple_plus(pts[0], (- half_radius, - half_radius))
		pts[1] = tuple_plus(pts[1], (- half_radius, + half_radius))
		pts[2] = tuple_plus(pts[2], (+ half_radius, + half_radius))
		pts[3] = tuple_plus(pts[3], (+ half_radius, - half_radius))
		self.points = pts

		polygon_set = []
		for p in pts:
			polygon_set.append(Point(p))

		self.shape = Polygon(polygon_set)

	def is_within(self, point):
		return Point(point).within(self.shape)

	def pt_top_left(self):
		return self.points[0]

	def pt_bottom_right(self):
		return self.points[2]

	def get_radius(self):
		return int(self.radius)

	def get_center(self):
		return self.location

	def get_JSON(self):
		return (self.location, self.radius)



class Observer:
	entity_radius = DIM_OBSERVER_RADIUS
	draw_depth = 25000
	cone_depth = 2000
	focus_angle = 60 / 2.0
	peripheral_angle = 160 / 2.0

	orientation = 0

	field_focus = []
	field_peripheral = []


	def __init__(self, location, angle):
		# print("LOCATION")
		# print(location)
		self.location = location
		self.orientation = angle

		focus_angle = self.focus_angle
		peripheral_angle = self.peripheral_angle
		
		# Add center of viewpoint
		focus_a = (0, self.cone_depth)
		focus_b = (0, self.cone_depth)
		periph_a = (0, int(self.cone_depth*1.5))
		periph_b = (0, int(self.cone_depth*1.5))

		focus_a = rotate((0,0), focus_a, focus_angle + angle)
		focus_b = rotate((0,0), focus_b, -focus_angle + angle)

		periph_a = rotate((0,0), periph_a, peripheral_angle + angle)
		periph_b = rotate((0,0), periph_b, -peripheral_angle + angle)
	
		focus_a = tuple_plus(focus_a, location)
		focus_b = tuple_plus(focus_b, location)

		periph_a = tuple_plus(periph_a, location)
		periph_b = tuple_plus(periph_b, location)

		self.field_focus = [location, focus_a, focus_b]
		self.field_peripheral = [location, periph_a, periph_b]

		# Add center of viewpoint
		draw_focus_a = (0, self.draw_depth)
		draw_focus_b = (0, self.draw_depth)
		draw_periph_a = (0, int(self.draw_depth*5))
		draw_periph_b = (0, int(self.draw_depth*5))

		draw_focus_a = rotate((0,0), draw_focus_a, focus_angle + angle)
		draw_focus_b = rotate((0,0), draw_focus_b, -focus_angle + angle)

		draw_periph_a = rotate((0,0), draw_periph_a, peripheral_angle + angle)
		draw_periph_b = rotate((0,0), draw_periph_b, -peripheral_angle + angle)
	
		draw_focus_a = tuple_plus(draw_focus_a, location)
		draw_focus_b = tuple_plus(draw_focus_b, location)

		draw_periph_a = tuple_plus(draw_periph_a, location)
		draw_periph_b = tuple_plus(draw_periph_b, location)

		self.draw_field_focus = [location, draw_focus_a, draw_focus_b]
		self.draw_field_peripheral = [location, draw_periph_a, draw_periph_b]

	def get_visibility(self, location):
		# print(location)
		fancy_location = fancyPoint(location)

		field_focus = fancyPolygon(self.field_focus)
		field_peripheral = fancyPolygon(self.field_peripheral)

		if fancy_location.within(field_focus):
			return 1
		elif fancy_location.within(field_peripheral):
			return .5
		return 0

	def get_center(self):
		return self.location

	def get_field_focus(self):
		return np.int32([self.field_focus])

	def get_field_peripheral(self):
		return np.int32([self.field_peripheral])

	def get_draw_field_focus(self):
		return np.int32([self.draw_field_focus])

	def get_draw_field_peripheral(self):
		return np.int32([self.draw_field_peripheral])

	def get_radius(self):
		return int(self.entity_radius)

UNITY_CORNERS = [(1.23, 3.05), (11.22, -10.7)]
ux1, uy1 = UNITY_CORNERS[0]
ux2, uy2 = UNITY_CORNERS[1]

IMG_CORNERS = [(0,0), (1000, 1375)]
ix1, iy1 = IMG_CORNERS[0]
ix2, iy2 = IMG_CORNERS[1]

UNITY_OFFSET_X = (ux1 - ix1)
UNITY_OFFSET_Y = (uy1 - iy1)
UNITY_SCALE_X = (ix2 - ix1) / (ux2 - ux1)
UNITY_SCALE_Y = (iy2 - iy1) / (uy2 - uy1)

length = ix2
width = iy2

# print(length)
# print(width)

# print("offsets")
# print(UNITY_OFFSET_X)
# print(UNITY_OFFSET_Y)
# print("scales")
# print(UNITY_SCALE_X)
# print(UNITY_SCALE_Y)

UNITY_TO_IRL_SCALE = 3
# bottom right corner in unity
# UNITY_OFFSET_X = -1.23
# UNITY_OFFSET_Y = 3.05


def unity_to_image(pt):
	x, y = pt
	nx = (x - UNITY_OFFSET_X) * UNITY_SCALE_X
	ny = (y - UNITY_OFFSET_Y) * UNITY_SCALE_Y
	return (int(ny), int(nx))

def image_to_unity(pt):
	y, x = pt
	x = float(x)
	y = float(y)
	nx = (x / UNITY_SCALE_X) + UNITY_OFFSET_X
	ny = (y / UNITY_SCALE_Y) + UNITY_OFFSET_Y
	return (nx, ny)

def plan_to_image(pt):
	x, y = pt
	nx = x
	ny = y
	return (nx, ny)


# # TESTING SUITE FOR CONVERSIONS
# u_a = (11.22, 3.05) 
# u_b = (1.23, 3.05)
# u_c = (11.22, -10.7)
# u_d = (1.23, -10.7)

# u_pts = [u_a, u_b, u_c, u_d]
# i_pts = []

# print("TARGETS")
# print(u_pts)
# for pt in u_pts:
# 	ip = unity_to_image(pt)
# 	# print(str(pt) + "->" + str(ip))
# 	i_pts.append(ip)

# print(i_pts)
# n_u_pts = []
# for pt in i_pts:
# 	ip = image_to_unity(pt)
# 	# print(ip)
# 	n_u_pts.append(ip)

# print(n_u_pts)


# nodes = width x length divided up by planning resolution\
n_width = int(width / resolution_planning) + 1
n_length = int(length / resolution_planning) + 1

goal_helper_pts = []

if generate_type == TYPE_PLOTTED:
	SCENARIO_IDENTIFIER = "3x2_all_full"
	start = (10, 10)

	# for i in range(num_tables):
	# 	new_goal = get_random_point_in_room(length, width)
	# 	goals.append(new_goal)

	# goal = goals[0]

	row1 = 60
	row2 = 360

	col1 = 100
	col2 = 300
	col3 = 500

	start = (col1 - 30, int((row1 + row2) / 2))

	table_pts = [(col1,row1), (col2,row1), (col3, row1), (col1,row2), (col2,row2), (col3, row2)]

	for pt in table_pts:
		table = Table(pt, generate_type)
		tables.append(table)

	for table in tables:
		obs1_pt = table.get_center()
		obs1_pt = tuple_plus(obs1_pt, (-60, 0))
		obs1_angle = 270
		obs1 = Observer(obs1_pt, obs1_angle)
		observers.append(obs1)


		obs2_pt = table.get_center()
		obs2_pt = tuple_plus(obs2_pt, (60, 0))
		obs2_angle = 90
		obs2 = Observer(obs2_pt, obs2_angle)
		observers.append(obs2)

		goal_pt = table.get_center()
		offset = (0,0)
		if (table.get_center()[1] == row1):
			offset = (0, 80)
		else: 
			offset = (0, -80)

		goal_pt = tuple_plus(goal_pt, offset)
		goal_angle = 0
		goals.append(goal_pt)

		goal_observers[goal_pt] = [obs1, obs2]

elif generate_type == TYPE_UNITY_ALIGNED:
	SCENARIO_IDENTIFIER = "_unity_v1_"
	# images will be made at the scale of
	
	# x1 = 3.05
	# x2 = -10.7
	# y1 = 11.22
	# y2 = 1.23

	# length = abs(y1 - y2)
	# width = abs(x1 - x2)

	# start = (7.4, 2.37)
	start = (6.0, 2.0)
	start = unity_to_image(start)

	length = 1000
	width = 1375

	waypoint = (6.0, 2.0)
	waypoint = unity_to_image(waypoint)

	unity_goal_pt = (4.43, -7.0)

	unity_table_pts = []
	unity_table_pts.append((3.6, -4.0))
	unity_table_pts.append((3.6, -7.0))
	unity_table_pts.append((5.6, -10.0))
	unity_table_pts.append((7.6, -7.0))

	unity_goal_stop_options = []
	unity_goal_stop_options.append((4.3, -4.3))
	unity_goal_stop_options.append((4.3, -7.3))
	unity_goal_stop_options.append((5.6, -9.3))
	unity_goal_stop_options.append((6.9, -7.3))

	unity_goal_options = []
	unity_goal_options.append((4.3, -4.0))
	unity_goal_options.append((4.429, -7.03)) #(4.3, -7.0)
	unity_goal_options.append((5.6, -9.3))
	unity_goal_options.append((6.9, -7.0))

	table_pts = []
	for t in unity_table_pts:
		table_pts.append(unity_to_image(t))

	goal_pts = []
	for g in unity_goal_stop_options:
		goal_helper_pts.append(unity_to_image(g))
		goal_pts.append(unity_to_image(g))

	goal = unity_to_image(unity_goal_pt)

	for pt in table_pts:
		table = Table(pt, generate_type)
		tables.append(table)

	# Set up observers
	obs1_pt = (3.50, -7.71)
	obs1_pt = unity_to_image(obs1_pt)

	obs1_angle = 55
	obs1 = Observer(obs1_pt, obs1_angle)
	observers.append(obs1)


	obs2_pt = (3.50, -6.37)
	obs2_pt = unity_to_image(obs2_pt)
	
	obs2_angle = 305
	obs2 = Observer(obs2_pt, obs2_angle)
	observers.append(obs2)

	goal_observers[goal] = [obs1, obs2]


elif generate_type == TYPE_RANDOM:
	random_id = ''.join([random.choice(string.ascii_letters 
			+ string.digits) for n in range(10)]) 
	SCENARIO_IDENTIFIER = "new_scenario_" + random_id

	start = get_random_point_in_room(length, width)

	for i in range(num_tables):
		new_goal = get_random_point_in_room(length, width)
		goals.append(new_goal)

	goal = goals[0]

	for i in range(num_tables):
		table_pt = get_random_point_in_room(length, width)
		
		table = Table(table_pt)
		tables.append(table)

	# add customer locations
	for i in range(num_observers):
		obs_loc = get_random_point_in_room(length, width)
		angle = random.randrange(360)
		print((obs_loc, angle))

		observers.append(Observer(obs_loc, angle))

else:
	print("Incorrect generate_type")

print("GOALS")
print(goals)

FILENAME_PICKLE_VIS += SCENARIO_IDENTIFIER
FILENAME_PICKLE_OBSTACLES += SCENARIO_IDENTIFIER
FILENAME_VIS_PREFIX += SCENARIO_IDENTIFIER
FILENAME_OBSTACLE_PREFIX += SCENARIO_IDENTIFIER
FILENAME_OVERVIEW_PREFIX += SCENARIO_IDENTIFIER

# Get paths
path = get_path(start, goal)

obstacle_vis = np.zeros((length,width,3), np.uint8)
obstacle_map = np.zeros((length,width), np.uint8)
# DRAW the environment

# Create a black image
img = np.zeros((length,width,3), np.uint8)
table_radius = int(.3 * UNITY_SCALE_X)
obs_radius = int(.125 * UNITY_SCALE_X)
goal_radius = int(.125 * UNITY_SCALE_X)
start_radius = int(.125 * UNITY_SCALE_X)

# observers = [observers[0]]
print(observers)
# for obs in observers:
if True:
	# Draw person
	# obs = observers[1]

	# cv2.circle(obstacle_vis, obs.get_center(), obs_radius, COLOR_OBSTACLE_BUFFER, obs_radius + DIM_NAVIGATION_BUFFER)
	# cv2.circle(obstacle_vis, obs.get_center(), obs_radius, COLOR_OBSTACLE_FULL, obs_radius)

	# cv2.circle(obstacle_map, obs.get_center(), obs.get_radius(), 1, obs.get_radius() + DIM_NAVIGATION_BUFFER)
	# cv2.circle(obstacle_map, obs.get_center(), obs.get_radius(), 1, obs.get_radius())

	# Draw shortened rep for view cones
	# cv2.fillPoly(img, obs.get_draw_field_peripheral(), COLOR_PERIPHERAL)
	# cv2.fillPoly(img, obs.get_draw_field_focus(), COLOR_FOCUS)

	overlay = img.copy()
	# Draw shortened rep for view cones
	# away = obs[1]
	obs = observers[1]
	# cv2.fillPoly(overlay, obs.get_draw_field_peripheral(), COLOR_PERIPHERAL_AWAY)
	cv2.fillPoly(overlay, obs.get_draw_field_focus(), COLOR_FOCUS_AWAY)
	alpha = 0.3  # Transparency factor.
	img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

	obs = observers[0]
	# cv2.fillPoly(overlay, obs.get_draw_field_peripheral(), COLOR_PERIPHERAL_TOWARDS)
	cv2.fillPoly(overlay, obs.get_draw_field_focus(), COLOR_FOCUS_TOWARDS)

	alpha = 0.3  # Transparency factor.
	img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
	# Following line overlays transparent rectangle over the image
	

	# cone = obs.get_draw_field_focus()[0]
	# print(cone)
	# cone0 = cone[0]
	# cone0 = (826, 248) #(cone0[0], cone0[1])
	# cone1 = cone[1]
	# cone0 = (826, 248) #(cone1[0], cone1[1])
	# cone2 = cone[2]
	# cone2 = (cone2[0], cone2[1])


	# print(cone0)
	# cv2.line(img, cone0, cone1, COLOR_FOCUS, thickness=2, lineType=8)
	# cv2.line(img, cone0, cone2, COLOR_FOCUS, thickness=2, lineType=8)

	# cone = obs.get_draw_field_peripheral()
	# cv2.line(img, cone[0], cone[1], COLOR_PERIPHERAL, thickness=2, lineType=8)
	# cv2.line(img, cone[0], cone[2], COLOR_PERIPHERAL, thickness=2, lineType=8)

# Draw tables
for table in tables:	
	table_center = table.get_center()


	cv2.circle(img, table_center, table_radius, COLOR_TABLE, table_radius)

	cv2.circle(obstacle_vis, table_center, table_radius + DIM_NAVIGATION_BUFFER, COLOR_OBSTACLE_BUFFER)
	cv2.circle(obstacle_vis, table_center, table_radius, COLOR_OBSTACLE_FULL)

	# cv2.rectangle(obstacle_map, table.pt_top_left(), table.pt_bottom_right(), 1, table.get_radius() + DIM_NAVIGATION_BUFFER)
	# cv2.rectangle(obstacle_map, table.pt_top_left(), table.pt_bottom_right(), 1, table.get_radius())


obs = observers[1]
cv2.circle(img, obs.get_center(), obs_radius, COLOR_P_BACK, obs_radius)
obs = observers[0]
cv2.circle(img, obs.get_center(), obs_radius, COLOR_P_FACING, obs_radius)
# Draw table.gets
# Draw observer cones


	# cv2.polylines(img, obs.get_draw_field_peripheral(), COLOR_PERIPHERAL)
	# cv2.polylines(img, obs.get_draw_field_focus(), COLOR_FOCUS)


for goal in goal_pts:
	# Draw person
	print(goal)
	cv2.circle(img, goal, goal_radius, COLOR_GOAL, goal_radius)

# print(start)
cv2.circle(img, start, start_radius, COLOR_START, start_radius)

obstacle_map = cv2.cvtColor(obstacle_vis, cv2.COLOR_BGR2GRAY)
(thresh, obstacle_map) = cv2.threshold(obstacle_map, 1, 255, cv2.THRESH_BINARY)

obstacles = {}
obstacles['map'] = copy.copy(obstacle_map)
obstacles['vis'] = copy.copy(obstacle_vis)
cv2.imwrite(FILENAME_OBSTACLE_PREFIX + '_map.png', obstacle_map) 
cv2.imwrite(FILENAME_OBSTACLE_PREFIX + '_vis.png', obstacle_vis) 

cv2.imwrite(FILENAME_OVERVIEW_PREFIX + ".png", img)
print("Exported overview pic without paths")
# ax = sns.heatmap(obstacle_map).set_title("Obstacle map of restaurant")
# plt.savefig()
# plt.clf()

dbfile = open(FILENAME_PICKLE_OBSTACLES, 'ab') 
pickle.dump(obstacles, dbfile)					  
dbfile.close()
print("Saved obstacle maps")


print("Importing pickle of obstacle info")
dbfile = open(FILENAME_PICKLE_OBSTACLES, 'rb')
obstacles = pickle.load(dbfile)
obstacle_map = obstacles['map']
obstacle_vis = obstacles['vis']
dbfile.close() 

# VISIBILITY Unit TEST
if False:
	score = 0
	rx, ry = (133, 232)
	for obs in observers:
		this_vis = obs.get_visibility((rx,ry))
		print(this_vis)
		score += this_vis

	print(score)

	print("OR")
	score = 0
	rx, ry = (233, 133)
	for obs in observers:
		this_vis = obs.get_visibility((rx,ry))
		print(this_vis)
		score += this_vis

	print(score)


if OPTION_FORCE_GENERATE_VISIBILITY:
	# visibility_maps[VIS_INFO_RESOLUTION] = resolution_visibility

	r_width = int(width / resolution_visibility)
	r_length = int(length / resolution_visibility)

	visibility = np.zeros((r_width, r_length))
	omni = copy.copy(visibility)
	omni = omni.T

	ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: Omniscient")
	visibility_maps[VIS_OMNI] = [copy.copy(visibility)]
	# print(visibility.shape)
	plt.savefig(FILENAME_VIS_PREFIX + '_omni.png')
	plt.clf()

	# DEPRECATED: SAVING AN ALL VERSION
	# for x in range(r_width):
	# 	for y in range(r_length):
	# 		rx = x*resolution_visibility
	# 		ry = y*resolution_visibility
	# 		score = 0
	# 		for obs in observers:
	# 			score += obs.get_visibility((rx,ry))

	# 		visibility[x,y] = score

	# visibility = visibility.T
	# # xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
	# ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: All")
	# visibility_maps[VIS_ALL] = [copy.copy(visibility)]
	# plt.savefig(FILENAME_VIS_PREFIX + '_all.png')
	# plt.clf()
	# plt.show()

	visibility = np.zeros((r_width, r_length))
	for x in range(r_width):
		for y in range(r_length):
			rx = x*resolution_visibility
			ry = y*resolution_visibility
			score = 0
			for obs in observers:
				score += obs.get_visibility((rx,ry))

			visibility[x,y] = score

	visibility = visibility.T
	# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
	ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: Both Perspectives")
	# plt.show()
	visibility_maps[VIS_MULTI] = [copy.copy(visibility)]
	plt.savefig(FILENAME_VIS_PREFIX + '_multi.png')
	plt.clf()

	print("generated multi vis")

	indic = 0
	# visibility_maps[VIS_INDIVIDUALS] = []
	solo_paths = []

	for obs in observers:
		visibility = np.zeros((r_width, r_length))
		for x in range(r_width):
			for y in range(r_length):
				rx = x*resolution_visibility
				ry = y*resolution_visibility
				score = 0
				score += obs.get_visibility((rx,ry))

				visibility[x,y] = score

		visibility = visibility.T
		# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
		ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: 1 Observer #" + str(indic))
		# plt.show()
		solo_paths.append(copy.copy(visibility))
		# visibility_maps[VIS_INDIVIDUALS].append(copy.copy(visibility))
		plt.savefig(FILENAME_VIS_PREFIX + '_person_' + str(indic) + '.png')
		print("generated person vis " + str(indic))
		indic += 1
		plt.clf()

	# visibility_maps[VIS_INDIVIDUALS] = solo_paths
	visibility_maps[VIS_A] = solo_paths[0]
	visibility_maps[VIS_B] = solo_paths[1]

	# Export the new visibility maps and resolution info
	dbfile = open(FILENAME_PICKLE_VIS, 'ab') 
	pickle.dump(visibility_maps, dbfile)					  
	dbfile.close()
	# Successfully dumped pickle


obs_i = cv2.flip(img, 0)
cv2.imwrite(FILENAME_EXPORT_IMGS_PREFIX + 'plain_o'+ '.png', obs_i) 
# cv2.imwrite(FILENAME_EXPORT_IMGS_PREFIX + 'plain_' + path_title + '.png', path_img) 


# Import the visibility info for work
# print("Importing pickle of visibility info")
# dbfile = open(FILENAME_PICKLE_VIS, 'rb')	  
# visibility_maps = pickle.load(dbfile)
# resolution_visibility = visibility_maps[VIS_INFO_RESOLUTION]
# print("Found maps at resolution " + str(resolution_visibility))
# dbfile.close() 

# # Raw coded paths
# # more vis
# p1 = [(490, 270), (490, 240), (490, 210), (460, 210), (430, 210), (400, 210), (370, 210), (340, 210), (310, 210), (280, 210), (250, 210), (220, 210), (190, 210), (160, 210), (130, 210), (100, 210), (70, 210)]
# # less vis
# p2 = [(490, 270),(490, 240),(460, 240),(460, 210),(430, 210),(400, 210),(370, 210),(340, 210),(310, 210),(280, 210),(250, 210),(220, 210),(190, 210),(160, 210),(130, 210),(100, 210),(70, 210)]
# p_names = ["first", "second"]

#TODO get real path with ASTAR or similar
# paths = []
# for visibility_map in visibility_maps:
# 	print("a map")
# 	for path in paths:
# 		print("this path")
# 		cost = 0
# 		for i in range(len(path) - 1):
# 			pos1 = path[i]
# 			pos2 = path[i + 1]
# 			cost += get_cost_of_move(pos1, pos2, obstacle_map, visibility_map)

# 		print("Cost = " + str(cost))

# paths = get_paths_astar(start, goal, obstacle_map, visibility_maps)
#p = [(490, 270), (490, 240), (490, 210), (460, 210), (430, 210), (400, 210), (370, 210), (340, 210), (310, 210), (280, 210), (250, 210), (220, 210), (190, 210), (160, 210), (130, 210), (100, 210), (70, 210)]
# p = [(490, 270),(490, 240),(460, 240),(460, 210),(430, 210),(400, 210),(370, 210),(340, 210),(310, 210),(280, 210),(250, 210),(220, 210),(190, 210),(160, 210),(130, 210),(100, 210),(70, 210)]

saved_paths = {}

# All the types there are
for vis_type in VIS_CHECKLIST:
	# print(visibility_maps.keys())
	# print(vis_type)

	# UPDATE TO BETTER MAPS
	# vis_map = visibility_maps[vis_type]
	# vis_map = visibility_maps[2][0]
	vis_map = None

	for goal_index in range(len(goal_pts)):
		end = goal_pts[goal_index]
		# print(str(goal_index) + "->" + str(end))

		pkey = vis_type + "-" + str(goal_index)

		new_path = get_path_spoof(start, end, goal_pts, table_pts, vis_type, vis_map)
		
		saved_paths[pkey] = new_path




# DISPLAY PATHS CODE
path_titles = ["OMNI", "TABLE", "Person A", "Person B"]

# omni_paths_img = img.copy()
# cv2.imwrite('generated/fig_path_' + "OMNISCIENT" + '.png', omni_paths_img) 

all_paths_img = img.copy()

img_deck = {}
for vis_type in VIS_CHECKLIST:
	type_img = img.copy()
	img_deck[vis_type] = type_img

for i in range(len(goal_pts)):
	type_img = img.copy()
	img_deck[str(i)] = type_img


for pkey in saved_paths.keys():
	path = saved_paths[pkey]
	path_img = img.copy()
	path_title = pkey

	vis_type, goal_index = pkey.split("-")
	color = VIS_COLOR_MAP[vis_type]
	by_method = img_deck[vis_type]
	by_goal = img_deck[goal_index]

	# Draw the path  
	for i in range(len(path) - 1):
		a = path[i]
		b = path[i + 1]
		
		cv2.line(path_img, a, b, color, thickness=6, lineType=8)
		cv2.line(all_paths_img, a, b, color, thickness=6, lineType=8)

		cv2.line(by_method, a, b, color, thickness=6, lineType=8)
		cv2.line(by_goal, a, b, color, thickness=6, lineType=8)		

	path_img = cv2.flip(path_img, 0)
	cv2.imwrite(FILENAME_EXPORT_IMGS_PREFIX + 'fig_path_' + path_title + '.png', path_img) 

all_paths_img = cv2.flip(all_paths_img, 0)
cv2.imwrite(FILENAME_EXPORT_IMGS_PREFIX + 'ALL_CONDITIONS' + '.png', all_paths_img) 
### END DISPLAY PATHS CODE

for key in img_deck.keys():
	this_img = img_deck[key]
	this_img = cv2.flip(this_img, 0)
	cv2.imwrite(FILENAME_EXPORT_IMGS_PREFIX + 'total_' + key + '.png', this_img) 


if OPTION_EXPORT:
	import json 

	if OPTION_EXPORT_MODE == EXPORT_MODE_JSON:
		# Ready JSON Export
		table_json = []
		for table in tables:
			table_json.append(table.get_JSON())


		print("Exporting path")
		file1 = open("generated/path_v1.txt","a")
		data = {}
		data['path'] = unity_path
		data['start'] = start
		data['goals'] = goals
		data['tables'] = table_json
		data['server_type'] = "SERVER_HUMAN"
		data['condition'] = "naive"

		# print(json.dumps(data, indent=4))

		file1.write(json.dumps(data, indent=4)) 
	
	elif OPTION_EXPORT_MODE == EXPORT_MODE_CSV:
		for pkey in saved_paths.keys():
			csv_name = FILENAME_EXPORT_CSV_PREFIX + pkey + ".csv"
			csv_file  = open(csv_name, "w")

			output_string = ""

			path = saved_paths[pkey]
			waypoint = (6.45477, 2.57)
			output_string += str(waypoint[0]) + "," + str(waypoint[1]) + "\r\n"
			unity_path = []
			for p in path:
				up = image_to_unity(p)
				unity_path.append(up)
				output_string += str(up[0]) + "," + str(up[1]) + "\r\n"


			csv_file.write(output_string) 
			csv_file.close()
			print("exported csv path to " + csv_name)


cv2.imwrite('generated/fig_tables.png', img) 

# cv2.imshow("Display window", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

print("Done")





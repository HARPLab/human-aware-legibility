goals = []
tables = []
observers = []
start = []
path = []
goal_observers = {}

agents = []

import numpy as np
import math
import random

from shapely.geometry import Point as fancyPoint
from shapely.geometry import Polygon as fancyPolygon

import sys
sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/ModelPredictiveTrajectoryGenerator/')
sys.path.append('/Users/AdaTaylor/Development/PythonRobotics/PathPlanning/StateLatticePlanner/')

print(sys.path)

import state_lattice_planner as slp
import model_predictive_trajectory_generator as mptj


OPTION_SHOW_VISIBILITY = True
OPTION_EXPORT = False

length = 450
width = 600
pixels_to_foot = 10

num_tables = 6
num_observers = 6

TYPE_PLOTTED = 0
TYPE_RANDOM = 1

COLOR_TABLE = (235, 64, 52)
COLOR_OBSERVER = (32, 85, 230)
COLOR_FOCUS = (52, 192, 235)
COLOR_PERIPHERAL = (178, 221, 235)
COLOR_GOAL = (50, 168, 82)
COLOR_START = (255, 255, 255)


def get_path(start, end, obs=[]):
	path = [start, end]
	x_start, y_start = start
	x_end, y_end = end

	print("Path")
	print(path)
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

	print(x)
	print(y)
	print(yaw)
	print(p)
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
	radius = 40

	def __init__(self, pt):
		self.location = pt
		
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
		return self.radius

	def get_center(self):
		return self.location

	def get_JSON(self):
		return (self.location, self.radius)



class Observer:
	entity_radius = 10
	draw_depth = 50
	cone_depth = max(length, width)*.5
	focus_angle = 60 / 2.0
	peripheral_angle = 120 / 2.0

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
		draw_periph_a = (0, int(self.draw_depth*1.5))
		draw_periph_b = (0, int(self.draw_depth*1.5))

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
		return self.entity_radius


class Agent:
	total_path = []
	done = False

	def __init__(self, start, goal, environment):
		self.start = start
		self.pos = start
		self.goal = goal
		self.envir = environment

	def step(self):
		pass

	def is_done(self):
		return done

	def get_radius(self):
		return self.entity_radius

generate_type = TYPE_PLOTTED

if generate_type == TYPE_PLOTTED:
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
		table = Table(pt)
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

elif generate_type == TYPE_RANDOM:

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

# Get paths
# goal = random.choice(goals)
goal = goals[5]

path = get_path(start, goal)


# #DRAW the environment
import cv2

# # Create a black image
img = np.zeros((length,width,3), np.uint8)

# Draw tables
for table in tables:
	# cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
	# print("TABLE at " + str(table.get_center()))
	cv2.rectangle(img, table.pt_top_left(), table.pt_bottom_right(), COLOR_TABLE, table.get_radius())
	# print(table.pt_top_left(), table.pt_bottom_right())

# Draw observers
# Draw observer cones
# Draw paths

for obs in observers:
	# Draw person
	cv2.circle(img, obs.get_center(), obs.get_radius(), COLOR_OBSERVER, obs.get_radius())
	cv2.fillPoly(img, obs.get_draw_field_peripheral(), COLOR_PERIPHERAL)
	# Draw view cones
	cv2.fillPoly(img, obs.get_draw_field_focus(), COLOR_FOCUS)

for goal in goals:
	# Draw person
	cv2.circle(img, goal, obs.get_radius(), COLOR_GOAL, obs.get_radius())

cv2.circle(img, start, obs.get_radius(), COLOR_START, obs.get_radius())


for i in range(len(path) - 1):
	a = path[i]
	b = path[i + 1]

	cv2.line(img, a, b, COLOR_START, thickness=2, lineType=8)


# VISIBILITY TEST
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


if OPTION_SHOW_VISIBILITY:
	resolution = 5

	r_width = int(width / resolution)
	r_length = int(length / resolution)

	visibility = np.zeros((r_width, r_length))
	for x in range(r_width):
		for y in range(r_length):
			rx = x*resolution
			ry = y*resolution
			score = 0
			for obs in observers:
				score += obs.get_visibility((rx,ry))

			visibility[x,y] = score

	import numpy as np
	import seaborn as sns
	import matplotlib.pylab as plt
	visibility = visibility.T
	# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
	ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: All")
	plt.savefig('generated/fig_vis_all.png')
	plt.clf()

	# plt.show()

	visibility = np.zeros((r_width, r_length))
	for x in range(r_width):
		for y in range(r_length):
			rx = x*resolution
			ry = y*resolution
			score = 0
			for obs in goal_observers[goal]:
				score += obs.get_visibility((rx,ry))

			visibility[x,y] = score

	visibility = visibility.T
	# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
	ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: 1 Table")
	# plt.show()
	plt.savefig('generated/fig_vis_table.png')
	plt.clf()


	indic = 0
	for obs in goal_observers[goal]:
		visibility = np.zeros((r_width, r_length))
		for x in range(r_width):
			for y in range(r_length):
				rx = x*resolution
				ry = y*resolution
				score = 0
				score += obs.get_visibility((rx,ry))

				visibility[x,y] = score

		visibility = visibility.T
		# xticklabels=range(0, width, resolution), yticklabels=range(0, length, resolution)
		ax = sns.heatmap(visibility).set_title("Visibility of Restaurant Tiles: 1 Observer #" + str(indic))
		# plt.show()
		plt.savefig('generated/fig_vis_person_' + str(indic) + '.png')
		indic += 1
		plt.clf()



if OPTION_EXPORT:
	import json 

	# Ready JSON Export
	table_json = []
	for table in tables:
		table_json.append(table.get_JSON())


	print("Exporting path")
	file1 = open("generated/path_v1.txt","a")
	data = {}
	data['path'] = path
	data['start'] = start
	data['goals'] = goals
	data['tables'] = table_json
	data['server_type'] = "SERVER_HUMAN"
	data['condition'] = "naive"

	# print(json.dumps(data, indent=4))

	file1.write(json.dumps(data, indent=4)) 
	file1.close() 


cv2.imwrite('generated/fig_tables.png', img) 

cv2.imshow("Display window", img)
cv2.waitKey()
cv2.destroyAllWindows()

print("Done")





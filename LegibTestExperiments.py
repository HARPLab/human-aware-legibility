# from __future__ import print_function

import os
import sys
import copy
import time

module_path = os.path.abspath(os.path.join('../ilqr'))
if module_path not in sys.path:
	sys.path.append(module_path)

import numpy as np
import LegibSolver as solver
import matplotlib.pyplot as plt
# from datetime import timedelta, datetime

# import theano.tensor as T

# from ilqr import iLQR
# from ilqr.cost import QRCost, PathQRCost
# from ilqr.dynamics import constrain
# from ilqr.dynamics import tensor_constrain

import PathingExperiment as ex
# from LegiblePathQRCost import LegiblePathQRCost
# from DirectPathQRCost import DirectPathQRCost
# from ObstaclePathQRCost import ObstaclePathQRCost
# from LegibilityOGPathQRCost import LegibilityOGPathQRCost
# from OALegiblePathQRCost import OALegiblePathQRCost
# from NavigationDynamics import NavigationDynamics
import LegibTestScenarios as test_scenarios

import utility_environ_descrip as resto

# ###### COST/SOLVER OPTIONS
# # exp.set_cost_label(ex.COST_LEGIB)
# # exp.set_cost_label(ex.COST_OBS)
# # exp.set_cost_label(ex.COST_OA)
# exp.set_cost_label(ex.COST_OA_AND_OBS)

# ###### WEIGHTING FUNCTION 
# ###	(DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
# exp.set_f_label(ex.F_VIS_BIN)
# # exp.set_f_label(ex.F_VIS_LIN)
# # exp.set_f_label(ex.F_NONE)

def run_all_tests():
	test_heading_useful_or_no()
	exit()
	test_obstacles_being_avoided()
	test_observers_being_respected()

def get_file_id_for_exp(label):
	# Create a new folder for this experiment, along with sending debug output there
	file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + label
	try:
		os.mkdir(PREFIX_EXPORT + file_id)
	except:
		print("FILE ALREADY EXISTS " + file_id)
	
	sys.stdout = open(PREFIX_EXPORT + file_id + '/output.txt','wt')


# def add_solver_settings(self, scenario):
# 	return scenario


def test_heading_useful_or_no():
	print("TESTING IF HEADING IS USEFUL")
	scenarios = test_scenarios.get_scenarios_heading()

	# for each test scenario
	# run it with heading, get the image
	# run it without heading part, get the image

	# export both individually
	# export side by side with markings
	#	include exp settings on export

	for key in scenarios.keys():
		scenario = scenarios[key]

		with_heading = copy.copy(scenario)
		without_heading = copy.copy(scenario)

	# RUN THE SOLVER WITH CONSTRAINTS ON EACH
	without_heading.set_heading_on(False)
	with_heading.set_heading_on(True)

	verts_with_heading, us_with_heading, cost_with_heading = solver.run_solver(with_heading)
	verts_wout_heading, us_wout_heading, cost_wout_heading = solver.run_solver(without_heading)

	fig, (ax1,ax2) = plt.subplots(ncols=2)
	# fig, axes = plt.subplot_mosaic("AB", figsize=(8, 6), gridspec_kw={'height_ratios':[1], 'width_ratios':[1, 1]})
	# ax1 = axes['A'] # plot of movements in space
	# ax2 = axes['B'] # 

	cost_wout_heading.get_overview_pic(verts_wout_heading, us_wout_heading, ax=ax1)
	cost_with_heading.get_overview_pic(verts_with_heading, us_with_heading, ax=ax2)

	plt.tight_layout()
	plt.savefig('with_without_heading.png')


def test_obstacles_being_avoided():
	scenarios = test_scenarios.get_scenarios_obstacles()


def test_observers_being_respected():
	# for each test scenario
	# compare with observer vs no
	scenarios = test_scenarios.get_scenarios_observers()


def test_observers_rotated():
	scenarios = test_scenarios.get_scenarios_observers()


def generate_scenarios_with_observer_rotating():
	# for each scenario, generate 
	pass

def main():
	run_all_tests()

if __name__ == "__main__":
	main()

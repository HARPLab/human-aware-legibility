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
from datetime import timedelta, datetime

# import theano.tensor as T

# from ilqr import iLQR
# from ilqr.cost import QRCost, PathQRCost
# from ilqr.dynamics import constrain
# from ilqr.dynamics import tensor_constrain

import PathingExperiment as ex
from LegiblePathQRCost import LegiblePathQRCost
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
# ###    (DISTRIBUTING LEGIBILITY ACCORDING TO TIME OR VIS, etc)
# exp.set_f_label(ex.F_VIS_BIN)
# # exp.set_f_label(ex.F_VIS_LIN)
# # exp.set_f_label(ex.F_NONE)

def run_all_tests():
    dashboard_folder = get_dashboard_folder()

    test_heading_useful_or_no(dashboard_folder)
    exit()
    test_obstacles_being_avoided(dashboard_folder)
    test_observers_being_respected(dashboard_folder)

def get_file_id_for_exp(dash_folder, label):
    # Create a new folder for this experiment, along with sending debug output there
    file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + label
    sys.stdout = open(dash_folder + file_id + '_output.txt','wt')
    return dash_folder + file_id

def get_dashboard_folder():
    dashboard_file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + "experiments"
    try:
        os.mkdir(LegiblePathQRCost.PREFIX_EXPORT + dashboard_file_id)
    except:
        print("FILE ALREADY EXISTS " + file_id)

    dash_folder = LegiblePathQRCost.PREFIX_EXPORT + dashboard_file_id + "/"
    sys.stdout = open(dash_folder + '/output.txt','wt')
    return dash_folder

def test_heading_useful_or_no(dash_folder):
    print("TESTING IF HEADING IS USEFUL")
    scenarios = test_scenarios.get_scenarios_heading()

    # for each test scenario
    # run it with heading, get the image
    # run it without heading part, get the image

    # export both individually
    # export side by side with markings
    #    include exp settings on export

    for key in scenarios.keys():
        scenario = scenarios[key]

        with_heading = copy.copy(scenario)
        without_heading = copy.copy(scenario)

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        without_heading.set_heading_on(False)
        with_heading.set_heading_on(True)

        save_location = get_file_id_for_exp(dash_folder, "heading-" + without_heading.get_exp_label())

        verts_with_heading, us_with_heading, cost_with_heading = solver.run_solver(with_heading)
        verts_wout_heading, us_wout_heading, cost_wout_heading = solver.run_solver(without_heading)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8, 6))

        cost_wout_heading.get_overview_pic(verts_wout_heading, us_wout_heading, ax=ax1)
        cost_with_heading.get_overview_pic(verts_with_heading, us_with_heading, ax=ax2)

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Without Heading", fontweight='bold')
        _ = ax2.set_title("With Heading", fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(save_location + ".png")


def test_obstacles_being_avoided(dash_folder):
    scenarios = test_scenarios.get_scenarios_obstacles()


def test_observers_being_respected(dash_folder):
    # for each test scenario
    # compare with observer vs no
    scenarios = test_scenarios.get_scenarios_observers()


def test_observers_rotated(dash_folder):
    scenarios = test_scenarios.get_scenarios_observers()


def generate_scenarios_with_observer_rotating():
    # for each scenario, generate 
    pass

def main():
    run_all_tests()

if __name__ == "__main__":
    main()

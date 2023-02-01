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
import utility_environ_descrip as resto

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
    test_amount_of_slack(dashboard_folder)
    # test_observers_rotated(dashboard_folder)
    # exit()
    # test_observers_being_respected(dashboard_folder)
    # exit()
    # test_obstacles_being_avoided(dashboard_folder)
    # exit()
    # test_heading_useful_or_no(dashboard_folder)
    # exit()

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

    for key in scenarios.keys():
        scenario = scenarios[key]

        obs_scenario = copy.copy(scenario)

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        obs_scenario.set_heading_on(True)
        save_location = get_file_id_for_exp(dash_folder, "obs-" + obs_scenario.get_exp_label())

        verts_with_obs, us_with_obs, cost_with_obs = solver.run_solver(obs_scenario)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1) = plt.subplots(ncols=1, figsize=(4, 3))

        cost_with_obs.get_overview_pic(verts_with_obs, us_with_obs, ax=ax1)
        _ = ax1.set_title("Check for Obstacle Issues", fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_location + ".png")


def test_observers_being_respected(dash_folder):
    # for each test scenario
    # compare with observer vs no
    scenarios = test_scenarios.get_scenarios_observers()

    for key in scenarios.keys():
        scenario = scenarios[key]

        with_oa = copy.copy(scenario)
        wout_oa = copy.copy(scenario)

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        with_oa.set_oa_on(True)
        wout_oa.set_oa_on(False)
        
        save_location = get_file_id_for_exp(dash_folder, "oa-" + wout_oa.get_exp_label())

        verts_with_oa, us_with_oa, cost_with_oa = solver.run_solver(with_oa)
        verts_wout_oa, us_wout_oa, cost_wout_oa = solver.run_solver(wout_oa)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8, 6))

        cost_wout_oa.get_overview_pic(verts_wout_oa, us_wout_oa, ax=ax1)
        cost_with_oa.get_overview_pic(verts_with_oa, us_with_oa, ax=ax2)

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Omniscient", fontweight='bold')
        _ = ax2.set_title("Observer-Aware", fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        plt.savefig(save_location + ".png")


def test_amount_of_slack(dash_folder):
    scenarios = test_scenarios.get_scenarios()

    scenario_output_list = []
    for key in scenarios.keys():
        scenario = scenarios[key]
        base_N = scenario.get_N()

        # n_scenarios = generate_8_scenarios_varying_N(scenario)
        save_location = get_file_id_for_exp(dash_folder, "N-" + scenario.get_exp_label())
        
        scale_set = [.5, 0.625, .75, 0.875, 1, 1.125, 1.25, 1.375, 1.5]

        outputs = {}
        label_dict = {}
        for multiplier in scale_set:
            # RUN THE SOLVER WITH CONSTRAINTS ON EACH
            n_scenario = copy.copy(scenario)
            new_N = int(base_N * multiplier)
            n_percent = int(100.0 * multiplier)

            label_dict[multiplier] = n_percent
            n_scenario.set_N(new_N)
            
            verts_with_n, us_with_n, cost_with_n = solver.run_solver(n_scenario)
            outputs[multiplier] = verts_with_n, us_with_n, cost_with_n


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, axes = plt.subplot_mosaic("ABC;IDJ;FGH", figsize=(8, 6), gridspec_kw={'height_ratios':[1, 1, 1], 'width_ratios':[1, 1, 1]})

        ax_mappings = {}
        ax_mappings[scale_set[4]] = axes['D']

        ax_mappings[scale_set[3]] = axes['I']
        ax_mappings[scale_set[2]] = axes['C']
        ax_mappings[scale_set[1]] = axes['B']
        ax_mappings[scale_set[0]] = axes['A']
        
        ax_mappings[scale_set[5]] = axes['J']
        ax_mappings[scale_set[6]] = axes['F']
        ax_mappings[scale_set[7]] = axes['G']
        ax_mappings[scale_set[8]] = axes['H']

        for key in outputs.keys():
            ax = ax_mappings[key]
            verts_with_n, us_with_n, cost_with_n = outputs[key]

            label = str(label_dict[key]) + "%"

            cost_with_n.get_overview_pic(verts_with_n, us_with_n, ax=ax)
            _ = ax.set_title("N= " + label, fontweight='bold')
            ax.get_legend().remove()
    
        plt.tight_layout()
        fig.suptitle("N=" + str(base_N))
        plt.savefig(save_location + ".png")


def test_observers_rotated(dash_folder):
    scenarios = test_scenarios.get_scenarios_observers()
    # 3 to either side at +30, +60, +90, and minus the same

    # for each test scenario
    # compare with observer vs no

    scenario_output_list = []
    for key in scenarios.keys():
        scenario = scenarios[key]

        rot_scenarios = generate_scenarios_with_observer_rotating(scenario)
        save_location = get_file_id_for_exp(dash_folder, "rot-" + scenario.get_exp_label())
        
        outputs = {}
        for rkey in rot_scenarios.keys():
            # RUN THE SOLVER WITH CONSTRAINTS ON EACH
            rot_scenario = rot_scenarios[rkey]
            rot_scenario.set_oa_on(True)
            
            verts_with_rot, us_with_rot, cost_with_rot = solver.run_solver(rot_scenario)
            outputs[rkey] = verts_with_rot, us_with_rot, cost_with_rot


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, axes = plt.subplot_mosaic("ABC;IDJ;FGH", figsize=(8, 6), gridspec_kw={'height_ratios':[1, 1, 1], 'width_ratios':[1, 1, 1]})

        ax_mappings = {}
        ax_mappings[0] = axes['D']

        ax_mappings[30] = axes['I']
        ax_mappings[60] = axes['C']
        ax_mappings[90] = axes['B']
        ax_mappings[120] = axes['A']
        
        ax_mappings[-30] = axes['J']
        ax_mappings[-60] = axes['F']
        ax_mappings[-90] = axes['G']
        ax_mappings[-120] = axes['H']

        for key in outputs.keys():
            ax = ax_mappings[key]
            verts_with_rot, us_with_rot, cost_with_rot = outputs[key]

            if key > 0:
                amount_label = "+"
            else:
                amount_label = ""
            amount_label += str(key)

            cost_with_rot.get_overview_pic(verts_with_rot, us_with_rot, ax=ax)
            _ = ax.set_title("Rotated " + amount_label, fontweight='bold')
            ax.get_legend().remove()
    
        plt.tight_layout()
        plt.savefig(save_location + ".png")



def generate_scenarios_with_observer_rotating(exp):
    # for each scenario, generate 
    observer = exp.get_observers()

    scenario_dict = {}
    for offset in [120, 90, 60, 30, 0, -30, -60, -90, -120]:
        rot_scenario = copy.copy(exp)
        obs = rot_scenario.get_observers()
        new_obs = copy.copy(obs[0])

        new_angle = new_obs.get_orientation() + offset
        new_angle = new_angle % 360

        new_obs.set_orientation(new_angle)
        new_obs_list = [new_obs]
        rot_scenario.set_observers(new_obs_list)

        scenario_dict[offset] = rot_scenario

    return scenario_dict

def main():
    run_all_tests()

if __name__ == "__main__":
    main()
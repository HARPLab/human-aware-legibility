purpose = "understanding_pilot"
# purpose = "test_dist_k"

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
import pandas as pd
import markupsafe
from random import randint
from matplotlib import colors

import utility_environ_descrip as resto

import PathingExperiment as ex
from LegiblePathQRCost import LegiblePathQRCost
import LegibTestScenarios as test_scenarios

import utility_environ_descrip as resto

test_log = []
np.set_printoptions(suppress=True)

def run_all_tests():
    dashboard_folder = get_dashboard_folder()
    
    # SET UP OPTIONS FOR THIS RUN
    # MAINLY IS IT'S A FULL RUN OR A QUICK ONE TO VERIFY CODE
    scenario_filters = {}
    scenario_filters[test_scenarios.SCENARIO_FILTER_MINI]       = True
    scenario_filters[test_scenarios.SCENARIO_FILTER_FAST_SOLVE] = False
    scenario_filters[test_scenarios.DASHBOARD_FOLDER]           = dashboard_folder

    # test_understanding_set(dashboard_folder, scenario_filters)
    test_locality_set(dashboard_folder, scenario_filters)

def get_file_id_for_exp(dash_folder, label):
    # Create a new folder for this experiment, along with sending debug output there
    file_id = label #+ "-" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    n = 5
    rand_id = ''.join(["{}".format(randint(0, 9)) for num in range(0, n)])
    # sys.stdout = open(dash_folder + file_id + "_" + str(rand_id) + '_output.txt','a')
    return dash_folder + file_id

def get_dashboard_folder():
    # purpose is a note field set at the top of this doc
    purpose_fn = purpose.replace(" ", "_")[:30]

    dashboard_file_id = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "-" + "exp-" + purpose_fn
    try:
        os.mkdir(LegiblePathQRCost.PREFIX_EXPORT + dashboard_file_id)

        with open(LegiblePathQRCost.PREFIX_EXPORT + dashboard_file_id + '/readme.txt', 'w') as f:
            f.write(purpose)
    except:
        print("FILE ALREADY EXISTS " + file_id)

    dash_folder = LegiblePathQRCost.PREFIX_EXPORT + dashboard_file_id + "/"
    # sys.stdout = open(dash_folder + '/output.txt','a')
    return dash_folder

def collate_and_report_on_results(dash_folder):
    df_cols = ['scenario', 'goal', 'test', 'condition', 'status_summary', 'converged', 'num_iterations', 'info', 'J_opt']
    df = pd.DataFrame(test_log, columns=df_cols)

    def _colorize(val):
        color = 'white'
        color = 'pink' if "INC" in str(val) else color
        color = 'lightgreen' if "CONV" in str(val) else color
        color = 'lightcyan' if "OK" in str(val) else color
        color = 'lightgrey' if "TODO" in str(val) else color
        return 'background-color: %s' % color

    save_location = dash_folder + "/status_overview" #get_file_id_for_exp(dash_folder, "status_overview.csv")
    df.to_csv(save_location + ".csv")
    df_shiny = df.style.applymap(_colorize)
    df_shiny.to_html(save_location + "-flat.html")

    # pandas.pivot(index, columns, values)
    df_dashboard = df.pivot_table(index=[df_cols[0], df_cols[1]], columns=[df_cols[2], df_cols[3]], values='status_summary', fill_value="TODO", aggfunc=lambda x: ' '.join(x))
    df_dashboard = df_dashboard.style.applymap(_colorize)

    save_location = dash_folder + "/dashboard" #get_file_id_for_exp(dash_folder, "status_overview.csv")
    df_dashboard.to_html(save_location + ".html") #sparse_index=True, sparse_columns=True

    return

    save_location = dash_folder + "/dashboard" #get_file_id_for_exp(dash_folder, "status_overview.csv")
    df_dashboard.to_html(save_location + ".html") #sparse_index=True, sparse_columns=True
    df_dashboard.to_latex(save_location + ".latex") #sparse_index=True, sparse_columns=True
    df_dashboard.to_excel(save_location + ".xls", merge_cells=True, engine='openpyxl')


def test_understanding_set(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios(scenario_filters)
    test_group = "understanding"

    test_setups_og = []

    new_test      = {'label':"u=gn",    'title':'Understanding Global None',    'und_target': 'global', 'und_secondary': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=gg",    'title':'Understanding Global Global',  'und_target': 'global', 'und_secondary': 'global'}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=gl",    'title':'Understanding Global Local',   'und_target': 'global', 'und_secondary': 'local'}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=nn",    'title':'Understanding None None',      'und_target': None,     'und_secondary': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=ng",    'title':'Understanding None Global',    'und_target': None,     'und_secondary': 'global'}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=nl",    'title':'Understanding None Local',     'und_target': None,     'und_secondary': 'local'}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=ln",    'title':'Understanding Local None',     'und_target': 'local',  'und_secondary': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=lg",    'title':'Understanding Local Global',   'und_target': 'local',  'und_secondary': 'global'}
    test_setups_og.append(new_test)

    new_test      = {'label':"u=ll",    'title':'Understanding Local Local',    'und_target': 'local',  'und_secondary': 'local'}
    test_setups_og.append(new_test)


    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)
        
        outputs = {}
        label_dict = {}

        for ti in range(len(test_setups_og)):
            for g_index in range(len(scenario.get_goals())):
                test = test_setups_og[ti]
                # RUN THE SOLVER WITH CONSTRAINTS ON EACH
                n_scenario = copy.copy(scenario)

                mega_scenario = copy.copy(scenario)
                mega_scenario.set_fn_note(test['label'])
                mega_scenario.set_test_options(test)
                mega_scenario.set_target_goal_index(g_index)

                save_location = get_file_id_for_exp(dash_folder, "und-" + mega_scenario.get_exp_label() + "_g" + str(g_index))

                verts_with_n, us_with_n, cost_with_n, info_packet = solver.run_solver(mega_scenario)
                outputs[(ti, g_index)] = verts_with_n, us_with_n, cost_with_n, test['label']

                test_log.append(mega_scenario.get_solve_quality_status(test_group))

                collate_and_report_on_results(dash_folder)


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        
        # EXPORT PATHS FOR EACH GOAL
        print("Exporting paths pic for each goal")
        goal_indexes = range(len(mega_scenario.get_goals()))
        for gi in goal_indexes:
            fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups_og)
            # EXPORT GRAPH ACROSS ALL GOALS
            for key in outputs.keys():
                ax_key, goal_key = key
                if goal_key == gi:
                    ax = ax_mappings[ax_key]
                    verts, us, cost, label = outputs[key]
                    
                    cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
                    _ = ax.set_title(label, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.get_legend().remove()
                    max_key = key

            for ax_index in range(len(test_setups_og), len(ax_mappings)):
                ax_mappings[ax_index].axis('off')

            mega_scenario.set_target_goal_index(gi)
            save_location = get_file_id_for_exp(dash_folder, "cross-" + mega_scenario.get_exp_label() + "_g" + str(gi))

            fig.suptitle("=g" + str(gi)) # + " " + mega_scenario.get_goal_label())
            plt.subplots_adjust(top=0.9)
            # plt.tight_layout()
            plt.savefig(save_location + ".png")
            plt.close()
            plt.clf()


        print("Exporting paths pic for all goals")
        fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups_og)
        # EXPORT GRAPH ACROSS ALL GOALS
        for key in outputs.keys():
            ax_key, goal_key = key
            ax = ax_mappings[ax_key]
            verts, us, cost, label = outputs[key]
            
            cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
            _ = ax.set_title(label, fontweight='bold')
            ax.get_legend().remove()
            max_key = key

        for ax_index in range(len(test_setups_og), len(ax_mappings)):
            ax_mappings[ax_index].axis('off')

        save_location = get_file_id_for_exp(dash_folder, "cross-" + "-" + mega_scenario.get_exp_label() + "-all")
        save_location = get_file_id_for_exp(dash_folder, "all-cross-" + "-" + mega_scenario.get_exp_label() + "-all")

        fig.suptitle("cross=all") # + " " + mega_scenario.get_goal_label())
        plt.subplots_adjust(top=0.9)
        # plt.tight_layout()
        plt.savefig(save_location + ".png")
        plt.close()
        plt.clf()


def test_locality_set(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios(scenario_filters)
    test_group = "locality"

    test_template      = {'label':"ll-",    'title':'Understanding Local Local',    'und_target': 'local',  'und_secondary': 'local'}

    # scale_set = [.5, 0.625, .75, 0.875, 1, 1.125, 1.25, 1.375, 1.5]
    scale_set = [1, 2, 4, 8, 16, 32, 64, 128][::-1]

    scale_set = [1, 16, 64][::-1]

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        longest_distance = scenario.get_max_dist()

        outputs = {}
        label_dict = {}

        test_setups = []

        for multiplier in scale_set:
            test = copy.copy(test_template)
            
            new_N = (longest_distance / multiplier)
            n_percent = int(100.0 * multiplier)

            label_dict[multiplier] = new_N

            test['label']           = 'local=' + str("{0:.3g}".format((new_N)))
            test['local_distance']  = new_N

            test_setups.append(test)

            for g_index in range(len(scenario.get_goals())):
                # RUN THE SOLVER WITH CONSTRAINTS ON EACH
                n_scenario = copy.copy(scenario)

                n_scenario.set_local_distance(new_N)
                n_scenario.set_fn_note("locdist_" + str(new_N))

                mega_scenario = copy.copy(scenario)
                # mega_scenario.set_fn_note(test['label'])
                mega_scenario.set_test_options(test)
                mega_scenario.set_target_goal_index(g_index)

                save_location = get_file_id_for_exp(dash_folder, "dist-" + mega_scenario.get_exp_label() + "_g" + str(g_index))

                verts_with_n, us_with_n, cost_with_n, info_packet = solver.run_solver(mega_scenario)
                outputs[(multiplier, g_index)] = verts_with_n, us_with_n, cost_with_n, test['label']

                test_log.append(mega_scenario.get_solve_quality_status(test_group))

                collate_and_report_on_results(dash_folder)


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        
        # EXPORT PATHS FOR EACH GOAL
        print("Exporting paths pic for each goal")
        goal_indexes = range(len(mega_scenario.get_goals()))
        for gi in goal_indexes:
            fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups)
            # EXPORT GRAPH ACROSS ALL GOALS
            for key in outputs.keys():
                ax_number, goal_key = key
                ax_key = scale_set.index(ax_number)

                if goal_key == gi:
                    ax = ax_mappings[ax_key]
                    verts, us, cost, label = outputs[key]
                    
                    cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
                    _ = ax.set_title(label, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.get_legend().remove()
                    max_key = key

            for ax_index in range(len(test_setups), len(ax_mappings)):
                ax_mappings[ax_index].axis('off')

            mega_scenario.set_target_goal_index(gi)
            save_location = get_file_id_for_exp(dash_folder, "cross-" + mega_scenario.get_exp_label() + "_g" + str(gi))

            fig.suptitle("=g" + str(gi)) # + " " + mega_scenario.get_goal_label())
            plt.subplots_adjust(top=0.9)
            # plt.tight_layout()
            plt.savefig(save_location + ".png")
            plt.close()
            plt.clf()


        print("Exporting paths pic for all goals")
        fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups)
        # EXPORT GRAPH ACROSS ALL GOALS
        for key in outputs.keys():
            ax_number, goal_key = key
            ax_key = scale_set.index(ax_number)

            ax = ax_mappings[ax_key]
            verts, us, cost, label = outputs[key]
            
            cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
            _ = ax.set_title(label, fontweight='bold')
            ax.get_legend().remove()
            max_key = key

        for ax_index in range(len(test_setups), len(ax_mappings)):
            ax_mappings[ax_index].axis('off')

        save_location = get_file_id_for_exp(dash_folder, "cross-" + "-" + mega_scenario.get_exp_label() + "-all")
        save_location = get_file_id_for_exp(dash_folder, "all-cross-" + "-" + mega_scenario.get_exp_label() + "-all")

        fig.suptitle("cross=all") # + " " + mega_scenario.get_goal_label())
        plt.subplots_adjust(top=0.9)
        # plt.tight_layout()
        plt.savefig(save_location + ".png")
        plt.close()
        plt.clf()


def test_raw_set(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios(scenario_filters)
    test_group = "raw-options"

    test_setups_og = []

    # new_test      = {'label':"und_all", 'title':'Understanding full path', 'mode_heading':None, 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"und_handoff", 'title':'Understanding handoff', 'mode_heading':None, 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"und_no_confuse", 'title':'Understanding No Confusion', 'mode_heading':None, 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    new_test      = {'label':"head_lin", 'title':'Pure linear heading', 'mode_heading':'lin', 'mode_dist':None, 'mode_blend': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"dist_lin", 'title':'Pure linear distance', 'mode_heading':None, 'mode_dist':'lin', 'mode_blend': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"dist_exp", 'title':'Pure OG', 'mode_heading':None, 'mode_dist':'exp', 'mode_blend': None}
    test_setups_og.append(new_test)


    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)
        
        outputs = {}
        label_dict = {}

        for ti in range(len(test_setups_og)):
            for g_index in range(len(scenario.get_goals())):
                test = test_setups_og[ti]
                # RUN THE SOLVER WITH CONSTRAINTS ON EACH
                n_scenario = copy.copy(scenario)

                mega_scenario = copy.copy(scenario)
                mega_scenario.set_fn_note(test['label'])
                mega_scenario.set_test_options(test)
                mega_scenario.set_target_goal_index(g_index)

                save_location = get_file_id_for_exp(dash_folder, "und-" + mega_scenario.get_exp_label() + "_g" + str(g_index))

                verts_with_n, us_with_n, cost_with_n, info_packet = solver.run_solver(mega_scenario)
                outputs[(ti, g_index)] = verts_with_n, us_with_n, cost_with_n, test['label']

                test_log.append(mega_scenario.get_solve_quality_status(test_group))

                collate_and_report_on_results(dash_folder)


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        
        # EXPORT PATHS FOR EACH GOAL
        print("Exporting paths pic for each goal")
        goal_indexes = range(len(mega_scenario.get_goals()))
        for gi in goal_indexes:
            fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups_og)
            # EXPORT GRAPH ACROSS ALL GOALS
            for key in outputs.keys():
                ax_key, goal_key = key
                if goal_key == gi:
                    ax = ax_mappings[ax_key]
                    verts, us, cost, label = outputs[key]
                    
                    cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
                    _ = ax.set_title(label, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.get_legend().remove()
                    max_key = key

            for ax_index in range(len(test_setups_og), len(ax_mappings)):
                ax_mappings[ax_index].axis('off')

            mega_scenario.set_target_goal_index(gi)
            save_location = get_file_id_for_exp(dash_folder, "cross-" + mega_scenario.get_exp_label() + "_g" + str(gi))

            fig.suptitle("cross=g" + str(gi)) # + " " + mega_scenario.get_goal_label())
            plt.subplots_adjust(top=0.9)
            # plt.tight_layout()
            plt.savefig(save_location + ".png")
            plt.close()
            plt.clf()


        print("Exporting paths pic for all goals")
        fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups_og)
        # EXPORT GRAPH ACROSS ALL GOALS
        for key in outputs.keys():
            ax_key, goal_key = key
            ax = ax_mappings[ax_key]
            verts, us, cost, label = outputs[key]
            
            cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
            _ = ax.set_title(label, fontweight='bold')
            ax.get_legend().remove()
            max_key = key

        for ax_index in range(len(test_setups_og), len(ax_mappings)):
            ax_mappings[ax_index].axis('off')

        save_location = get_file_id_for_exp(dash_folder, "cross-" + "-" + mega_scenario.get_exp_label() + "-all")
        save_location = get_file_id_for_exp(dash_folder, "all-cross-" + "-" + mega_scenario.get_exp_label() + "-all")

        fig.suptitle("cross=all") # + " " + mega_scenario.get_goal_label())
        plt.subplots_adjust(top=0.9)
        # plt.tight_layout()
        plt.savefig(save_location + ".png")
        plt.close()
        plt.clf()

def test_full_set(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios(scenario_filters)
    test_group = "all-cross"

    test_setups_og = []

    # new_test      = {'label':"no-legib", 'title':'No Legibility, just direct', 'mode_heading':None, 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"head_lin", 'title':'Pure linear heading', 'mode_heading':'lin', 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"head_sqr", 'title':'Pure squared heading', 'mode_heading':'sqr', 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    new_test      = {'label':"dist_lin", 'title':'Dist linear heading', 'mode_heading':None, 'mode_dist':'lin', 'mode_blend': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"dist_sqr", 'title':'Dist square heading', 'mode_heading':None, 'mode_dist':'sqr', 'mode_blend': None}
    test_setups_og.append(new_test)

    new_test      = {'label':"dist_exp", 'title':'Pure OG', 'mode_heading':None, 'mode_dist':'exp', 'mode_blend': None}
    test_setups_og.append(new_test)

    # new_test      = {'label':"head_exp", 'title':'Pure exp heading', 'mode_heading':'exp', 'mode_dist':None, 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"dist_sqr", 'title':'Dist square heading', 'mode_heading':None, 'mode_dist':'sqr', 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"dist_lin", 'title':'Dist linear heading', 'mode_heading':None, 'mode_dist':'lin', 'mode_blend': None}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"mixed_sqr", 'title':'Mixed Dist / sqr heading', 'mode_heading':'sqr', 'mode_dist':'sqr', 'mode_blend': 'min'}
    # test_setups_og.append(new_test)

    # new_test      = {'label':"mixed_lin", 'title':'Mixed Dist / linear heading', 'mode_heading':'lin', 'mode_dist':'lin', 'mode_blend': 'min'}
    # test_setups_og.append(new_test)


    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)
        
        outputs = {}
        label_dict = {}

        for ti in range(len(test_setups_og)):
            for g_index in range(len(scenario.get_goals())):
                test = test_setups_og[ti]
                # RUN THE SOLVER WITH CONSTRAINTS ON EACH
                n_scenario = copy.copy(scenario)

                mega_scenario = copy.copy(scenario)
                mega_scenario.set_fn_note(test['label'])
                mega_scenario.set_test_options(test)
                mega_scenario.set_target_goal_index(g_index)

                save_location = get_file_id_for_exp(dash_folder, "cross-" + mega_scenario.get_exp_label() + "_g" + str(g_index))

                verts_with_n, us_with_n, cost_with_n, info_packet = solver.run_solver(mega_scenario)
                outputs[(ti, g_index)] = verts_with_n, us_with_n, cost_with_n, test['label']

                test_log.append(mega_scenario.get_solve_quality_status(test_group))

                collate_and_report_on_results(dash_folder)


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        
        # EXPORT PATHS FOR EACH GOAL
        print("Exporting paths pic for each goal")
        goal_indexes = range(len(mega_scenario.get_goals()))
        for gi in goal_indexes:
            fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups_og)
            # EXPORT GRAPH ACROSS ALL GOALS
            for key in outputs.keys():
                ax_key, goal_key = key
                if goal_key == gi:
                    ax = ax_mappings[ax_key]
                    verts, us, cost, label = outputs[key]
                    
                    cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
                    _ = ax.set_title(label, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.get_legend().remove()
                    max_key = key

            for ax_index in range(len(test_setups_og), len(ax_mappings)):
                ax_mappings[ax_index].axis('off')

            mega_scenario.set_target_goal_index(gi)
            save_location = get_file_id_for_exp(dash_folder, "cross-" + mega_scenario.get_exp_label() + "_g" + str(gi))

            fig.suptitle("cross=g" + str(gi)) # + " " + mega_scenario.get_goal_label())
            plt.subplots_adjust(top=0.9)
            # plt.tight_layout()
            plt.savefig(save_location + ".png")
            plt.close()
            plt.clf()


        print("Exporting paths pic for all goals")
        fig, axes, ax_mappings = setup_axes_for_test_setups(test_setups_og)
        # EXPORT GRAPH ACROSS ALL GOALS
        for key in outputs.keys():
            ax_key, goal_key = key
            ax = ax_mappings[ax_key]
            verts, us, cost, label = outputs[key]
            
            cost.get_overview_pic(verts, us, ax=ax, info_packet=info_packet, dash_folder=dash_folder, multilayer_draw=True)
            _ = ax.set_title(label, fontweight='bold')
            ax.get_legend().remove()
            max_key = key

        for ax_index in range(len(test_setups_og), len(ax_mappings)):
            ax_mappings[ax_index].axis('off')

        save_location = get_file_id_for_exp(dash_folder, "cross-" + "-" + mega_scenario.get_exp_label() + "-all")
        save_location = get_file_id_for_exp(dash_folder, "all-cross-" + "-" + mega_scenario.get_exp_label() + "-all")

        fig.suptitle("cross=all") # + " " + mega_scenario.get_goal_label())
        plt.subplots_adjust(top=0.9)
        # plt.tight_layout()
        plt.savefig(save_location + ".png")
        plt.close()
        plt.clf()



def test_vanilla_set(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios(scenario_filters)
    test_group = "vanilla"

    test_setups_og = []

    new_test      = {'label':"no-legib", 'title':'No Legibility, just direct', 'heading-on':False, 'pure-heading':False, 'heading_sqr':False, 'dist-legib-on':False}
    test_setups_og.append(new_test)

    new_test      = {'label':"pure_dist", 'title':'Pure OG', 'heading-on':False, 'pure-heading':False, 'heading_sqr':False, 'dist-legib-on':True}
    test_setups_og.append(new_test)

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        save_location = get_file_id_for_exp(dash_folder, "vanil-" + scenario.get_exp_label())
        
        outputs = {}
        label_dict = {}
        for ti in range(len(test_setups_og)):
            test = test_setups_og[ti]
            # RUN THE SOLVER WITH CONSTRAINTS ON EACH
            n_scenario = copy.copy(scenario)

            mega_scenario = copy.copy(scenario)
            mega_scenario.set_fn_note(test['label'])

            # RUN THE SOLVER WITH CONSTRAINTS ON EACH
            mega_scenario.set_heading_on(test['heading-on'])
            mega_scenario.set_mode_pure_heading(test['pure-heading'])
            mega_scenario.set_mode_dist_legib_on(test['dist-legib-on'])
            mega_scenario.set_mode_heading_err_sqr(test['heading_sqr'])
            
            verts_with_n, us_with_n, cost_with_n, info_packet = solver.run_solver(mega_scenario)
            outputs[ti] = verts_with_n, us_with_n, cost_with_n, test['label']

            test_log.append(mega_scenario.get_solve_quality_status(test_group))


        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, axes = plt.subplot_mosaic("AB", figsize=(8, 6), gridspec_kw={'height_ratios':[1], 'width_ratios':[1, 1]})
        ax_mappings = {}
        ax_mappings[0] = axes['A']
        ax_mappings[1] = axes['B']
        # ax_mappings[6] = axes['G']
        # ax_mappings[7] = axes['H']
        # ax_mappings[8] = axes['I']

        for key in outputs.keys():
            ax = ax_mappings[key]
            verts_with_n, us_with_n, cost_with_n, label = outputs[key]

            # label = str(label_dict[key]) + "%"
            
            cost_with_n.get_overview_pic(verts_with_n, us_with_n, ax=ax, info_packet=info_packet, dash_folder=dash_folder)
            _ = ax.set_title(label, fontweight='bold')
            ax.get_legend().remove()

        plt.tight_layout()
        fig.suptitle("legib=" + " " + mega_scenario.get_goal_label())
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_location + ".png")


        collate_and_report_on_results(dash_folder)

def test_amount_of_slack(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios(scenario_filters)
    test_group = "amt slack"

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)
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

            n_scenario.set_fn_note("n_" + str(new_N))
            
            verts_with_n, us_with_n, cost_with_n, info_packet = solver.run_solver(n_scenario)
            outputs[multiplier] = verts_with_n, us_with_n, cost_with_n

            test_log.append(n_scenario.get_solve_quality_status(test_group))


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

            cost_with_n.get_overview_pic(verts_with_n, us_with_n, ax=ax, info_packet=info_packet, dash_folder=dash_folder)
            _ = ax.set_title("N= " + label, fontweight='bold')
            ax.get_legend().remove()
    
        plt.tight_layout()
        fig.suptitle("N=" + str(base_N) + " " + n_scenario.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)



def test_observers_rotated(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios_observers(scenario_filters)
    test_group = 'obs rotated ok?'
    # 3 to either side at +30, +60, +90, and minus the same

    # for each test scenario
    # compare with observer vs no
    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        rot_scenarios = generate_scenarios_with_observer_rotating(scenario)
        save_location = get_file_id_for_exp(dash_folder, "rot-" + scenario.get_exp_label())
        
        outputs = {}
        for rkey in rot_scenarios.keys():
            # RUN THE SOLVER WITH CONSTRAINTS ON EACH
            rot_scenario = rot_scenarios[rkey]
            rot_scenario.set_oa_on(True)
            
            verts_with_rot, us_with_rot, cost_with_rot, info_packet = solver.run_solver(rot_scenario)
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

            cost_with_rot.get_overview_pic(verts_with_rot, us_with_rot, ax=ax, info_packet=info_packet, dash_folder=dash_folder)
            _ = ax.set_title("Rotated " + amount_label, fontweight='bold')
            ax.get_legend().remove()
    
        plt.tight_layout()
        fig.suptitle("Goal = " + rot_scenario.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

    # collate_and_report_on_results(dash_folder)

def test_heading_useful_or_no(dash_folder, scenario_filters):
    test_group = 'heading useful?'

    print("TESTING IF HEADING IS USEFUL")
    scenarios = test_scenarios.get_scenarios_heading(scenario_filters)

    # for each test scenario
    # run it with heading, get the image
    # run it without heading part, get the image

    # export both individually
    # export side by side with markings
    #    include exp settings on export

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        without_heading = copy.copy(scenario)
        mixed_heading   = copy.copy(scenario)
        pure_heading    = copy.copy(scenario)

        without_heading.set_fn_note("wout_head")
        mixed_heading.set_fn_note("mixd_head")
        pure_heading.set_fn_note("pure_head")

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        without_heading.set_heading_on(False)
        without_heading.set_mode_pure_heading(False)

        mixed_heading.set_heading_on(True)
        mixed_heading.set_mode_pure_heading(False)
        
        pure_heading.set_heading_on(True)
        pure_heading.set_mode_pure_heading(True)

        save_location = get_file_id_for_exp(dash_folder, "headings-" + without_heading.get_exp_label())

        verts_pure_heading, us_pure_heading, cost_pure_heading, info_packet1    = solver.run_solver(pure_heading)
        verts_mixed_heading, us_mixed_heading, cost_mixed_heading, info_packet2 = solver.run_solver(mixed_heading)
        verts_wout_heading, us_wout_heading, cost_wout_heading, info_packet3    = solver.run_solver(without_heading)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1,ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 4))

        cost_wout_heading.get_overview_pic(verts_wout_heading, us_wout_heading, ax=ax1, dash_folder=dash_folder)
        cost_mixed_heading.get_overview_pic(verts_mixed_heading, us_mixed_heading, ax=ax2, dash_folder=dash_folder)
        cost_pure_heading.get_overview_pic(verts_pure_heading, us_pure_heading, ax=ax3, dash_folder=dash_folder)

        blurb1 = without_heading.get_solver_status_blurb()
        blurb2 = mixed_heading.get_solver_status_blurb()
        blurb3 = pure_heading.get_solver_status_blurb()

        test_log.append(without_heading.get_solve_quality_status(test_group))
        test_log.append(mixed_heading.get_solve_quality_status(test_group))
        test_log.append(pure_heading.get_solve_quality_status(test_group))

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Without Heading\n" + blurb1, fontweight='bold')
        _ = ax2.set_title("Mixed Heading\n" + blurb2, fontweight='bold')
        _ = ax3.set_title("Pure Heading\n" + blurb3, fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        fig.suptitle("Goal = " + pure_heading.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)


def test_heading_sqr_or_no(dash_folder, scenario_filters):
    print("TESTING IF HEADING PLAIN OR SQR BETTER")
    test_group = 'heading lin or sqr?'
    scenarios = test_scenarios.get_scenarios_heading(scenario_filters)

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        heading_lin = copy.copy(scenario)
        heading_sqr = copy.copy(scenario)

        heading_lin.set_fn_note("head_lin")
        heading_sqr.set_fn_note("head_sqr")

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        heading_lin.set_heading_on(True)
        heading_lin.set_mode_pure_heading(True)
        heading_lin.set_mode_heading_err_sqr(False)
        
        heading_sqr.set_heading_on(True)
        heading_sqr.set_mode_pure_heading(True)
        heading_sqr.set_mode_heading_err_sqr(True)

        save_location = get_file_id_for_exp(dash_folder, "head-sqr-" + heading_lin.get_exp_label())

        verts_heading_lin, us_heading_lin, cost_heading_lin, info_packet1    = solver.run_solver(heading_lin)
        verts_heading_sqr, us_heading_sqr, cost_heading_sqr, info_packet3    = solver.run_solver(heading_sqr)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(6, 4))

        cost_heading_lin.get_overview_pic(verts_heading_lin, us_heading_lin, ax=ax1, dash_folder=dash_folder)
        cost_heading_sqr.get_overview_pic(verts_heading_sqr, us_heading_sqr, ax=ax3, dash_folder=dash_folder)

        blurb1 = heading_lin.get_solver_status_blurb()
        blurb3 = heading_sqr.get_solver_status_blurb()

        test_log.append(heading_lin.get_solve_quality_status(test_group))
        test_log.append(heading_sqr.get_solve_quality_status(test_group))

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Linear Heading\n" + blurb1, fontweight='bold')
        _ = ax3.set_title("Sqr Heading\n" + blurb3, fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        fig.suptitle("Goal = " + heading_sqr.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)


def test_normalized_or_no(dash_folder, scenario_filters):
    print("TESTING IF NORMALIZED MATTERS")
    test_group = 'normalized or no?'
    scenarios = test_scenarios.get_scenarios_observers(scenario_filters)

    # for each test scenario
    # run it with heading, get the image
    # run it without heading part, get the image

    # export both individually
    # export side by side with markings
    #    include exp settings on export

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        with_heading = copy.copy(scenario)
        without_heading = copy.copy(scenario)

        with_heading.set_fn_note("with_head")
        without_heading.set_fn_note("wout_head")

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        without_heading.set_norm_on(False)
        with_heading.set_norm_on(True)

        save_location = get_file_id_for_exp(dash_folder, "norm-" + without_heading.get_exp_label())

        verts_with_heading, us_with_heading, cost_with_heading, info_packet1 = solver.run_solver(with_heading)
        verts_wout_heading, us_wout_heading, cost_wout_heading, info_packet2 = solver.run_solver(without_heading)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8, 4))

        cost_wout_heading.get_overview_pic(verts_wout_heading, us_wout_heading, ax=ax1, dash_folder=dash_folder)
        cost_with_heading.get_overview_pic(verts_with_heading, us_with_heading, ax=ax2, dash_folder=dash_folder)

        blurb1 = without_heading.get_solver_status_blurb()
        blurb2 = with_heading.get_solver_status_blurb()

        test_log.append(without_heading.get_solve_quality_status(test_group))
        test_log.append(with_heading.get_solve_quality_status(test_group))

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Without OOS Normalization \n" + blurb1, fontweight='bold')
        _ = ax2.set_title("With Norm\n" + blurb2, fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        fig.suptitle("Goal = " + with_heading.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)


def test_weighted_by_distance_or_no(dash_folder, scenario_filters):
    print("TESTING IF WEIGHTED NEAR MATTERS")
    test_group = "weighted by distance?"
    scenarios = test_scenarios.get_scenarios(scenario_filters)

    # for each test scenario
    # run it with heading, get the image
    # run it without heading part, get the image

    # export both individually
    # export side by side with markings
    #    include exp settings on export

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        with_heading = copy.copy(scenario)
        without_heading = copy.copy(scenario)

        with_heading.set_fn_note("even_wt")
        without_heading.set_fn_note("dist_wt")

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        without_heading.set_weighted_close_on(False)
        with_heading.set_weighted_close_on(True)

        save_location = get_file_id_for_exp(dash_folder, "wt-dist-" + without_heading.get_exp_label())

        verts_with_heading, us_with_heading, cost_with_heading, info_packet1 = solver.run_solver(with_heading)
        verts_wout_heading, us_wout_heading, cost_wout_heading, info_packet2 = solver.run_solver(without_heading)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8, 3.5))

        cost_wout_heading.get_overview_pic(verts_wout_heading, us_wout_heading, ax=ax1, dash_folder=dash_folder)
        cost_with_heading.get_overview_pic(verts_with_heading, us_with_heading, ax=ax2, dash_folder=dash_folder)

        blurb1 = without_heading.get_solver_status_blurb()
        blurb2 = with_heading.get_solver_status_blurb()

        test_log.append(without_heading.get_solve_quality_status(test_group))
        test_log.append(with_heading.get_solve_quality_status(test_group))

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Even weighting of goals \n" + blurb1, fontweight='bold')
        _ = ax2.set_title("Weight by closeness \n" + blurb2, fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        fig.suptitle("Goal = " + scenario.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)



def test_obstacles_being_avoided(dash_folder, scenario_filters):
    scenarios = test_scenarios.get_scenarios_obstacles(scenario_filters)
    test_group = 'obstacles avoided?'

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        obs_scenario = copy.copy(scenario)

        obs_scenario.set_fn_note("obs-" + key)

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        obs_scenario.set_heading_on(True)
        save_location = get_file_id_for_exp(dash_folder, "obs-" + obs_scenario.get_exp_label())

        verts_with_obs, us_with_obs, cost_with_obs, info_packet = solver.run_solver(obs_scenario)

        blurb = obs_scenario.get_solver_status_blurb()

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time,
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1) = plt.subplots(ncols=1, figsize=(4, 3))

        cost_with_obs.get_overview_pic(verts_with_obs, us_with_obs, ax=ax1, dash_folder=dash_folder)
        _ = ax1.set_title("Check for Obstacle Issues\n" + blurb, fontweight='bold')

        test_log.append(obs_scenario.get_solve_quality_status(test_group))

        plt.tight_layout()
        fig.suptitle("Goal = " + scenario.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)



def test_observers_being_respected(dash_folder, scenario_filters):
    # for each test scenario
    # compare with observer vs no
    scenarios = test_scenarios.get_scenarios_observers(scenario_filters)
    test_group = "obervers respected?"

    for key in scenarios.keys():
        scenario = scenarios[key]
        scenario.set_run_filters(scenario_filters)

        with_oa = copy.copy(scenario)
        wout_oa = copy.copy(scenario)

        with_oa.set_fn_note("with_oa")
        wout_oa.set_fn_note("wout_oa")

        # RUN THE SOLVER WITH CONSTRAINTS ON EACH
        with_oa.set_oa_on(True)
        wout_oa.set_oa_on(False)
        
        save_location = get_file_id_for_exp(dash_folder, "oa-" + wout_oa.get_exp_label())

        verts_with_oa, us_with_oa, cost_with_oa, info_packet1 = solver.run_solver(with_oa)
        verts_wout_oa, us_wout_oa, cost_wout_oa, info_packet2 = solver.run_solver(wout_oa)

        # This placement of the figure statement is actually really important
        # numpy only likes to have one plot open at a time, 
        # so this is a fresh one not dependent on the graphing within the solver for each
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8, 6))

        cost_wout_oa.get_overview_pic(verts_wout_oa, us_wout_oa, ax=ax1, dash_folder=dash_folder)
        cost_with_oa.get_overview_pic(verts_with_oa, us_with_oa, ax=ax2, dash_folder=dash_folder)

        # _ = ax1.set_xlabel("Time", fontweight='bold')
        # _ = ax1.set_ylabel("Legibility", fontweight='bold')
        _ = ax1.set_title("Omniscient", fontweight='bold')
        _ = ax2.set_title("Observer-Aware", fontweight='bold')
        # ax2.legend() #loc="upper left")
        # ax2.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        fig.suptitle("Goal = " + cost_wout_oa.get_goal_label())
        plt.savefig(save_location + ".png")
        plt.close('all')

        collate_and_report_on_results(dash_folder)

def setup_axes_for_test_setups(test_setups_og):
    if len(test_setups_og) > 6:
        fig, axes = plt.subplot_mosaic("ABC;DEF;IGH", figsize=(8, 6), gridspec_kw={'height_ratios':[1, 1, 1], 'width_ratios':[1, 1, 1]})
        ax_mappings = {}
        ax_mappings[0] = axes['A']
        ax_mappings[1] = axes['B']
        ax_mappings[2] = axes['C']
        ax_mappings[3] = axes['D']
        ax_mappings[4] = axes['E']
        ax_mappings[5] = axes['F']
        ax_mappings[6] = axes['G']
        ax_mappings[7] = axes['H']
        ax_mappings[8] = axes['I']
        axes['I'].axis('off')
    elif len(test_setups_og) == 4:
        fig, axes = plt.subplot_mosaic("ABCD", figsize=(8, 4), gridspec_kw={'height_ratios':[1], 'width_ratios':[1, 1, 1, 1]})
        ax_mappings = {}
        ax_mappings[0] = axes['A']
        ax_mappings[1] = axes['B']
        ax_mappings[2] = axes['C']
        ax_mappings[3] = axes['D']
    elif len(test_setups_og) == 1:
        fig, axes = plt.subplot_mosaic("A", figsize=(4, 4), gridspec_kw={'height_ratios':[1], 'width_ratios':[1]})
        ax_mappings = {}
        ax_mappings[0] = axes['A']
    elif len(test_setups_og) < 4:
        fig, axes = plt.subplot_mosaic("ABC", figsize=(8, 4), gridspec_kw={'height_ratios':[1], 'width_ratios':[1, 1, 1]})
        ax_mappings = {}
        ax_mappings[0] = axes['A']
        ax_mappings[1] = axes['B']
        ax_mappings[2] = axes['C']
    else:
        fig, axes = plt.subplot_mosaic("ABC;DEF", figsize=(8, 6), gridspec_kw={'height_ratios':[1, 1], 'width_ratios':[1, 1, 1]})
        ax_mappings = {}
        ax_mappings[0] = axes['A']
        ax_mappings[1] = axes['B']
        ax_mappings[2] = axes['C']
        ax_mappings[3] = axes['D']
        ax_mappings[4] = axes['E']
        ax_mappings[5] = axes['F']

    return fig, axes, ax_mappings


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

        rot_scenario.set_fn_note("rot_" + str(new_angle))

        scenario_dict[offset] = rot_scenario

    return scenario_dict

def main():
    run_all_tests()

if __name__ == "__main__":
    main()

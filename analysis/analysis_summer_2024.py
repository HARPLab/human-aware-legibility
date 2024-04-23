import os
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt  

pd.set_option('display.max_colwidth', None)
np.set_printoptions(suppress=True)


##### Import keypress file data
##### Next: shillelagh to import directly from file?
# keypress_file = open(keypress_path)
# for line in keypress_file.readlines()[1:]:
#     data = line.replace('\n', "")
#     data = data.split(",")
#     clean_data = [int(data[0]), data[1]]

#     keypress_data.append(clean_data)
# keypress_file.close()

FLAG_ANALYZE_EXPANDED   = False

keypress_path           = "keypress_log.csv"

keypress_data = []
df_keypress = pd.read_csv(keypress_path)
df_keypress = df_keypress.astype({"timestamp": int, "guess": str})


##### Import the robot movement data
dir_list    = os.listdir("mission_reports/")
dir_list    = [f for f in dir_list if 'csv' in f]

mini_list       = [f for f in dir_list if 'mini' in f]
mission_list    = [f for f in dir_list if 'mission' in f]

print("IMPORTING")
print(mini_list)
print(mission_list)

titles_dict = {'AB': 'back_short_to', 'BA': 'back_short_from', 'BC': 'back_short_from', 'CB': 'back_short_to', 'DE': 'front_short_to', 'ED': 'front_short_from', 'EF': 'front_short_from', 'FE': 'front_short_to', 'AC': 'back_long', 'CA': 'back_long', 'DF': 'front_long', 'FD': 'front_long', 'DC': 'diag_long_to', 'CD': 'diag_long_from', 'AF': 'diag_long_to', 'FA': 'diag_long_from', 'AE': 'diag_short_to', 'EA': 'diag_short_from', 'CE': 'diag_short_to', 'EC': 'diag_short_from', 'BE': 'mid_short', 'EB': 'mid_short', 'AD': 'side_short_to', 'DA': 'side_short_from', 'CF': 'side_short_to', 'FC': 'side_short_from', 'BD': 'back_diag_short_from', 'BF': 'back_diag_short_from', 'DB': 'back_diag_short_to', 'FB': 'back_diag_short_to', 'DC_OBS': 'diag_obs_long_to', 'CD_OBS': 'diag_obs_long_from', 'AF_OBS': 'diag_obs_long_to', 'FA_OBS': 'diag_obs_long_from', 'AC_OBS': 'back_long_obs', 'CA_OBS': 'back_long_obs', 'DF_OBS': 'front_long_obs', 'FD_OBS': 'front_long_obs'}
label_dict = {'back_short_to': ['AB', 'CB'], 'back_short_from': ['BA', 'BC'], 'front_short_to': ['DE', 'FE'], 'front_short_from': ['ED', 'EF'], 'back_long': ['AC', 'CA'], 'front_long': ['DF', 'FD'], 'diag_long_to': ['DC', 'AF'], 'diag_long_from': ['CD', 'FA'], 'diag_short_to': ['AE', 'CE'], 'diag_short_from': ['EA', 'EC'], 'mid_short': ['BE', 'EB'], 'side_short_to': ['AD', 'CF'], 'side_short_from': ['DA', 'FC'], 'back_diag_short_from': ['BD', 'BF'], 'back_diag_short_to': ['DB', 'FB'], 'diag_obs_long_to': ['DC_OBS', 'AF_OBS'], 'diag_obs_long_from': ['CD_OBS', 'FA_OBS'], 'back_long_obs': ['AC_OBS', 'CA_OBS'], 'front_long_obs': ['DF_OBS', 'FD_OBS']}

df_chunks_mini      = []
df_chunks_mission   = []

trial_names = []
### SIMPLIFIED DATA
for file_name in mini_list:
    waypoints_path = "mission_reports/" + file_name
    path_name = file_name.replace('.csv','')

    keypress_file   = open(waypoints_path)    
    df_chunk        = pd.read_csv(waypoints_path, index_col=False) #, columns=['path_label', 'status', 'timestamp', 'time_elapsed'])
    df_chunk = df_chunk.rename(columns={" status": "status", " time": "time"})

    df_chunk['trial']       = df_chunk['point'].str.replace("[", "").str.replace("'", "")
    df_chunk['status']      = df_chunk['status'].str.replace("'", "")
    df_chunk['time']        = df_chunk['time'].str.replace("'", "")

    df_chunk['participant_id'] = file_name
    df_chunk[['path_name', 'path_style']] = df_chunk['trial'].str.split('-', n=1, expand=True)

    ### Label with the matching mission file
    ### If needed, also put participant number on here
    df_chunk["path_type"] = df_chunk['path_name'].replace(titles_dict)
    df_chunk["path_type_unique_id"] = df_chunk['path_type'] + "-" + df_chunk['path_style'] + " @ " + df_chunk['participant_id']

    df_chunk['is_ambig'] = df_chunk['path_type'].isin(['diag_long_to', 'diag_long_from', 'back_long', 'front_long', 'diag_obs_long_to', 'diag_obs_long_from'])
    
    df_chunk = df_chunk.loc[:,['time', 'trial', 'path_name', 'path_style', 'path_type', 'status', 'is_ambig', 'participant_id', 'path_type_unique_id']]

    trial_names.append(path_name)
    df_chunks_mini.append(copy.copy(df_chunk))


df_robot_trials = pd.concat(df_chunks_mini)


if FLAG_ANALYZE_EXPANDED:
    ### FULL DATA
    for file_name in mission_list:
        waypoints_path = "mission_reports/" + file_name
        path_name = file_name.replace('.csv','')

        keypress_file   = open(waypoints_path)
        ## TODO: clean it first of parking flags?
        df_chunk        = pd.read_csv(waypoints_path) #, columns=['path_label', 'status', 'timestamp', 'time_elapsed'])

        ### Label with the matching mission file
        ### If needed, also put participant number on here
        df_chunk['trial_group'] = path_name
        
        trial_names.append(path_name)
        df_chunks_mission.append(copy.copy(df_chunk))

##### Turn them both into a single Pandas?


##### Run comparisons across symmetrical sets?
# For each trial in the deck
# match up the trial with button presses along the way
# Also match up with locations in space via expanded set

rows_for_analysis = []

trials_for_analysis = df_robot_trials['path_type_unique_id'].unique()
# print("TRIALS FOR ANALYSIS")
# print(trials_for_analysis)

weird_recordings = []

results_list = []
for trial in trials_for_analysis:
    # Get the start and end points

    df_relevant_data = df_robot_trials[df_robot_trials['path_type_unique_id'] == trial]
    # 'time', 'trial', 'path_name', 'path_type', 'status', 'participant_id', 'path_type_unique_id'

    # print(df_relevant_data[['path_name', 'path_style', 'status', 'path_type']])
    # PREP, START, END, PARKED

    # print(df_relevant_data['status'])

    try:
        timestamp_prep      = df_relevant_data[df_relevant_data['status'].str.contains('PREP')]['time'].item()
        timestamp_start     = df_relevant_data[df_relevant_data['status'].str.contains('START')]['time'].item()
        timestamp_end       = df_relevant_data[df_relevant_data['status'].str.contains('END')]['time'].item()

        timestamp_prep      = float(timestamp_prep)
        timestamp_start     = float(timestamp_start)
        timestamp_end       = float(timestamp_end)
    except:
        print("Issue with collecting all statuses for " + trial)
        weird_recordings.append(trial)

        # timestamp_start     = df_relevant_data[df_relevant_data['status'].str.contains('START')]['time'].item()
        # timestamp_end       = df_relevant_data[df_relevant_data['status'].str.contains('END')]['time'].item()
        continue

    # print(timestamp_prep, timestamp_start, timestamp_end)
    # exit()

    time_conversion = 1000000000 # ROS is much more precise, so convert between them
    timestamp_prep  /= time_conversion
    timestamp_start /= time_conversion
    timestamp_end   /= time_conversion

    ### skim those zones from the button click database
    # print(timestamp_prep, timestamp_start, timestamp_end)

    ## Get the relevant clicks
    df_clicks = df_keypress[df_keypress['timestamp'].between(timestamp_prep, timestamp_end, inclusive='both')]
    # print(df_relevant_data[['path_name', 'path_style', 'status', 'path_type']])
    # print(df_clicks)
    # print("")

    path_name = df_relevant_data['path_name'].unique()[0]
    correct_answer = path_name[1]
    # print("Correct answer")
    # print(correct_answer)

    path_type   = df_relevant_data['path_type'].unique()[0]
    path_style  = df_relevant_data['path_style'].unique()[0]

    with_obs = df_relevant_data['path_name'].unique()[0]
    with_obs = "OBS" in with_obs

    # Guess time offset from start
    for index, click in df_clicks.iterrows():
        click_time          = click['timestamp']
        click_guess         = click['guess']

        phase = "NONE"
        if click_time > timestamp_prep and click_time < timestamp_start:
            phase = "SETUP"
        elif click_time > timestamp_start and click_time < timestamp_end:
            phase = "MAIN"
        else:
            ### How many times do people ring in after the robot is at the goal?s
            pass

        result = [click_guess, timestamp_end - click_time, (correct_answer == click_guess), with_obs, phase, path_type, path_style, trial]

        results_list.append(result)

df_results = pd.DataFrame(results_list, columns=['guess', 'time_before_end', 'is_correct', 'with_obs', 'phase', 'path_type', 'path_style', 'trial'])

##### Graph the results
path_style_options  = list(df_results['path_style'].unique())
path_type_options   = list(df_results['path_type'].unique())


df_results_correct = df_results[df_results['is_correct'] == True]
df_pivot_correct_all = df_results_correct.pivot_table(values='time_before_end', index=['path_type'], columns='path_style', aggfunc='mean')
df_pivot_correct_all.to_csv("graphics/" + "avgs.csv")


df_results_incorrect = df_results[df_results['is_correct'] == False]
df_pivot_incorrect_all = df_results_incorrect.pivot_table(values='is_correct', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
df_pivot_incorrect_all.to_csv("graphics/" + "miscues.csv")

df_results_diag_long = df_results[df_results['path_type'].isin(['diag_long_to', 'diag_long_from'])]
# df_pivot_incorrect_all = df_results_incorrect.pivot_table(values='is_correct', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
# df_pivot_incorrect_all.to_csv("graphics/" + "miscues.csv")



export_location = 'graphics/'
for pt in path_type_options:
    df_inspect = df_results[df_results['path_type'] == pt]

    df_inspect_last_correct     =  df_inspect[df_inspect['is_correct'] == True]
    df_inspect_all_incorrect    =  df_inspect[df_inspect['is_correct'] == False]

    print(df_inspect[['time_before_end', 'path_type', 'path_style', 'is_correct']])

    if len(df_inspect_last_correct) > 0:
        fig = plt.figure();
        bp = df_inspect_last_correct.boxplot(by =['path_style', 'with_obs'], column =['time_before_end']) #, by=['time_before_end']
        bp.set_title(pt)
        plt.savefig(export_location + pt + ".png")
        plt.clf()


    fig, ax = plt.subplots(1)
    df_pivot_incorrect = df_inspect_all_incorrect.pivot_table(values='is_correct', index=['path_style'], aggfunc='count', fill_value=0)

    # df_inspect.groupby(['is_correct']).count().plot(kind='bar')
    # counts = df_inspect_all_incorrect[['path_style', 'is_correct']].groupby(['path_style', 'is_correct']).agg(len)
    # print(counts)
    # if len(counts) > 0:

    # print("PIVOT TABLE")
    # print(df_pivot_incorrect)

    if len(df_pivot_incorrect) > 0:
        df_pivot_incorrect.plot(kind='bar')

        # bp = df_inspect_all_incorrect.boxplot(by =['path_style', 'with_obs'], column =['time_before_end']) #, by=['time_before_end']
        bp.set_title(pt)
        ax.set_ylim(ymin=0)
        plt.savefig(export_location + "misclicks-" + pt + ".png")
        plt.clf()








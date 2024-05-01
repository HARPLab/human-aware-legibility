import os
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt  
import pingouin as pg
from datetime import datetime as dt

pd.set_option('display.max_colwidth', None)
np.set_printoptions(suppress=True)
plt.rcParams.update({'figure.max_open_warning': 0})


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
df_keypress = pd.read_csv(keypress_path)
df_keypress = df_keypress.astype({"timestamp": int, "guess": str})
df_keypress['time'] = df_keypress['timestamp']

distractor_path           = "distractors.csv"
df_distractors  = pd.read_csv(keypress_path)
# df_keypress = df_keypress.astype({"timestamp": int, "guess": str})
# df_keypress['time'] = df_keypress['timestamp']


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

ambig_list = ['diag_long_to', 'diag_long_from', 'back_long', 'front_long', 'diag_obs_long_to', 'diag_obs_long_from', 'front_long_obs', 'back_long_obs']

df_chunks_mini      = []
df_chunks_mission   = []

def add_trial_numbers(df):
    # df = df.assign(iteration_number=range(len(df)))
    # df['iteration_number'] = pd.Series(df_chunk['iteration_number'], dtype="int")

    count = 1
    df_chunks = []
    num_splits = (len(df) / 4) + 1
    for chunk in np.array_split(df, num_splits):
        chunk['iteration_number'] = count
        count += 1

        df_chunks.append(chunk)

    df = pd.concat(df_chunks)
    # exit()
    return df

def format_duration(duration):
    duration = duration / 1000000000.0

    mapping = [
        ('s', 60),
        ('m', 60),
        ('h', 24),
    ]
    duration = int(duration)
    result = []
    for symbol, max_amount in mapping:
        amount = duration % max_amount
        result.append(f'{amount}{symbol}')
        duration //= max_amount
        if duration == 0:
            break

    if duration:
        result.append(f'{duration}d')

    return ' '.join(reversed(result))

ellie_list = ['2024_04_19-09_37_28_PM-mini_report.csv', '2024_04_19-10_02_13_PM-mini_report.csv']


trial_names = []
### SIMPLIFIED DATA
for file_name in mini_list:
    waypoints_path = "mission_reports/" + file_name
    path_filename = file_name.replace('.csv','')
    # print(path_name)

    keypress_file   = open(waypoints_path)
    df_chunk        = pd.read_csv(waypoints_path, index_col=False) #, columns=['path_label', 'status', 'timestamp', 'time_elapsed'])
    df_chunk = df_chunk.rename(columns={" status": "status", " time": "time", " iteration_number": "iteration_number"})

    if len(df_chunk) == 0:
        continue

    time_last   = int(df_chunk['time'].max())
    time_first  = int(df_chunk['time'].min())
    # time_last   = dt.fromtimestamp(time_last)
    # time_first  = dt.fromtimestamp(time_first)

    total_time = time_last - time_first
    # print("Total trial time of " + str(total_time))

    print(path_filename)
    print("\t " + format_duration(total_time))

    # if file_name in ellie_list:
    #     df_chunk = add_trial_numbers(df_chunk)

    df_chunk['trial']       = df_chunk['point'].str.replace("[", "").str.replace("'", "")
    df_chunk['status']      = df_chunk['status'].str.replace("'", "")
    
    if df_chunk['time'].dtype == str:
        df_chunk['time']        = df_chunk['time'].str.replace("'", "")

    df_chunk['participant_id'] = path_filename
    df_chunk[['path_name', 'path_style']] = df_chunk['trial'].str.split('-', n=1, expand=True)

    df_chunk['iteration_string'] = pd.Series(df_chunk['iteration_number'], dtype="string")

    ### Label with the matching mission file
    ### If needed, also put participant number on here
    df_chunk["path_type"] = df_chunk['path_name'].replace(titles_dict)
    df_chunk["path_type_unique_id"] = df_chunk['path_name'] + "-" + df_chunk['path_style'] + "-" + df_chunk['iteration_string'] + " @ " + df_chunk['participant_id']
    df_chunk["path_type_unique_id"] = pd.Series(df_chunk['path_type_unique_id'], dtype="string")

    df_chunk['is_ambig'] = df_chunk['path_type'].isin(ambig_list)
    df_chunk = df_chunk.loc[:,['time', 'trial', 'path_name', 'path_style', 'path_type', 'status', 'is_ambig', 'iteration_number', 'participant_id', 'path_type_unique_id']]

    df_chunk['is_long']     = df_chunk['path_type'].str.contains("long")
    df_chunk['is_short']    = df_chunk['path_type'].str.contains("short")
    df_chunk['respect_obs'] = df_chunk['path_type'].str.contains("obs")
    
    trial_names.append(path_filename)
    df_chunks_mini.append(copy.copy(df_chunk))


df_robot_trials = pd.concat(df_chunks_mini)

if FLAG_ANALYZE_EXPANDED:
    ### FULL DATA
    for file_name in mission_list:
        waypoints_path = "mission_reports/" + file_name
        path_name = file_name.replace('.csv','')

        ## TODO: clean it first of parking flags?
        df_moves        = pd.read_csv(waypoints_path) #, columns=['path_label', 'status', 'timestamp', 'time_elapsed'])

        ### Label with the matching mission file
        ### If needed, also put participant number on here
        df_moves['trial_group'] = path_name
        
        trial_names.append(path_name)
        df_chunks_mission.append(copy.copy(df_chunk))

##### Turn them both into a single Pandas?


##### Run comparisons across symmetrical sets?
# For each trial in the deck
# match up the trial with button presses along the way
# Also match up with locations in space via expanded set

rows_for_analysis = []

df_weird = copy.copy(df_robot_trials)


order_style = ["early", "even", "late"]
order_status = ["PREP", "START", "END", "PARKED"]

# df_weird["path_style"] = pd.Categorical(df_weird["path_style"], categories=order_style)
# df_weird = df_weird.sort_values('path_style')
# df_weird["status"] = pd.Categorical(df_weird["status"], categories=order_status)
# df_weird = df_weird.sort_values('status')

# df_weird["path_style"]  = df_weird["path_style"].astype(pd.api.types.CategoricalDtype(categories=["early", "even", "late"]))
# df_weird["status"]      = df_weird["status"].astype(pd.api.types.CategoricalDtype(categories=["PREP", "START", "END", "PARK"]))
# df_weird["path_style"].cat.set_categories(order_style)
# df_weird["status"].cat.set_categories(order_status)

# df_weird['path_style']  = pd.Categorical(df_weird['path_style'], categories=["early", "even", "late"], ordered=True)
# df_weird['status']      = pd.Categorical(df_weird['status'], categories=["PREP", "START", "END", "PARK"], ordered=True)
df_weird = df_weird.pivot_table(values='time', index=['path_name'], columns=['path_style'], aggfunc='count', fill_value=0)


# new_index = pd.MultiIndex.from_product(
#     [order_style, order_status], 
#     names=['path_style', 'status']
# )
# print(new_index)
# df_weird.reindex(new_index)
# df_weird = df_weird.reindex(axis='index', level=0, labels=yourlabels_list)

df_weird.to_csv('graphics/csvs/' + "weird_situations.csv")

trials_for_analysis = list(df_robot_trials['path_type_unique_id'].dropna().unique())

# print("TRIALS FOR ANALYSIS")
# print(trials_for_analysis)

weird_recordings_no_start   = []
weird_recordings_weirder    = []

results_list = []
for trial in trials_for_analysis:
    # Get the start and end points

    df_relevant_data = df_robot_trials[df_robot_trials['path_type_unique_id'] == trial]
    # print(df_relevant_data)
    # 'time', 'trial', 'path_name', 'path_type', 'status', 'participant_id', 'path_type_unique_id'

    # print(df_relevant_data[['path_name', 'path_style', 'status', 'path_type']])
    # PREP, START, END, PARKED

    # What if there are multiple to examine?
    try:
        timestamp_prep      = df_relevant_data[df_relevant_data['status'].str.contains('PREP')]['time'].item()
        timestamp_start     = df_relevant_data[df_relevant_data['status'].str.contains('START')]['time'].item()
        timestamp_end       = df_relevant_data[df_relevant_data['status'].str.contains('END')]['time'].item()

    except:
        # print("Issue with collecting all statuses for " + trial)
        # weird_recordings.append(trial)
        # print(df_relevant_data[['time', 'status']])

        # try:
        timestamp_prep      = df_relevant_data[df_relevant_data['status'].str.contains('PREP')]['time'].item()
        timestamp_start     = timestamp_prep #df_relevant_data[df_relevant_data['status'].str.contains('START')]['time'].item()
        timestamp_end       = df_relevant_data[df_relevant_data['status'].str.contains('END')]['time'].item()

        weird_recordings_no_start.append(trial)
        # print(df_relevant_data[['time', 'status']])
        # print(timestamp_prep, timestamp_start, timestamp_end)

    try:
        timestamp_parked    = df_relevant_data[df_relevant_data['status'].str.contains('PARK')]['time'].item()
    except:
        timestamp_parked    = df_relevant_data[df_relevant_data['status'].str.contains('END')]['time'].item()

    timestamp_prep      = float(timestamp_prep)
    timestamp_start     = float(timestamp_start)
    timestamp_end       = float(timestamp_end)

    time_conversion     = 1000000000 # ROS is much more precise, so convert between them
    timestamp_prep      /= time_conversion
    timestamp_start     /= time_conversion
    timestamp_end       /= time_conversion
    timestamp_parked    /= time_conversion

    ### skim those zones from the button click database
    # print(timestamp_prep, timestamp_start, timestamp_end)

    ## Get the relevant clicks
    df_clicks = df_keypress[df_keypress['timestamp'].between(timestamp_prep, timestamp_parked, inclusive='both')]
    # print(df_relevant_data[['path_name', 'path_style', 'status', 'path_type']])
    # print(df_clicks)
    # print("")

    path_name = df_relevant_data['path_name'].unique()[0]
    correct_answer = path_name[1]
    # print("Correct answer")
    # print(correct_answer)

    path_name   = df_relevant_data['path_name'].unique()[0]
    path_type   = df_relevant_data['path_type'].unique()[0]
    path_style  = df_relevant_data['path_style'].unique()[0]

    if len(df_clicks['distractor'].unique()) > 0:
        is_distracted = df_clicks['distractor'].unique()[0]
    else:
        is_distracted = False

    with_obs = df_relevant_data['path_name'].unique()[0]
    with_obs = "OBS" in with_obs

    # Guess time offset from start
    ## move backwards through the thing

    has_final, has_final_park = False, False

    to_add = []
    is_final = True
    guess_from_end = None

    is_final_park = True
    guess_from_park = None

    participant_id = df_relevant_data['participant_id'].unique()[0]

    for index, click in df_clicks.iloc[::-1].iterrows():
        click_time          = click['timestamp']
        click_guess         = click['guess']

        phase = "NONE"
        if click_time > timestamp_prep and click_time < timestamp_start:
            phase = "SETUP"
        elif click_time > timestamp_start and click_time < timestamp_end:
            phase = "MAIN"
        elif click_time > timestamp_end and click_time < timestamp_parked:
            phase = "PARKING"
        else:
            ### How many times do people ring in after the robot is at the goal?s
            pass

        guess_from_end = click_guess

        result = [click_guess, click_time, timestamp_end - click_time, (timestamp_parked - click_time), (correct_answer == click_guess), is_final, is_final_park, is_distracted, with_obs, phase, path_name, path_type, path_style, trial, participant_id]
        to_add.append(result)

        ### TODO ADA HAS_FINAL MARK ONLY THE LAST
        if is_final: # and (guess_from_end == correct_answer):
            is_final = False

        if is_final_park:
            is_final_park = False


    results_list.extend(to_add[::-1])

    ### The last correct guess is_final


####### SET UP RESULTS #######################
df_results = pd.DataFrame(results_list, columns=['guess', 'time', 'time_before_end', 'time_before_park', 'is_correct', 'is_final', 'is_final_park', 'is_distracted', 'with_obs', 'phase', 'path_name', 'path_type', 'path_style', 'trial', 'participant_id'])

### Add sorting labels for convenience ###
df_results['is_ambig'] = df_results['path_type'].isin(ambig_list)
df_results['is_long']     = df_results['path_type'].str.contains("long")
df_results['is_short']    = df_results['path_type'].str.contains("short")
df_results['respect_obs'] = df_results['path_type'].str.contains("obs")

df_results.to_csv("results.csv")
############################################

print("QUALITY CHECKS")

############ QUALITY CHECK
for participant_id in list(df_results['participant_id'].unique()):

    print("+++++ " + participant_id)
    df_part = df_results[df_results['participant_id'] == participant_id]
    part_time_last   = df_part['time'].max()
    part_time_first  = df_part['time'].min()

    key_time_last   = df_keypress['timestamp'].max()
    key_time_first  = df_keypress['timestamp'].min()

    df_clicks = df_keypress[df_keypress['timestamp'].between(part_time_first, part_time_last, inclusive='both')]

    # print(part_time_first, part_time_last)
    # print(key_time_first, key_time_last)

    # print(len(df_clicks))
    # print(len(df_part))

    keypresses  = (df_clicks['timestamp'].unique())
    guesses     = df_part['time'].unique()

    remaining = keypresses - guesses
    # print("REMAINING")
    # print(remaining)

    # print(len(keypresses))
    # print(len(guesses))




    # duration / 1000000000.0

####################################


####### PIVOT TABLES #######################
### TODO MAKE THIS BETTER SO JUST WHETHER THERE'S MORE THAN ONE CLICK PER
df_pivot_count = df_results.pivot_table(values='trial', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
df_pivot_count.to_csv("graphics/csvs/" + "raw_counts.csv")

##### Graph the results
path_style_options  = list(df_results['path_style'].unique())
path_type_options   = list(df_results['path_type'].unique())

df_results_correct = df_results[df_results['is_correct'] == True]
df_results_correct = df_results_correct[df_results_correct['is_final'] == True]
df_pivot_correct_all = df_results_correct.pivot_table(values='time_before_end', index=['path_type'], columns='path_style', aggfunc='mean')
df_pivot_correct_all.to_csv("graphics/csvs/" + "avgs-end.csv")
df_pivot_correct_all = df_results_correct.pivot_table(values='time_before_end', index=['path_type'], columns='path_style', aggfunc='count')
df_pivot_correct_all.to_csv("graphics/csvs/" + "correct-before-end.csv")


df_results_incorrect = df_results[df_results['is_correct'] == False]
df_pivot_incorrect_all = df_results_incorrect.pivot_table(values='is_correct', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
df_pivot_incorrect_all.to_csv("graphics/csvs/" + "miscues.csv")

df_results_ambig = df_results_incorrect.pivot_table(values='time_before_end', index=['path_type'], columns=['with_obs', 'path_style'], aggfunc='mean', fill_value=0)
df_results_ambig.to_csv('graphics/csvs/' + "ambig-end.csv")

df_inspect_correct              =  df_results[df_results['is_correct'] == True]
df_inspect_early_confused       =  df_inspect_correct[df_inspect_correct['is_final'] == False]
df_pivot_right_wrong            = df_inspect_early_confused.pivot_table(values='is_correct', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
df_pivot_right_wrong.to_csv('graphics/csvs/' + "right_then_wrong.csv")

df_results_final        = df_results[df_results['is_final'] == True]
df_results_final_corr   = df_results_final[df_results_final['is_correct'] == True]


############# Get some stats based on distractors
### get the click sequence for each participant
df_pivot_distraction = df_results.pivot_table(values='time_before_end', index=['path_type'], columns=['path_style', 'is_distracted'], aggfunc='mean', fill_value=0)
df_pivot_distraction.to_csv('graphics/csvs/' + "time_before_end-distractors.csv")

### get the click sequence for each participant
df_pivot_distraction = df_results.pivot_table(values='time_before_end', index=['path_style'], columns=['is_distracted'], aggfunc='mean', fill_value=0)
df_pivot_distraction.to_csv('graphics/csvs/' + "time_before_end-distractors-concise.csv")

### What percent of the time do observers know the robot's correct goal by the end of the path?
df_pct = df_results_final.groupby(['path_type', 'path_style'])['is_correct'].mean()
df_pct.to_csv('graphics/csvs/' + "pct-final-correct.csv")

##########################

# Check if 

#########################
for participant_id in list(df_results['participant_id'].unique()):
    participant_id_short = participant_id.replace("-mini_report", "")

    path = "individuals/" + participant_id_short + "/"
    # Check whether the specified path exists or not
    path_exists = os.path.exists(path)
    if not path_exists:
       # Create a new directory because it does not exist
       os.makedirs(path)
    export_loco = path

    df_part = df_results[df_results['participant_id'] == participant_id]
    df_part.to_csv(export_loco + participant_id_short + "-all-clicks.csv")

    df_part_final = df_results[df_results['is_final'] == True]
    df_part_final = df_part_final.groupby(['path_name', 'path_style']).agg({'guess': list})
    df_part_final.to_csv(export_loco + participant_id_short + "-finals.csv")


    ### get the click sequence for each participant
    df_pivot_count = df_part.pivot_table(values='trial', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
    df_pivot_count.to_csv(export_loco + participant_id_short + "-raw_counts.csv")

    ### get the click sequence for each participant
    df_pivot_count = df_part.pivot_table(values='time_before_end', index=['path_type'], columns='path_style', aggfunc='mean', fill_value=0)
    df_pivot_count.to_csv(export_loco + participant_id_short + "-time_before_end.csv")

    ### get the click sequence for each participant
    # df_guesses = df_part.pivot_table(values='guess', index=['path_type'], columns='path_style', aggfunc='join')
    df_guesses = df_part.groupby(['path_type', 'path_style']).agg({'is_correct': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-correctness.csv")

    df_guesses = df_part.groupby(['path_name', 'path_style']).agg({'guess': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-raw-guesses.csv")

    df_guesses = df_part.groupby(['path_name', 'path_style']).agg({'guess': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-raw-guesses.csv")

    ### get the click sequence for each participant
    # df_guesses = df_part.pivot_table(values='guess', index=['path_type'], columns='path_style', aggfunc='join')
    df_guesses = df_part[df_part['is_long'] == True].groupby(['path_type', 'path_style']).agg({'is_correct': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-long-guesses.csv")


    ### get the click sequence for each participant
    # df_guesses = df_part.pivot_table(values='guess', index=['path_type'], columns='path_style', aggfunc='join')
    df_guesses = df_part[df_part['is_long'] == False].groupby(['path_type', 'path_style']).agg({'is_correct': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-short-guesses.csv")

    ### get the click sequence for each participant
    # df_guesses = df_part.pivot_table(values='guess', index=['path_type'], columns='path_style', aggfunc='join')
    df_guesses = df_part[df_part['is_ambig'] == True].groupby(['path_type', 'path_style']).agg({'is_correct': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-ambig-guesses.csv")

    ### get the click sequence for each participant
    # df_guesses = df_part.pivot_table(values='guess', index=['path_type'], columns='path_style', aggfunc='join')
    df_guesses = df_part[df_part['is_ambig'] == False].groupby(['path_type', 'path_style']).agg({'is_correct': list})
    df_guesses.to_csv(export_loco + participant_id_short + "-unambig-guesses.csv")


#########################
### COMPARE THE SUBGROUPS
df_results_diag_long = df_results[df_results['path_type'].isin(['diag_long_to', 'diag_long_from'])]
# df_pivot_incorrect_all = df_results_incorrect.pivot_table(values='is_correct', index=['path_type'], columns='path_style', aggfunc='count', fill_value=0)
# df_pivot_incorrect_all.to_csv("graphics/" + "miscues.csv")



fig, ax = plt.subplots(1)

export_location = 'graphics/'
for pt in path_type_options:
    df_inspect = df_results[df_results['path_type'] == pt]

    df_inspect_last_correct     =  df_inspect[df_inspect['is_correct'] == True]
    df_inspect_all_incorrect    =  df_inspect[df_inspect['is_correct'] == False]

    # print(df_inspect[['time_before_end', 'path_type', 'path_style', 'is_correct', 'is_final']])

    if len(df_inspect_last_correct) > 0:
        bp = df_inspect_last_correct.boxplot(by =['path_style', 'with_obs'], column =['time_before_end']) #, by=['time_before_end']
        bp.set_title(pt)
        plt.savefig(export_location + pt + "before-end.png")
        plt.clf()

        bp = df_inspect_last_correct.boxplot(by =['path_style', 'with_obs'], column =['time_before_park']) #, by=['time_before_end']
        bp.set_title(pt)
        plt.savefig(export_location + pt + "before-park.png")
        plt.clf()

    df_pivot_incorrect = df_inspect_all_incorrect.pivot_table(values='is_correct', index=['path_style'], aggfunc='count', fill_value=0)
    df_pivot_incorrect.to_csv("graphics/csvs/" + "incorrect_guesses.csv")

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


print("Done with analysis")



# def draw_results():

def draw_paths_by_dict(inspection_save_path, early_dict, late_dict, even_dict, obstacle_dict):
    fig, axes = plt.subplot_mosaic("ABCD;EFGH;IJKL", figsize=(8, 6), gridspec_kw={'width_ratios':[1, 1, 1, 1], 'height_ratios':[1, 1, 1]})

    ax_mappings = {}
    ax_early    = axes['A']
    ax_even     = axes['E']
    ax_late     = axes['I']

    ax_early2   = axes['B']
    ax_even2    = axes['F']
    ax_late2    = axes['J']

    ax_early3   = axes['C']
    ax_even3    = axes['G']
    ax_late3    = axes['K']

    ax_early4   = axes['D']
    ax_even4    = axes['H']
    ax_late4    = axes['L']




    # show a legend on the plot
    # plt.legend() #loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
     
    # function to show the plot
    plt.tight_layout()
    plt.savefig('graphics/' + "overview" + '.png')
    print('graphics/' + "overview" + '.png')
    plt.clf()
    plt.close()

    
    print("Exported images of all paths")



def do_anova_and_posthoc(df, analysis_label):
    subject_id = 'participant_id'
    method_label = 'path_style'

    aov = pg.mixed_anova(dv=analysis_label, within=method_label, subject=subject_id, data=df)
    aov.round(3)

    aov.to_csv(analysis_label + '_anova.csv')

    # read off the p-value
    p_val = aov['p-unc'][0]
    if p_val < .05:
        # then something is here
        print("(+) ANOVA found significance for " + analysis_label + "!")
        f_val = str(aov['F'][0])
        dof1 = str(aov['ddof1'][0])
        dof2 = str(aov['ddof2'][0])

        if float(p_val) < .001:
            lt      = '.001'
        elif float(p_val) < .01:
            lt      = '.01'
        elif float(p_val) < .05:
            lt      = '.05'
        
        print("(" + "F(" + str(dof1) + "," + dof2 + ") = " + f_val + ", p<" +  lt + ")")
    else:
        print("(-) ANOVA found no significance for " + analysis_label + ".")
        return
    
    posthocs = pg.pairwise_tukey(dv=analysis_label, between=method_label, data=df)
    posthocs.to_csv(analysis_label + '_posthocs.csv')

    # print("Look at the values saved in " + analysis_label + '_posthocs.csv' + " to find which differences are ss")
    # print('p<.05 = *,     p<.01 = **,     p<.001 = ***')
    # print("Then you can denote it on your graphs and in results")
    
    for index, row in posthocs.iterrows():

        a   = str(row['A'])
        b   = str(row['B'])

        a_mean = str(row['mean(A)'])
        b_mean = str(row['mean(A)'])

        p   = str(row['p-tukey'])

        lt              = 'ns'
        stars           = ''
        relationship    = '~='

        if float(p) < .001:
            lt      = '.001'
            stars   = '***'
        elif float(p) < .01:
            lt      = '.01'
            stars   = '**'
        elif float(p) < .05:
            lt      = '.05'
            stars   = '*'

        if lt != 'ns':
            if a_mean > b_mean:
                relationship = ' > '
            else:
                relationship = ' < '

        output = "\t" + a + relationship + b + " \t:\t" + stars
        if lt != 'ns':
            output += "\t(p<" + lt + ")"

        print(output)


# print(df_results.columns)


# metric_col = 'time_before_end'
# do_anova_and_posthoc(df_results, metric_col)





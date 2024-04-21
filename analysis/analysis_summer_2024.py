import os
import pandas as pd
import copy
pd.set_option('display.max_colwidth', None)


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
    # print(df_chunk.columns)

    df_chunk['trial'] = df_chunk['point'].str.replace("[", "").str.replace("'", "")
    df_chunk['participant_id'] = file_name
    df_chunk[['path_name', 'path_type']] = df_chunk['trial'].str.split('-', n=1, expand=True)

    ### Label with the matching mission file
    ### If needed, also put participant number on here
    df_chunk["path_type"] = df_chunk['path_name'].replace(titles_dict)
    
    df_chunk = df_chunk.rename(columns={" status": "status", " time": "time"})
    df_chunk = df_chunk.loc[:,['time', 'trial', 'path_name', 'path_type', 'status', 'participant_id']]

    print(df_chunk)
    exit()

    trial_names.append(path_name)
    df_chunks_mini.append(copy.copy(df_chunk))

print(df_chunks_mini)


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

# For each trial in the deck
# match up the trial with button presses along the way
# Also match up with locations in space via expanded set


##### Run comparisons across symmetrical sets?








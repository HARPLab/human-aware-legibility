import os
import pandas as pd
import copy

##### Import keypress file data
##### Next: shillelagh to import directly from file?
# keypress_file = open(keypress_path)
# for line in keypress_file.readlines()[1:]:
#     data = line.replace('\n', "")
#     data = data.split(",")
#     clean_data = [int(data[0]), data[1]]

#     keypress_data.append(clean_data)
# keypress_file.close()

keypress_path           = "keypress_log.csv"
FLAG_ANALYZE_EXPANDED   = False

keypress_data = []
df_keypress = pd.read_csv(keypress_path)

##### Import the robot movement data
dir_list = os.listdir("mission_reports/")
dir_list = [f for f in dir_list if 'csv' in f]

mini_list = [f for f in dir_list if 'mini' in f]
mission_list = [f for f in dir_list if 'mission' in f]

print("IMPORTING")
print(mini_list)
print(mission_list)

df_chunks_mini      = []
df_chunks_mission   = []

trial_names = []
### SIMPLIFIED DATA
for file_name in mini_list:
    waypoints_path = "mission_reports/" + file_name
    path_name = file_name.replace('.csv','')

    keypress_file   = open(waypoints_path)    
    df_chunk        = pd.read_csv(waypoints_path) #, columns=['path_label', 'status', 'timestamp', 'time_elapsed'])

    ### Label with the matching mission file
    ### If needed, also put participant number on here
    df_chunk['trial_group'] = path_name
    
    trial_names.append(path_name)
    df_chunks_mini.append(copy.copy(df_chunk))


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








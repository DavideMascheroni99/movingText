import pandas as pd
import numpy as np
import duckdb
import ml_constants
from IPython.display import display



'''LOAD ALL CSV'''

all_dfs = {}

for tester in range(1, ml_constants.total_testers+1):
    for session in range(1, 4):
        for trial in range(1, 4):
            for fname in ml_constants.filenames:
                for anim_name in ml_constants.animname:
                    key = f"T{tester}_S{session}_TRY{trial}_{anim_name}_{fname}"
                    path = rf"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Results_csv\Tester{tester}\Session{session}\Trial{trial}\T{tester}-S{session}-TRY{trial}-{anim_name}_{fname}.csv"
                    try:
                        df = pd.read_csv(path)
                        all_dfs[key] = df
                        print(f"Loaded: {key}")
                    except FileNotFoundError:
                        print(f"Missing: {key}")




'''CREATE THE DATAFRAME WITH THE FIRST COLUMN, WHICH IS THE NUMBER OF FIXATIONS (F0)'''
feature_vectors = []
# Loop through all files in the dataframe
for key, df in all_dfs.items():
    #Start from 1 because the fixation with id zero is our first fixation
    num_fix = 1
    if 'FPOGID' in df.columns and not df.empty:
        # Convert to list for easier iteration
        fix_id_series = df['FPOGID'].tolist() 
        # Check that the list is not empty
        if fix_id_series: 
            previous_id = fix_id_series[0]
        # Start from the second element
        for current_id in fix_id_series[1:]:
            if current_id != previous_id:
                num_fix += 1
                previous_id = current_id

    # Create a feature vector with key and features
    feature_vectors.append({
        'file_key': key,
        'f0': num_fix
    })
     
# Create a DataFrame from all feature vectors
feature_df = pd.DataFrame(feature_vectors)

# Display the current dataset
display(feature_df)



'''OBTAIN THE DURATION FOR EVERY FPOID IN A FILE'''

fix_duration_vector = []  # Stores the final feature vector per file
# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    # SQL query using DuckDB to get max FPOGD per FPOGID
    result = duckdb.query("""
        SELECT FPOGID, MAX(FPOGD) as max_FPOGD
        FROM df
        GROUP BY FPOGID
        ORDER BY FPOGID
    """).to_df()

    # Optional: flatten max_FPOGD column into a single list/vector if needed
    max_fpogd_vector = result['max_FPOGD'].tolist()
    fpogid_vector = result['FPOGID'].tolist()

    # Store in the feature vector list
    fix_duration_vector.append({
        'file_key': key,
        'FPOGID': fpogid_vector,
        'max_FPOGD': max_fpogd_vector
    })

    print(f"File: {key}")
    print(result)



'''MINIIMUM FIXATION DURATION(F1)'''
#Minimun fixation duration that join with the feature vector
min_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

    #Check if empty
    if max_fpogd: 
        min_duration = np.min(max_fpogd)
    else:
        min_duration = None 

    min_fixation_data.append({
        'file_key': file_key,
        'f1': min_duration
    })

# Create pandas DataFrame
min_fixation_df = pd.DataFrame(min_fixation_data)

feature_df = feature_df.merge(min_fixation_df, on='file_key', how='left')
display(feature_df)

#Create the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)



'''MAXIMUM FIXATION DURATION (F2)'''

max_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

    #Check if empty
    if max_fpogd: 
        max_duration = max(max_fpogd)
    else:
        max_duration = None 

    max_fixation_data.append({
        'file_key': file_key,
        'f2': max_duration
    })

# Create pandas DataFrame
max_fixation_df = pd.DataFrame(max_fixation_data)

feature_df = feature_df.merge(max_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)



'''ARITHMETIC MEAN FOR FIXATION DURATION (F3)'''
mean_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

    #Check if empty
    if max_fpogd: 
        mean_duration = np.mean(max_fpogd)
    else:
        mean_duration = None 

    mean_fixation_data.append({
        'file_key': file_key,
        'f3': mean_duration
    })

# Create pandas DataFrame
mean_fixation_df = pd.DataFrame(mean_fixation_data)

feature_df = feature_df.merge(mean_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


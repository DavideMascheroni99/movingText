#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

number_of_testers = 35
filenames = ["SL_LIT", "SL_BIG", "FA_LIT", "FA_BIG"]
animname = ["VB", "HS"]
all_dfs = {}

base_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Results_csv"

for tester in range(1, number_of_testers + 1):
    # First, check if all expected files exist
    all_files_exist = True
    expected_paths = []

    for session in range(1, 4):
        for trial in range(1, 4):
            for fname in filenames:
                for anim_name in animname:
                    path = os.path.join(
                        base_path,
                        f"Tester{tester}",
                        f"Session{session}",
                        f"Trial{trial}",
                        f"T{tester}-S{session}-TRY{trial}-{anim_name}_{fname}.csv"
                    )
                    expected_paths.append((path, session, trial, anim_name, fname))

    for path, *_ in expected_paths:
        if not os.path.isfile(path):
            print(f"Tester {tester} skipped due to missing file: {path}")
            all_files_exist = False
            break

    if not all_files_exist:
        continue  # Skip this tester

    # All files exist, now load them
    for path, session, trial, anim_name, fname in expected_paths:
        key = f"T{tester}_S{session}_TRY{trial}_{anim_name}_{fname}"
        df = pd.read_csv(path)
        all_dfs[key] = df
        print(f"Loaded: {key}")


# In[2]:


#Fixation vector for each file
import duckdb

fpogs_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    
    result = duckdb.query("""
        SELECT 
            AVG(FPOGX) AS FPOGX, 
            AVG(FPOGY) AS FPOGY, 
            MAX(FPOGS) AS FPOGS, 
            MAX(FPOGD) AS FPOGD, 
            FPOGID
        FROM df
        WHERE FPOGV = '1'
        GROUP BY FPOGID
        ORDER BY FPOGID
    """).to_df()
    
    # Extract vectors from result
    fpogx_vector = result['FPOGX'].tolist()
    fpogy_vector = result['FPOGY'].tolist()
    fpogs_values = result['FPOGS'].tolist()
    fpogd_vector = result['FPOGD'].tolist()
    fpogid_vector = result['FPOGID'].tolist()
    
    # Append structured result
    fpogs_vector.append({
        'file_key': key,
        'FPOGX': fpogx_vector,
        'FPOGY': fpogy_vector,
        'FPOGS': fpogs_values,
        'FPOGD': fpogd_vector,
        'FPOGID': fpogid_vector
    })

# Convert to DataFrame
fpogs_vector_df = pd.DataFrame(fpogs_vector)

# Display the DataFrame
display(fpogs_vector_df)


# In[3]:


#Show the fixation data in a nicer way into a csv file
import pandas as pd

# Flatten the list into a structured tabular format
flattened_fpogs_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    for fpogid, fpogx, fpogy, fpogs, fpogd in zip(
        item['FPOGID'], item['FPOGX'], item['FPOGY'], item['FPOGS'], item['FPOGD']
    ):
        flattened_fpogs_data.append({
            'file_key': file_key,
            'FPOGID': fpogid,
            'FPOGX': fpogx,
            'FPOGY': fpogy,
            'FPOGS': fpogs,
            'FPOGD': fpogd
        })

# Create the final DataFrame
fpogs_table_df = pd.DataFrame(flattened_fpogs_data)

# Display the table
display(fpogs_table_df)

#Create the csv 
fpogs_table_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\fixation_vector.csv', index=False)


# In[7]:


#Count the number of fixations

feature_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    
    # Use DuckDB to count distinct FPOGID values
    result = duckdb.query("""
        SELECT COUNT(DISTINCT FPOGID) AS number_of_fixations
        FROM df
    """).to_df()
    
    # Append the result to the list
    feature_vector.append({
        'file_key': key,
        'f0': result['number_of_fixations'][0]
    })

# Convert the list of dicts to a DataFrame
feature_df = pd.DataFrame(feature_vector)

# Display the result
display(feature_df)


# In[8]:


#Obtain the duration of every FPOGID in a file
import duckdb

fix_duration_vector = []

for fix in fpogs_vector:
    
    fpogd_vector = fix['FPOGD']
    fpogid_vector = fix['FPOGID']

    # Store in the feature vector list
    fix_duration_vector.append({
        'file_key': key,
        'FPOGID': fpogid_vector,
        'FPOGD': fpogd_vector
    })

# Create a final DataFrame
fix_duration_vector_df = pd.DataFrame(fix_duration_vector)

# Display the DataFrame
display(fix_duration_vector_df)


# In[9]:


#Minimum fixation duration

min_fix_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    fpogd_list = item['FPOGD']
    
    if fpogd_list:
        min_fpogd = np.min(fpogd_list)
    else:
        min_fpogd = None  
    
    min_fix_data.append({ 
        'file_key': file_key,
        'f1': min_fpogd
    })

# Convert to DataFrame for easy use or merging
min_fixation_df = pd.DataFrame(min_fix_data)

feature_df = feature_df.merge(min_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[10]:


#Maximum fixation duration that join with the feature vector
max_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        max_duration = np.max(max_fpogd)
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


# In[11]:


#Arithmetic mean of fixation duration that join with feature vector
mean_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

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


# In[12]:


#Geometric mean of fixation duration that join with feature vector
from scipy.stats import gmean

geom_mean_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        geom_mean_duration = gmean(max_fpogd)
    else:
        geom_mean_duration = None 

    geom_mean_fixation_data.append({
        'file_key': file_key,
        'f4': geom_mean_duration
    })

# Create pandas DataFrame
geom_mean_fixation_df = pd.DataFrame(geom_mean_fixation_data)

feature_df = feature_df.merge(geom_mean_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[13]:


#Median of fixation duration that join with feature vector
median_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        median_duration = np.median(max_fpogd)
    else:
        median_duration = None 

    median_fixation_data.append({
        'file_key': file_key,
        'f5': median_duration
    })

# Create pandas DataFrame
median_fixation_df = pd.DataFrame(median_fixation_data)

feature_df = feature_df.merge(median_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[14]:


#Standard deviation of fixation duration that join with feature vector
std_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        std_duration = np.std(max_fpogd)
    else:
        std_duration = None 

    std_fixation_data.append({
        'file_key': file_key,
        'f6': std_duration
    })

# Create pandas DataFrame
std_fixation_df = pd.DataFrame(std_fixation_data)

feature_df = feature_df.merge(std_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[15]:


#Median absolute deviation of fixation duration that join with feature vector
from scipy.stats import median_abs_deviation

mad_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        mad_duration = median_abs_deviation(max_fpogd, scale=1)
    else:
        mad_duration = None 

    mad_fixation_data.append({
        'file_key': file_key,
        'f7': mad_duration
    })

# Create pandas DataFrame
mad_fixation_df = pd.DataFrame(mad_fixation_data)

feature_df = feature_df.merge(mad_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[16]:


#Skewness of fixation duration that join with feature vector
from scipy.stats import skew

skew_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        skew_duration = skew(max_fpogd)
    else:
        skew_duration = None 

    skew_fixation_data.append({
        'file_key': file_key,
        'f8': skew_duration
    })

# Create pandas DataFrame
skew_fixation_df = pd.DataFrame(skew_fixation_data)

feature_df = feature_df.merge(skew_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[17]:


#Interquartile range of fixation duration that join with feature vector
from scipy.stats import iqr

iqr_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        iqr_duration = iqr(max_fpogd)
    else:
        iqr_duration = None 

    iqr_fixation_data.append({
        'file_key': file_key,
        'f9': iqr_duration
    })

# Create pandas DataFrame
iqr_fixation_df = pd.DataFrame(iqr_fixation_data)

feature_df = feature_df.merge(iqr_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[18]:


#Kurtosis of fixation duration that join with feature vector
from scipy.stats import kurtosis

kurt_fixation_data = []

for item in fpogs_vector:
    file_key = item['file_key']
    max_fpogd = item['FPOGD']

    #Check if empty
    if max_fpogd: 
        kurt_duration = kurtosis(max_fpogd)
    else:
        kurt_duration = None 

    kurt_fixation_data.append({
        'file_key': file_key,
        'f10': kurt_duration
    })

# Create pandas DataFrame
kurt_fixation_df = pd.DataFrame(kurt_fixation_data)

feature_df = feature_df.merge(kurt_fixation_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[19]:


#Distance between consecutive fixations
#The distance is computed taking in consideration the euclidean distance from the average FPOGX and FPOGY of two consecutive fixations
from scipy.spatial import distance

fpogs_diff_vector = []

for fix in fpogs_vector:
    file_key = fix['file_key']
    fpogid_vector = fix['FPOGID']
    fpogx_vector = fix['FPOGX']
    fpogy_vector = fix['FPOGY']

    # Build list of consecutive FPOGID pairs 
    fpogid_pairs = []
    for i in range(len(fpogid_vector) - 1):
        pair = (fpogid_vector[i], fpogid_vector[i + 1])
        fpogid_pairs.append(pair)

    # Build list of consecutive Euclidean distances
    fpogs_differences = []
    for i in range(len(fpogx_vector) - 1):
            point1 = [fpogx_vector[i], fpogy_vector[i]]
            point2 = [fpogx_vector[i + 1], fpogy_vector[i + 1]]
            dist = distance.euclidean(point1, point2)
            fpogs_differences.append(dist)
       
    # Append result 
    fpogs_diff_vector.append({
        'file_key': file_key,
        'FPOGID_pairs': fpogid_pairs,
        'FPOGS_differences': fpogs_differences
    })

# Create the DataFrame
fpogs_diff_vector_df = pd.DataFrame(fpogs_diff_vector)

# Display the DataFrame
display(fpogs_diff_vector_df)


# In[20]:


#Minimum difference between consecutive fixations that join with feature vector
min_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        min_diff = np.min(differences)
    else:
        min_diff = None

    min_diff_data.append({
        'file_key': file_key,
        'f11': min_diff
    })

# Create DataFrame from min differences
min_diff_df = pd.DataFrame(min_diff_data)

feature_df = feature_df.merge(min_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[21]:


#Maximum difference between consecutive fixations that join with feature vector
max_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        max_diff = np.max(differences)
    else:
        max_diff = None

    max_diff_data.append({
        'file_key': file_key,
        'f12': max_diff
    })

# Create the DataFrame 
max_diff_df = pd.DataFrame(max_diff_data)

feature_df = feature_df.merge(max_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[22]:


#Arithmetic mean between consecutive fixations that join with feature vector
mean_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        mean_diff = np.mean(differences)
    else:
        mean_diff = None

    mean_diff_data.append({
        'file_key': file_key,
        'f13': mean_diff
    })

# Create the DataFrame 
mean_diff_df = pd.DataFrame(mean_diff_data)

feature_df = feature_df.merge(mean_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[23]:


#Geometric mean between consecutive fixations that join with feature vector
gmean_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        gmean_diff = gmean(differences)
    else:
        gmean_diff = None

    gmean_diff_data.append({
        'file_key': file_key,
        'f14': gmean_diff
    })

# Create the DataFrame 
gmean_diff_df = pd.DataFrame(gmean_diff_data)

feature_df = feature_df.merge(gmean_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[24]:


#Median between consecutive fixations that join with feature vector
median_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        median_diff = np.median(differences)
    else:
        median_diff = None

    median_diff_data.append({
        'file_key': file_key,
        'f15': median_diff
    })

# Create the DataFrame 
median_diff_df = pd.DataFrame(median_diff_data)

feature_df = feature_df.merge(median_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[25]:


#STD between consecutive fixations that join with feature vector
std_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        std_diff = np.std(differences)
    else:
        std_diff = None

    std_diff_data.append({
        'file_key': file_key,
        'f16': std_diff
    })

# Create the DataFrame 
std_diff_df = pd.DataFrame(std_diff_data)

feature_df = feature_df.merge(std_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[26]:


#MAD between consecutive fixations that join with feature vector
mad_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        mad_diff = median_abs_deviation(differences, scale=1)
    else:
        mad_diff = None

    mad_diff_data.append({
        'file_key': file_key,
        'f17': mad_diff
    })

# Create the DataFrame 
mad_diff_df = pd.DataFrame(mad_diff_data)

feature_df = feature_df.merge(mad_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[27]:


#Skewness between consecutive fixations that join with feature vector
skew_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        skew_diff = skew(differences)
    else:
        skew_diff = None

    skew_diff_data.append({
        'file_key': file_key,
        'f18': skew_diff
    })

# Create the DataFrame 
skew_diff_df = pd.DataFrame(skew_diff_data)

feature_df = feature_df.merge(skew_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[28]:


#Interquartile range between consecutive fixations that join with feature vector
iqr_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        iqr_diff = iqr(differences)
    else:
        iqr_diff = None

    iqr_diff_data.append({
        'file_key': file_key,
        'f19': iqr_diff
    })

# Create the DataFrame 
iqr_diff_df = pd.DataFrame(iqr_diff_data)

feature_df = feature_df.merge(iqr_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[29]:


#Kurtosis between consecutive fixations that join with feature vector
kurt_diff_data = []

for item in fpogs_diff_vector:
    file_key = item['file_key']
    differences = item['FPOGS_differences']

    #Check if the list is empty
    if differences:  
        kurt_diff = kurtosis(differences)
    else:
        kurt_diff = None

    kurt_diff_data.append({
        'file_key': file_key,
        'f20': kurt_diff
    })

# Create the DataFrame 
kurt_diff_df = pd.DataFrame(kurt_diff_data)

feature_df = feature_df.merge(kurt_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[30]:


#Compute the speed of every saccade in a file

# Load fixation csv
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\fixation_vector.csv"
df = pd.read_csv(csv_path)

saccade_speed_vector = []

# Group by 'file_key'
for file_key, group in df.groupby('file_key'):
    # Sort by fixation ID
    group = group.sort_values('FPOGID')
    
    # Extract relevant columns
    fpogx_vector = group['FPOGX'].tolist()
    fpogy_vector = group['FPOGY'].tolist()
    fpogd_vector = group['FPOGD'].tolist()
    fpogs_vector = group['FPOGS'].tolist()
    fpogid_vector = group['FPOGID'].tolist()

    # Initialize list for this group
    saccade_speeds = []
    id_pairs = []

    for i in range(len(fpogx_vector) - 1):
        # Coordinates of two consecutive fixations
        point1 = [fpogx_vector[i], fpogy_vector[i]]
        point2 = [fpogx_vector[i + 1], fpogy_vector[i + 1]]

        # Euclidean distance between fixations
        dist = distance.euclidean(point1, point2)

        # Saccade time = time between fixations - duration of fixation
        diff = fpogs_vector[i + 1] - fpogs_vector[i]
        time = diff - fpogd_vector[i]

        # Compute speed safely
        if time > 0:
            speed = dist / time
        else:
            speed = np.nan

        saccade_speeds.append(speed)
        id_pairs.append((fpogid_vector[i], fpogid_vector[i + 1]))

    saccade_speed_vector.append({
        'file_key': file_key,
        'saccade_id_pairs': id_pairs,
        'FPOGS_speeds': saccade_speeds
    })

# Convert to DataFrame
saccade_speed_vector_df = pd.DataFrame(saccade_speed_vector)
display(saccade_speed_vector_df)


# In[31]:


#Minimum saccade speed
min_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        min_speed = np.min(speeds)
    else:
        min_speed = None

    min_speed_data.append({
        'file_key': file_key,
        'f21': min_speed
    })

# Create DataFrame from min speeds
min_speed_df = pd.DataFrame(min_speed_data)

feature_df = feature_df.merge(min_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[32]:


#Maximim saccade speed
max_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        max_speed = np.max(speeds)
    else:
        max_speed = None

    max_speed_data.append({
        'file_key': file_key,
        'f22': max_speed
    })

# Create DataFrame from max speeds
max_speed_df = pd.DataFrame(max_speed_data)

feature_df = feature_df.merge(max_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[33]:


#Arithmetic mean saccade speed
mean_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        mean_speed = np.mean(speeds)
    else:
        mean_speed = None

    mean_speed_data.append({
        'file_key': file_key,
        'f23': mean_speed
    })

# Create DataFrame from mean speeds
mean_speed_df = pd.DataFrame(mean_speed_data)

feature_df = feature_df.merge(mean_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[34]:


#Geometric mean saccade speed
gmean_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        gmean_speed = gmean(speeds)
    else:
        gmean_speed = None

    gmean_speed_data.append({
        'file_key': file_key,
        'f24': gmean_speed
    })

# Create DataFrame from gmean speeds
gmean_speed_df = pd.DataFrame(gmean_speed_data)

feature_df = feature_df.merge(gmean_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[35]:


#Median saccade speed
median_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        median_speed = np.median(speeds)
    else:
        median_speed = None

    median_speed_data.append({
        'file_key': file_key,
        'f25': median_speed
    })

# Create DataFrame from median speeds
median_speed_df = pd.DataFrame(median_speed_data)

feature_df = feature_df.merge(median_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[36]:


#STD saccade speed
std_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        std_speed = np.std(speeds)
    else:
        std_speed = None

    std_speed_data.append({
        'file_key': file_key,
        'f26': std_speed
    })

# Create DataFrame from std speeds
std_speed_df = pd.DataFrame(std_speed_data)

feature_df = feature_df.merge(std_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[37]:


#MAD saccade speed
mad_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        mad_speed = median_abs_deviation(speeds, scale=1)
    else:
        mad_speed = None

    mad_speed_data.append({
        'file_key': file_key,
        'f27': mad_speed
    })

# Create DataFrame from mad speeds
mad_speed_df = pd.DataFrame(mad_speed_data)

feature_df = feature_df.merge(mad_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[38]:


#Skewness saccade speed
skew_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        skew_speed = skew(speeds)
    else:
        skew_speed = None

    skew_speed_data.append({
        'file_key': file_key,
        'f28': skew_speed
    })

# Create DataFrame from skew speeds
skew_speed_df = pd.DataFrame(skew_speed_data)

feature_df = feature_df.merge(skew_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[39]:


#IQR saccade speed
iqr_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        iqr_speed = iqr(speeds)
    else:
        iqr_speed = None

    iqr_speed_data.append({
        'file_key': file_key,
        'f29': iqr_speed
    })

# Create DataFrame from iqr speeds
iqr_speed_df = pd.DataFrame(iqr_speed_data)

feature_df = feature_df.merge(iqr_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[40]:


#Kurtosis saccade speed
kurt_speed_data = []

for item in saccade_speed_vector:
    file_key = item['file_key']
    speeds = item['FPOGS_speeds']

    #Check if the list is empty
    if speeds:  
        kurt_speed = kurtosis(speeds)
    else:
        kurt_speed = None

    kurt_speed_data.append({
        'file_key': file_key,
        'f30': kurt_speed
    })

# Create DataFrame from kurt speeds
kurt_speed_df = pd.DataFrame(kurt_speed_data)

feature_df = feature_df.merge(kurt_speed_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[41]:


#Scanpath lenght for every file
#The scanpath lenght is computed summing the euclidean distance of every consecutive point in the file

# Load fixation csv
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\fixation_vector.csv"
df = pd.read_csv(csv_path)

fpogs_scan_vector = []

# Group by 'file_key'
for file_key, group in df.groupby('file_key'):
    # Sort by fixation ID
    group = group.sort_values('FPOGID')
    
    fpogx_vector = group['FPOGX'].tolist()
    fpogy_vector = group['FPOGY'].tolist()

    tot = 0
    for i in range(len(fpogx_vector) - 1):
        point1 = [fpogx_vector[i], fpogy_vector[i]]
        point2 = [fpogx_vector[i + 1], fpogy_vector[i + 1]]
        dist = distance.euclidean(point1, point2)
        tot += dist

    fpogs_scan_vector.append({
        'file_key': file_key,
        'f31': tot
    })

fpogs_scan_vector_df = pd.DataFrame(fpogs_scan_vector)
feature_df = feature_df.merge(fpogs_scan_vector_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[42]:


#List of all valid(LPV = 1) left pupil diameter(LPD) for each file
lpd_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    # Query using DuckDB
    result = duckdb.query("""
        SELECT LPD 
        FROM df
        WHERE LPV = '1'
    """).to_df()

    # Extract vectors
    lpd_list = result['LPD'].tolist()

    # Append result 
    lpd_vector.append({
        'file_key': key,
        'LPD': lpd_list
    })

# Create the DataFrame
lpd_vector_df = pd.DataFrame(lpd_vector)

# Display the DataFrame
display(lpd_vector_df)


# In[43]:


#Minimum left pupil diameter
min_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        min_lpd = np.min(lpd)
    else:
        min_lpd = None

    min_lpd_data.append({
        'file_key': file_key,
        'f32': min_lpd
    })

# Create DataFrame from min differences
min_lpd_df = pd.DataFrame(min_lpd_data)

feature_df = feature_df.merge(min_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[44]:


#Maximum left pupil diameter
max_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        max_lpd = np.max(lpd)
    else:
        max_lpd = None

    max_lpd_data.append({
        'file_key': file_key,
        'f33': max_lpd
    })

# Create DataFrame 
max_lpd_df = pd.DataFrame(max_lpd_data)

feature_df = feature_df.merge(max_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[45]:


#Arithmetic mean of left pupil diameter
mean_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        mean_lpd = np.mean(lpd)
    else:
        mean_lpd = None

    mean_lpd_data.append({
        'file_key': file_key,
        'f34': mean_lpd
    })

# Create DataFrame 
mean_lpd_df = pd.DataFrame(mean_lpd_data)

feature_df = feature_df.merge(mean_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[46]:


#Geometric mean of left pupil diameter
gmean_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        gmean_lpd = gmean(lpd)
    else:
        gmean_lpd = None

    gmean_lpd_data.append({
        'file_key': file_key,
        'f35': gmean_lpd
    })

# Create DataFrame 
gmean_lpd_df = pd.DataFrame(gmean_lpd_data)

feature_df = feature_df.merge(gmean_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[47]:


#Median of left pupil diameter
median_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        median_lpd = np.median(lpd)
    else:
        median_lpd = None

    median_lpd_data.append({
        'file_key': file_key,
        'f36': median_lpd
    })

# Create DataFrame 
median_lpd_df = pd.DataFrame(median_lpd_data)

feature_df = feature_df.merge(median_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[48]:


#STD of left pupil diameter
std_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        std_lpd = np.std(lpd)
    else:
        median_lpd = None

    std_lpd_data.append({
        'file_key': file_key,
        'f37': std_lpd
    })

# Create DataFrame 
std_lpd_df = pd.DataFrame(std_lpd_data)

feature_df = feature_df.merge(std_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[49]:


#MAD of left pupil diameter
mad_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        mad_lpd = median_abs_deviation(lpd, scale=1)
    else:
        median_lpd = None

    mad_lpd_data.append({
        'file_key': file_key,
        'f38': mad_lpd
    })

# Create DataFrame 
mad_lpd_df = pd.DataFrame(mad_lpd_data)

feature_df = feature_df.merge(mad_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[50]:


#SKEWNESS of left pupil diameter
skew_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        skew_lpd = skew(lpd)
    else:
        median_lpd = None

    skew_lpd_data.append({
        'file_key': file_key,
        'f39': skew_lpd
    })

# Create DataFrame 
skew_lpd_df = pd.DataFrame(skew_lpd_data)

feature_df = feature_df.merge(skew_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[51]:


#IQR of left pupil diameter
iqr_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        iqr_lpd = iqr(lpd)
    else:
        median_lpd = None

    iqr_lpd_data.append({
        'file_key': file_key,
        'f40': iqr_lpd
    })

# Create DataFrame 
iqr_lpd_df = pd.DataFrame(iqr_lpd_data)

feature_df = feature_df.merge(iqr_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[52]:


#Kurtosis of left pupil diameter
kurt_lpd_data = []

for item in lpd_vector:
    file_key = item['file_key']
    lpd = item['LPD']

    #Check if the list is empty
    if lpd:  
        kurt_lpd = kurtosis(lpd)
    else:
        median_lpd = None

    kurt_lpd_data.append({
        'file_key': file_key,
        'f41': kurt_lpd
    })

# Create DataFrame 
kurt_lpd_df = pd.DataFrame(kurt_lpd_data)

feature_df = feature_df.merge(kurt_lpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[53]:


#List of all valid(RPV = 1) right pupil diameter(RPD) for each file
rpd_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    # Query using DuckDB
    result = duckdb.query("""
        SELECT RPD 
        FROM df
        WHERE RPV = '1'
    """).to_df()

    # Extract vectors
    rpd_list = result['RPD'].tolist()

    # Append result 
    rpd_vector.append({
        'file_key': key,
        'RPD': rpd_list
    })

# Create the DataFrame
rpd_vector_df = pd.DataFrame(rpd_vector)

# Display the DataFrame
display(rpd_vector_df)


# In[54]:


#Minimum right pupil diameter
min_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        min_rpd = np.min(rpd)
    else:
        min_rpd = None

    min_rpd_data.append({
        'file_key': file_key,
        'f42': min_rpd
    })

# Create DataFrame from min differences
min_rpd_df = pd.DataFrame(min_rpd_data)

feature_df = feature_df.merge(min_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[55]:


#Maximum right pupil diameter
max_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        max_rpd = np.max(rpd)
    else:
        max_rpd = None

    max_rpd_data.append({
        'file_key': file_key,
        'f43': max_rpd
    })

# Create DataFrame 
max_rpd_df = pd.DataFrame(max_rpd_data)

feature_df = feature_df.merge(max_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[56]:


#Arithmetic mean of right pupil diameter
mean_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        mean_rpd = np.mean(rpd)
    else:
        mean_rpd = None

    mean_rpd_data.append({
        'file_key': file_key,
        'f44': mean_rpd
    })

# Create DataFrame 
mean_rpd_df = pd.DataFrame(mean_rpd_data)

feature_df = feature_df.merge(mean_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[57]:


#Geometric mean of right pupil diameter
gmean_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        gmean_rpd = gmean(rpd)
    else:
        gmean_rpd = None

    gmean_rpd_data.append({
        'file_key': file_key,
        'f45': gmean_rpd
    })

# Create DataFrame 
gmean_rpd_df = pd.DataFrame(gmean_rpd_data)

feature_df = feature_df.merge(gmean_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[58]:


#Median of right pupil diameter
median_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        median_rpd = np.median(rpd)
    else:
        median_rpd = None

    median_rpd_data.append({
        'file_key': file_key,
        'f46': median_rpd
    })

# Create DataFrame 
median_rpd_df = pd.DataFrame(median_rpd_data)

feature_df = feature_df.merge(median_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[59]:


#STD of right pupil diameter
std_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        std_rpd = np.std(rpd)
    else:
        median_rpd = None

    std_rpd_data.append({
        'file_key': file_key,
        'f47': std_rpd
    })

# Create DataFrame 
std_rpd_df = pd.DataFrame(std_rpd_data)

feature_df = feature_df.merge(std_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[60]:


#MAD of right pupil diameter
mad_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        mad_rpd = median_abs_deviation(rpd, scale=1)
    else:
        median_rpd = None

    mad_rpd_data.append({
        'file_key': file_key,
        'f48': mad_rpd
    })

# Create DataFrame 
mad_rpd_df = pd.DataFrame(mad_rpd_data)

feature_df = feature_df.merge(mad_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[61]:


#SKEWNESS of right pupil diameter
skew_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        skew_rpd = skew(rpd)
    else:
        median_rpd = None

    skew_rpd_data.append({
        'file_key': file_key,
        'f49': skew_rpd
    })

# Create DataFrame 
skew_rpd_df = pd.DataFrame(skew_rpd_data)

feature_df = feature_df.merge(skew_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[62]:


#IQR of right pupil diameter
iqr_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        iqr_rpd = iqr(rpd)
    else:
        median_rpd = None

    iqr_rpd_data.append({
        'file_key': file_key,
        'f50': iqr_rpd
    })

# Create DataFrame 
iqr_rpd_df = pd.DataFrame(iqr_rpd_data)

feature_df = feature_df.merge(iqr_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[63]:


#Kurtosis of right pupil diameter
kurt_rpd_data = []

for item in rpd_vector:
    file_key = item['file_key']
    rpd = item['RPD']

    #Check if the list is empty
    if rpd:  
        kurt_rpd = kurtosis(rpd)
    else:
        median_rpd = None

    kurt_rpd_data.append({
        'file_key': file_key,
        'f51': kurt_rpd
    })

# Create DataFrame 
kurt_rpd_df = pd.DataFrame(kurt_rpd_data)

feature_df = feature_df.merge(kurt_rpd_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[64]:


#List of all valid(LPV = 1 and RPV = 1) ratio between left pupil and right pupil diameter for each file
ratio_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    # Query using DuckDB
    result = duckdb.query("""
        SELECT LPD/RPD as ratio 
        FROM df
        WHERE LPV = '1' and RPV = '1'
    """).to_df()

    # Extract vectors
    ratio_list = result['ratio'].tolist()

    # Append result 
    ratio_vector.append({
        'file_key': key,
        'ratio': ratio_list
    })

# Create the DataFrame
ratio_vector_df = pd.DataFrame(ratio_vector)

# Display the DataFrame
display(ratio_vector_df)


# In[65]:


#Minimum ratio bewtween left and right pupil diameter
min_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        min_ratio = np.min(ratio)
    else:
        min_ratio = None

    min_ratio_data.append({
        'file_key': file_key,
        'f52': min_ratio
    })

# Create DataFrame from min ratio
min_ratio_df = pd.DataFrame(min_ratio_data)

feature_df = feature_df.merge(min_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[66]:


#Maximum ratio bewtween left and right pupil diameter
max_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        max_ratio = np.max(ratio)
    else:
        max_ratio = None

    max_ratio_data.append({
        'file_key': file_key,
        'f53': max_ratio
    })

# Create DataFrame from max ratio
max_ratio_df = pd.DataFrame(max_ratio_data)

feature_df = feature_df.merge(max_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[67]:


#Arithmetic mean ratio bewtween left and right pupil diameter
mean_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        mean_ratio = np.mean(ratio)
    else:
        mean_ratio = None

    mean_ratio_data.append({
        'file_key': file_key,
        'f54': mean_ratio
    })

# Create DataFrame from mean ratio
mean_ratio_df = pd.DataFrame(mean_ratio_data)

feature_df = feature_df.merge(mean_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[68]:


#Geometric mean ratio bewtween left and right pupil diameter
gmean_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        gmean_ratio = gmean(ratio)
    else:
        gmean_ratio = None

    gmean_ratio_data.append({
        'file_key': file_key,
        'f55': gmean_ratio
    })

# Create DataFrame from gmean ratio
gmean_ratio_df = pd.DataFrame(gmean_ratio_data)

feature_df = feature_df.merge(gmean_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[69]:


#median ratio bewtween left and right pupil diameter
median_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        median_ratio = np.median(ratio)
    else:
        median_ratio = None

    median_ratio_data.append({
        'file_key': file_key,
        'f56': median_ratio
    })

# Create DataFrame from median ratio
median_ratio_df = pd.DataFrame(median_ratio_data)

feature_df = feature_df.merge(median_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[70]:


#std ratio bewtween left and right pupil diameter
std_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        std_ratio = np.std(ratio)
    else:
        std_ratio = None

    std_ratio_data.append({
        'file_key': file_key,
        'f57': std_ratio
    })

# Create DataFrame from std ratio
std_ratio_df = pd.DataFrame(std_ratio_data)

feature_df = feature_df.merge(std_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[71]:


#MAD ratio bewtween left and right pupil diameter
mad_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        mad_ratio = median_abs_deviation(ratio, scale=1)
    else:
        mad_ratio = None

    mad_ratio_data.append({
        'file_key': file_key,
        'f58': mad_ratio
    })

# Create DataFrame from mad ratio
mad_ratio_df = pd.DataFrame(mad_ratio_data)

feature_df = feature_df.merge(mad_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[72]:


#Skewness ratio bewtween left and right pupil diameter
skew_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        skew_ratio = skew(ratio)
    else:
        skew_ratio = None

    skew_ratio_data.append({
        'file_key': file_key,
        'f59': skew_ratio
    })

# Create DataFrame from skew ratio
skew_ratio_df = pd.DataFrame(skew_ratio_data)

feature_df = feature_df.merge(skew_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[73]:


#IQR ratio bewtween left and right pupil diameter
iqr_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        iqr_ratio = iqr(ratio)
    else:
        iqr_ratio = None

    iqr_ratio_data.append({
        'file_key': file_key,
        'f60': iqr_ratio
    })

# Create DataFrame from iqr ratio
iqr_ratio_df = pd.DataFrame(iqr_ratio_data)

feature_df = feature_df.merge(iqr_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[74]:


#kurtosis ratio bewtween left and right pupil diameter
kurt_ratio_data = []

for item in ratio_vector:
    file_key = item['file_key']
    ratio = item['ratio']

    #Check if the list is empty
    if ratio:  
        kurt_ratio = kurtosis(ratio)
    else:
        kurt_ratio = None

    kurt_ratio_data.append({
        'file_key': file_key,
        'f61': kurt_ratio
    })

# Create DataFrame from kurt ratio
kurt_ratio_df = pd.DataFrame(kurt_ratio_data)

feature_df = feature_df.merge(kurt_ratio_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[75]:


#List of all valid(LPV = 1 and RPV = 1) difference between left pupil and right pupil diameter for each file
diff_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    # Query using DuckDB
    result = duckdb.query("""
        SELECT (LPD - RPD) as diff
        FROM df
        WHERE LPV = '1' and RPV = '1'
    """).to_df()

    # Extract vectors
    diff_list = result['diff'].tolist()

    # Append result 
    diff_vector.append({
        'file_key': key,
        'diff': diff_list
    })

# Create the DataFrame
diff_vector_df = pd.DataFrame(diff_vector)

# Display the DataFrame
display(diff_vector_df)


# In[76]:


#Minimum difference bewtween left and right pupil diameter
min_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        min_diff = np.min(diff)
    else:
        min_diff = None

    min_diff_data.append({
        'file_key': file_key,
        'f62': min_diff
    })

# Create DataFrame from min diff
min_diff_df = pd.DataFrame(min_diff_data)

feature_df = feature_df.merge(min_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[77]:


#Maximum difference bewtween left and right pupil diameter
max_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        max_diff = np.max(diff)
    else:
        max_diff = None

    max_diff_data.append({
        'file_key': file_key,
        'f63': max_diff
    })

# Create DataFrame from max diff
max_diff_df = pd.DataFrame(max_diff_data)

feature_df = feature_df.merge(max_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[78]:


#Arithmetic mean difference bewtween left and right pupil diameter
mean_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        mean_diff = np.mean(diff)
    else:
        mean_diff = None

    mean_diff_data.append({
        'file_key': file_key,
        'f64': mean_diff
    })

# Create DataFrame from mean diff
mean_diff_df = pd.DataFrame(mean_diff_data)

feature_df = feature_df.merge(mean_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[79]:


#median difference bewtween left and right pupil diameter
median_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        median_diff = np.median(diff)
    else:
        median_diff = None

    median_diff_data.append({
        'file_key': file_key,
        'f65': median_diff
    })

# Create DataFrame from median diff
median_diff_df = pd.DataFrame(median_diff_data)

feature_df = feature_df.merge(median_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[80]:


#std difference bewtween left and right pupil diameter
std_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        std_diff = np.std(diff)
    else:
        std_diff = None

    std_diff_data.append({
        'file_key': file_key,
        'f66': std_diff
    })

# Create DataFrame from std diff
std_diff_df = pd.DataFrame(std_diff_data)

feature_df = feature_df.merge(std_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[81]:


#MAD difference bewtween left and right pupil diameter
mad_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        mad_diff = median_abs_deviation(diff, scale=1)
    else:
        mad_diff = None

    mad_diff_data.append({
        'file_key': file_key,
        'f67': mad_diff
    })

# Create DataFrame from mad diff
mad_diff_df = pd.DataFrame(mad_diff_data)

feature_df = feature_df.merge(mad_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[82]:


#Skewness difference bewtween left and right pupil diameter
skew_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        skew_diff = skew(diff)
    else:
        skew_diff = None

    skew_diff_data.append({
        'file_key': file_key,
        'f68': skew_diff
    })

# Create DataFrame from skew diff
skew_diff_df = pd.DataFrame(skew_diff_data)

feature_df = feature_df.merge(skew_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[83]:


#IQR difference bewtween left and right pupil diameter
iqr_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        iqr_diff = iqr(diff)
    else:
        iqr_diff = None

    iqr_diff_data.append({
        'file_key': file_key,
        'f69': iqr_diff
    })

# Create DataFrame from iqr diff
iqr_diff_df = pd.DataFrame(iqr_diff_data)

feature_df = feature_df.merge(iqr_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[84]:


#kurtosis difference bewtween left and right pupil diameter
kurt_diff_data = []

for item in diff_vector:
    file_key = item['file_key']
    diff = item['diff']

    #Check if the list is empty
    if diff:  
        kurt_diff = kurtosis(diff)
    else:
        kurt_diff = None

    kurt_diff_data.append({
        'file_key': file_key,
        'f70': kurt_diff
    })

# Create DataFrame from kurt diff
kurt_diff_df = pd.DataFrame(kurt_diff_data)

feature_df = feature_df.merge(kurt_diff_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[85]:


#Number of blinks
#Obtained counting the number of BKID different from zero in each file

bkid_counts = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    result = duckdb.query("""
        SELECT COUNT(DISTINCT BKID) AS BKID_COUNT
        FROM df
        WHERE CAST(BKID AS INTEGER) > 0
    """).to_df()

    # Append result with file key and count
    bkid_counts.append({
        'file_key': key,
        'f71': result['BKID_COUNT'][0]
    })

# Convert list of dicts to a DataFrame
bkid_counts_df = pd.DataFrame(bkid_counts)

feature_df = feature_df.merge(bkid_counts_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[86]:


#Blink duration list
#Obtained selecting all the BKDUR different from zero

bkdur_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    result = duckdb.query("""
        SELECT BKDUR
        FROM df
        WHERE CAST(BKDUR AS DOUBLE) > 0.0
    """).to_df()

    bkdur_list = result['BKDUR'].tolist()


    # Append result with file key and count
    bkdur_vector.append({
        'file_key': key,
        'BKDUR': bkdur_list
    })

# Convert list of dicts to a DataFrame
bkdur_vector_df = pd.DataFrame(bkdur_vector)

display(bkdur_vector_df)


# In[87]:


#Arithmetic mean over all the BKDUR different from zero

mean_bkdur = []

for item in bkdur_vector:
    file_key = item['file_key']
    bkdur = item['BKDUR']

    #Check if the list is empty
    if bkdur:  
        mean = np.mean(bkdur)
    else:
        mean = 0

    mean_bkdur.append({
        'file_key': file_key,
        'f72': mean
    })

# Create DataFrame from mean diff
mean_bkdur_df = pd.DataFrame(mean_bkdur)

feature_df = feature_df.merge(mean_bkdur_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[88]:


#Minimum over all the BKDUR different from zero

min_bkdur = []

for item in bkdur_vector:
    file_key = item['file_key']
    bkdur = item['BKDUR']

    #Check if the list is empty
    if bkdur:  
        min = np.min(bkdur)
    else:
        min = 0

    min_bkdur.append({
        'file_key': file_key,
        'f73': min
    })

# Create DataFrame from min diff
min_bkdur_df = pd.DataFrame(min_bkdur)

feature_df = feature_df.merge(min_bkdur_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[89]:


#Maximum over all the BKDUR different from zero

max_bkdur = []

for item in bkdur_vector:
    file_key = item['file_key']
    bkdur = item['BKDUR']

    #Check if the list is empty
    if bkdur:  
        max = np.max(bkdur)
    else:
        max = 0

    max_bkdur.append({
        'file_key': file_key,
        'f74': max
    })

# Create DataFrame from max diff
max_bkdur_df = pd.DataFrame(max_bkdur)

feature_df = feature_df.merge(max_bkdur_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


# In[90]:


#Count the number of ascending, descending, right and left movement during a fixation and the total distance of these.
#The distance of a movement in one direction is computed with the absolute value difference 

# Load fixation CSV
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\fixation_vector.csv"
df = pd.read_csv(csv_path)

movement_vector = []

# Group by 'file_key'
for file_key, group in df.groupby('file_key'):
    # Sort by fixation ID
    group = group.sort_values('FPOGID')

    # Extract relevant columns
    fpogx_vector = group['FPOGX'].tolist()
    fpogy_vector = group['FPOGY'].tolist()

    # Initialize counters
    count_asc = count_desc = count_right = count_left = 0
    tot_asc = tot_desc = tot_right = tot_left = 0.0

    for i in range(len(fpogx_vector) - 1):
        diff_x = fpogx_vector[i+1] - fpogx_vector[i]
        if diff_x > 0:
            count_right += 1
            tot_right += abs(diff_x)
        elif diff_x < 0:
            count_left += 1
            tot_left += abs(diff_x)
        #skip when the difference is zero
        else:
            continue

    for i in range(len(fpogy_vector) - 1):
        diff_y = fpogy_vector[i+1] - fpogy_vector[i]
        if diff_y > 0:
            count_desc += 1
            tot_desc += abs(diff_y)
        elif diff_y < 0:
            count_asc += 1
            tot_asc += abs(diff_y)
        #skip when the difference is zero
        else:
            continue

    movement_vector.append({
        'file_key': file_key,
        'f75': tot_asc,
        'f76': tot_desc,
        'f77': tot_right,
        'f78': tot_left,
        'f79': count_asc,
        'f80': count_desc,
        'f81': count_right,
        'f82': count_left
    })

# Convert to DataFrame and sort by file_key
movement_vector_df = pd.DataFrame(movement_vector)

feature_df = feature_df.merge(movement_vector_df, on='file_key', how='left')
display(feature_df)

#Overwrite the csv 
feature_df.to_csv(r'C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv', index=False)


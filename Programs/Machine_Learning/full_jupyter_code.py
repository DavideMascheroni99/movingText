import pandas as pd
import numpy as np
import duckdb
import ml_constants
from IPython.display import display
from scipy.stats import gmean
from scipy.stats import median_abs_deviation
from scipy.stats import skew
from scipy.stats import iqr
from scipy.stats import kurtosis


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



'''OBTAIN THE DURATION FOR EVERY FPOGID IN A FILE'''
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



'''GEOMETRIC MEAN FOR FIXATION DURATION (F4)'''
geom_mean_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''MEDIAN FOR FIXATION DURATION (F5)'''
median_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''STD FOR FIXATION DURATION (F6)'''
std_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''MEDIAN ABSOLUTE DEVIATION FOR FIXATION DURATION (F7)'''
mad_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''SKEWNESS FOR FIXATION DURATION (F8)'''
skew_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''IQR FOR FIXATION DURATION (F9)'''
iqr_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''IQR FOR FIXATION DURATION (F10)'''
kurt_fixation_data = []

for item in fix_duration_vector:
    file_key = item['file_key']
    max_fpogd = item['max_FPOGD']

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



'''OBTAIN THE DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS DURATION FOR EVERY FPOGID IN A FILE'''
#FPOGS is the starting time of the fixation POG in seconds since the system initialization or calibration. 
#I used this time to compute the difference between two consecutive fixations in the same file
fpogs_diff_vector = []

# Loop through all DataFrames in the dictionary
for key, df in all_dfs.items():
    # Query using DuckDB
    result = duckdb.query("""
        SELECT FPOGID, FPOGS 
        FROM df
        GROUP BY FPOGID, FPOGS
        ORDER BY FPOGID
    """).to_df()

    # Extract vectors
    fpogid_vector = result['FPOGID'].tolist()
    fpogs_vector = result['FPOGS'].tolist()

    # Build list of consecutive FPOGID pairs 
    fpogid_pairs = []
    for i in range(len(fpogid_vector) - 1):
        pair = (fpogid_vector[i], fpogid_vector[i + 1])
        fpogid_pairs.append(pair)

    # Build list of consecutive FPOGS difference 
    fpogs_differences = []
    for i in range(len(fpogs_vector) - 1):
        diff = fpogs_vector[i + 1] - fpogs_vector[i]
        fpogs_differences.append(diff)

    # Append result 
    fpogs_diff_vector.append({
        'file_key': key,
        'FPOGID_pairs': fpogid_pairs,
        'FPOGS_differences': fpogs_differences
    })

# Create the DataFrame
fpogs_diff_vector_df = pd.DataFrame(fpogs_diff_vector)

# Display the DataFrame
display(fpogs_diff_vector_df)



'''MINIMUM FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F11)'''
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



'''MAXIMUM FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F12)'''
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



'''ARITHMETIC MEAN FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F13)'''
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



'''GEOMETRIC MEAN FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F14)'''
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



'''MEDIAN FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F15)'''
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



'''STD FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F16)'''
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



'''MAD FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F17)'''
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



'''SKEWNESS FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F18)'''
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



'''IQR FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F19)'''
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



'''KURTOSIS FOR DIFFERENCE BETWEEN CONSECUTIVE FIXATIONS(F20)'''
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

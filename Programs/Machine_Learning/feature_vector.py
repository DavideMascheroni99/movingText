'''import pandas as pd
import ml_constants
from IPython.display import display
import duckdb


#Load all csv files
def load_csv():
    all_dfs = {}

    import pandas as pd

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
    return all_dfs


#Count the number of fixations
def fixation_number(all_dfs):

    feature_vectors = []
    # Loop through all files in the dataframe
    for key, df in all_dfs.items():
        num_fix = 0
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
            'num_fix': num_fix
        })

    # Create a DataFrame from all feature vectors
    feature_df = pd.DataFrame(feature_vectors)

    # Display the current dataset
    display(feature_df)



#All the fixation with its duration in each file
def all_fixations(all_dfs):

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

        # Store in the feature vector list
        fix_duration_vector.append({
            'file_key': key,
            'FPOGID': result['FPOGID'].tolist(),
            'max_FPOGD': max_fpogd_vector
        })

        print(f"File: {key}")
        print(result)


#Minimun fixation duration that join with the feature vector
def min_fixation_duration():

    min_fixation_data = []

    for item in fix_duration_vector:
        file_key = item['file_key']
        max_fpogd = item['max_FPOGD']

        #Check if empty
        if max_fpogd: 
            min_duration = min(max_fpogd)
        else:
            min_duration = None 

        min_fixation_data.append({
            'file_key': file_key,
            'min_fixation_duration': min_duration
        })

    # Create pandas DataFrame
    min_fixation_df = pd.DataFrame(min_fixation_data)

    feature_df = feature_df.merge(min_fixation_df, on='file_key', how='left')
    display(feature_df)



def main():
    all_dfs = load_csv()
    fixation_number(all_dfs)

if __name__ == "__main__":
    main()

'''
import pandas as pd
import ml_constants
from IPython.display import display


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



def main():
    all_dfs = load_csv()
    fixation_number(all_dfs)

if __name__ == "__main__":
    main()


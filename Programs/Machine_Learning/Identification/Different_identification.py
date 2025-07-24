import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings

# disable an unexpected warning on the new pandas version
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

'''LOAD THE DATASET'''
# csv_path of the PC in the lab
csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
#csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

#Obtain the animation name from the file key
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))

num_seed = 3

'''DEFINITION OF EACH PIPELINE WITH THEIR RESPECTIVE PARAMETER GRID'''

def get_rf_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', RandomForestClassifier())
    ])
    param_grid = {
        'scaler': [StandardScaler(), RobustScaler(), MinMaxScaler()],  
        'clf__n_estimators': [700],
        'clf__criterion': ['gini', 'log_loss'],
        'clf__max_depth': [5, 9],
        'clf__random_state': [0]
    }
    return pipeline, param_grid


def get_svc_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # placeholder
        ('clf', SVC())
    ])
    param_grid = {
        'scaler': [StandardScaler(), RobustScaler(), MinMaxScaler()],
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': [1, 0.1, 0.01],
        'clf__kernel': ['rbf', 'sigmoid']
    }
    return pipeline, param_grid


'''GRID SEARCH FUNCTION'''

def run_grid_search(X_train, y_train, X_test, y_test, pipeline, param_grid, title):
    print(f"\n=== {title} ===")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    train_score = grid_search.best_estimator_.score(X_train, y_train)
    test_score = grid_search.best_estimator_.score(X_test, y_test)

    print("Best parameters:", best_params)
    print("Best CV accuracy:", best_cv_score)
    print("Train accuracy:", train_score)
    print("Test accuracy:", test_score)

    return best_params, best_cv_score, train_score, test_score

'''WRITE RESULTS FUNCTION'''

def write_results(title, best_params, best_cv_score, train_score, test_score, results_path, animation_name):
    results = {
        'Model': title,
        'Animation': animation_name,
        'Best Parameters': str(best_params),
        'Best CV Accuracy': round(best_cv_score, 4),
        'Train Accuracy': round(train_score, 4),
        'Test Accuracy': round(test_score, 4)
    }
    df = pd.DataFrame([results])
    if not os.path.exists(results_path):
        df.to_csv(results_path, index=False)
    else:
        df.to_csv(results_path, mode='a', header=False, index=False)

'''RUN THE MODELS FOR EACH ANIMATION'''

# --- Model definitions ---
model_list = [
    ("Random Forest", get_rf_pipeline),
    ("SVC", get_svc_pipeline)
]

# --- Results file path ---
results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Single_feature_results_df.csv"

# Delete previous results file if it exists
if os.path.exists(results_file):
    os.remove(results_file)

# Get all unique animations
animation_names = dataset['anim_name'].unique()

dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

feature_cols = list(dataset.loc[:, 'f0':'f71'].columns)

animation_names = dataset['anim_name'].unique()

# --- PART 1: RANDOM SPLIT PER ANIMATION (as before) ---
for anim in animation_names:
    print(f"\n=== Animation: {anim} ===")
    subset = dataset[dataset['anim_name'] == anim]
    X = subset.loc[:, feature_cols]
    y = subset['tester_id']

    for model_name, model_fn in model_list:
        print(f"\n--- {model_name} ---")
        
        best_cv_scores, train_scores, test_scores = [], [], []
        best_param_list = []
        num_seed = 10

        for seed in range(num_seed):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )
            pipeline, param_grid = model_fn()
            best_params, best_cv_score, train_score, test_score = run_grid_search(
                X_train, y_train, X_test, y_test,
                pipeline, param_grid,
                f"{model_name} (Random split run {seed+1} - {anim})"
            )
            best_cv_scores.append(best_cv_score)
            train_scores.append(train_score)
            test_scores.append(test_score)
            best_param_list.append((best_params, best_cv_score))

        mean_cv = np.mean(best_cv_scores)
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        best_params = max(best_param_list, key=lambda x: x[1])[0]

        write_results(
            f"{model_name} (Random split mean - {anim})",
            best_params, mean_cv, mean_train, mean_test,
            results_file, animation_name=anim
        )

# --- PART 2: SESSION-BASED TRAINING (train on all S1+S2, test on each animation S3) ---

print("\n=== Session-based training on all animations combined ===")

# Prepare train data for S1+S2 across all animations
train_data = dataset[dataset['session_id'].isin(['S1', 'S2'])]
X_train_all = train_data.loc[:, feature_cols]
y_train_all = train_data['tester_id']

for model_name, model_fn in model_list:
    print(f"\n--- {model_name} ---")

    pipeline, param_grid = model_fn()
    best_params, best_cv_score, train_score, _ = run_grid_search(
        X_train_all, y_train_all, X_train_all, y_train_all,
        pipeline, param_grid,
        model_name + " (All S1+S2 training)"
    )
    pipeline.set_params(**best_params)
    pipeline.fit(X_train_all, y_train_all)

    for anim in animation_names:
        test_subset = dataset[(dataset['anim_name'] == anim) & (dataset['session_id'] == 'S3')]
        if test_subset.empty:
            print(f"Skipping animation {anim} — no S3 data.")
            continue
        X_test_sess = test_subset.loc[:, feature_cols]
        y_test_sess = test_subset['tester_id']
        test_score = pipeline.score(X_test_sess, y_test_sess)
        print(f"{model_name} Test accuracy on animation {anim} (S3): {test_score:.4f}")
        write_results(
            f"{model_name} (S1+S2 → {anim}_S3)",
            best_params, best_cv_score, train_score, test_score,
            results_file, animation_name=anim
        )

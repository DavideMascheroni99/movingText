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

dataset = pd.read_csv(csv_path)

#Obtain the animation name from the file key
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))

'''DEFINITION OF EACH PIPELINE WITH THEIR RESPECTIVE PARAMETER GRID'''

def get_rf_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
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
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()), 
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

def write_results(title, best_params, best_cv_score, train_score, test_score, results_path, anim):
    results = {
        'Model': title,
        'Animation': anim,
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

model_list = [
    ("Random Forest", get_rf_pipeline),
    ("SVC", get_svc_pipeline)
]

# Result file path
results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Indentification_test\Identification_single_mp.csv" 

# Delete previous results file if exists
if os.path.exists(results_file):
    os.remove(results_file)

# Get all unique animations in the dataset
animation_names = dataset['anim_name'].unique()

# Put every animation's first two sessions in the train
train_subset = dataset[dataset['session_id'].isin(['S1', 'S2'])]

X_train_sess = train_subset.loc[:, 'f0':'f71']
y_train_sess = train_subset['tester_id']

for anim in animation_names:

    # Subset with all the data for a single animation
    subset = dataset[dataset['anim_name'] == anim].copy()
    # Test subset contains only the S3 data for every animations
    test_subset = subset[subset['session_id'] == 'S3']

    subset['tester_id'] = subset['file_key'].apply(lambda x: x.split('_')[0])
    subset['session_id'] = subset['file_key'].apply(lambda x: x.split('_')[1])

    X_test_sess = test_subset.loc[:, 'f0':'f71']
    y_test_sess = test_subset['tester_id']

    '''# Extract from the subset the tester and the session id
    subset['tester_id'] = subset['file_key'].apply(lambda x: x.split('_')[0])
    subset['session_id'] = subset['file_key'].apply(lambda x: x.split('_')[1])

    X = subset.loc[:, 'f0':'f82']
    y = subset['tester_id']'''


    '''for model_name, model_fn in model_list:
        best_cv_scores, train_scores, test_scores = [], [], []
        best_param_list = []
        num_seed = 10

        for i in range(num_seed):
            X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
                X, y, test_size=0.2, random_state=i, stratify=y
            )

            pipeline, param_grid = model_fn()
            best_params, best_cv_score, train_score, test_score = run_grid_search(
                X_train_rand, y_train_rand, X_test_rand, y_test_rand, pipeline, param_grid, model_name + f" (80/20 Run {i+1})")

            best_cv_scores.append(best_cv_score)
            train_scores.append(train_score)
            test_scores.append(test_score)
            best_param_list.append((best_params, best_cv_score))

        mean_cv = np.mean(best_cv_scores)
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        best_params = max(best_param_list, key=lambda x: x[1])[0]

        write_results(model_name + " (80/20)", best_params, mean_cv, mean_train, mean_test, results_file, anim)'''

    for model_name, model_fn in model_list:
        pipeline, param_grid = model_fn()
        best_params, best_cv_score, train_score, test_score = run_grid_search(
            X_train_sess, y_train_sess, X_test_sess, y_test_sess, pipeline, param_grid, model_name + " (S1+S2 vs S3)")

        write_results(model_name + " (S1+S2 vs S3)", best_params, best_cv_score, train_score, test_score, results_file, anim)
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
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, f_classif


# disable an unexpected warning on the new pandas version
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

'''LOAD THE DATASET'''
#csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"

dataset = pd.read_csv(csv_path)

# Obtain the animation name from the file key
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

'''DEFINITION OF EACH PIPELINE WITH THEIR RESPECTIVE PARAMETER GRID'''

def get_nb_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('nb', GaussianNB())
    ])
    param_grid = {'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
                  'feature_selection__k': [30, 40, 50, 60, 70]}
    return pipeline, param_grid

'''def get_knn_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski', 'euclidean', 'manhattan']
    }
    return pipeline, param_grid

def get_logreg_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('logreg', LogisticRegression(max_iter=1000, random_state=0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    return pipeline, param_grid

def get_nusvc_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('nusvc', NuSVC())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'nusvc__nu': [0.25, 0.5, 0.75],
        'nusvc__kernel': ['rbf', 'poly', 'sigmoid'],
        'nusvc__gamma': ['scale', 'auto']
    }
    return pipeline, param_grid

def get_rf_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'rf__n_estimators': [20, 30, 50, 100, 200],
        'rf__max_features': ['sqrt'],
        'rf__max_depth': [5, 10, 20, 30]
    }
    return pipeline, param_grid

def get_svc_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('svc', SVC())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'svc__kernel': ['rbf', 'poly']
    }
    return pipeline, param_grid

def get_mlp_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('mlp' , MLPClassifier(max_iter=4000, random_state = 0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'mlp__hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'mlp__activation': ['tanh', 'relu'],
        'mlp__alpha':  [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__solver': ['adam']
        }
    return pipeline, param_grid'''

'''GRID SEARCH FUNCTION'''

def run_grid_search(X, y, pipeline, param_grid, title, seed=0):

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    train_score = grid_search.best_estimator_.score(X, y)

    return best_params, best_cv_score, train_score, None, grid_search.best_estimator_

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

model_list = [
    ("Naive Bayes", get_nb_pipeline),
    #("KNN", get_knn_pipeline),
    #("Logistic Regression", get_logreg_pipeline),
    #("NuSVC", get_nusvc_pipeline),
    #("Random Forest", get_rf_pipeline),
    #("SVC", get_svc_pipeline),
    #("MLP", get_mlp_pipeline)
]

# Result file path
#results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"
results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"

# Delete previous results file if exists
if os.path.exists(results_file):
    os.remove(results_file)

# Get all unique animations in the dataset
animation_names = dataset['anim_name'].unique()

'''RANDOM SPLIT TRAINING'''

num_seed = 10

for model_name, model_fn in model_list:
    for anim in animation_names:
        subset = dataset[dataset['anim_name'] == anim].copy()

        X = subset.loc[:, 'f0':'f82']
        y = subset['tester_id']

        best_cv_scores, train_scores, best_param_list = [], [], []
        test_scores_per_seed = []

        for i in range(num_seed):
            X_train_anim, X_test_anim, y_train_anim, y_test_anim = train_test_split(
                X, y, test_size=0.2, random_state=i, stratify=y
            )

            pipeline, param_grid = model_fn()
            best_params, best_cv_score, train_score, _, best_estimator = run_grid_search(
                X_train_anim, y_train_anim, pipeline, param_grid, model_name + f" (80/20 Run {i+1})", seed=i)

            best_cv_scores.append(best_cv_score)
            train_scores.append(train_score)
            best_param_list.append((best_params, best_cv_score))

            test_score = best_estimator.score(X_test_anim, y_test_anim)
            test_scores_per_seed.append(test_score)

        mean_cv = np.mean(best_cv_scores)
        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores_per_seed)
        best_params = max(best_param_list, key=lambda x: x[1])[0]

        write_results(
            model_name + " (80/20)", 
            best_params,
            mean_cv,
            mean_train,
            mean_test,    
            results_file,
            anim
        )

'''SPLIT S1+S2 vs S3 PER ANIMATION'''

for model_name, model_fn in model_list:
    for anim in animation_names:
        df_anim = dataset[dataset['anim_name'] == anim]

        train_subset = df_anim[df_anim['session_id'].isin(['S1', 'S2'])]
        test_subset = df_anim[df_anim['session_id'] == 'S3']

        X_train_sess = train_subset.loc[:, 'f0':'f82']
        y_train_sess = train_subset['tester_id']
        X_test_sess = test_subset.loc[:, 'f0':'f82']
        y_test_sess = test_subset['tester_id']

        pipeline, param_grid = model_fn()
        best_params, best_cv_score, train_score, _, best_estimator = run_grid_search(
            X_train_sess, y_train_sess, pipeline, param_grid, model_name + " (S1+S2 vs S3)"
        )

        test_score = best_estimator.score(X_test_sess, y_test_sess)

        write_results(
            model_name + " (S1+S2 vs S3)",
            best_params,
            best_cv_score,
            train_score,
            test_score,
            results_file,
            anim
        )

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
#csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

#Obtain the animation name from the file key
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))

'''DEFINITION OF EACH PIPELINE WITH THEIR RESPECTIVE PARAMETER GRID'''

def get_nb_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'nb__var_smoothing': [1e-9, 1e-8, 1e-7]  # increase smoothing to reduce overfitting
    }
    return pipeline, param_grid

def get_knn_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'knn__n_neighbors': [3, 5],               # fewer neighbors to avoid overfitting
        'knn__weights': ['uniform'],               # simpler weights
        'knn__metric': ['minkowski']               # restrict metric for consistency
    }
    return pipeline, param_grid

def get_logreg_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=0, penalty='l2', solver='lbfgs'))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'logreg__C': [0.01, 0.1, 1],              # stronger regularization (smaller C)
    }
    return pipeline, param_grid

def get_nusvc_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('nusvc', NuSVC())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'nusvc__nu': [0.25, 0.5],                 # smaller nu to control support vectors
        'nusvc__kernel': ['rbf'],                  # focus on 'rbf' kernel for better generalization
        'nusvc__gamma': ['scale']
    }
    return pipeline, param_grid

def get_rf_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'rf__n_estimators': [50, 100],             # moderate number of trees
        'rf__max_features': ['sqrt'],
        'rf__max_depth': [5, 10],                   # shallower trees to reduce overfitting
        'rf__min_samples_split': [5, 10],           # increase min samples per split to regularize
        'rf__min_samples_leaf': [2, 4]               # min samples per leaf
    }
    return pipeline, param_grid

def get_svc_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('svc', SVC())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'svc__C': [0.01, 0.1, 1],                  # stronger regularization (smaller C)
        'svc__gamma': ['scale'],
        'svc__kernel': ['rbf']                      # restrict to 'rbf' for better generalization
    }
    return pipeline, param_grid

def get_mlp_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('mlp' , MLPClassifier(max_iter=2000, random_state=0, early_stopping=True))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'mlp__hidden_layer_sizes': [(50,), (100,)],  # simpler networks
        'mlp__activation': ['relu'],                  # prefer relu for better convergence
        'mlp__alpha': [0.001, 0.01, 0.1],             # stronger L2 regularization
        'mlp__learning_rate_init': [0.001],
        'mlp__solver': ['adam']
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

def write_results(title, best_params, best_cv_score, train_score, test_score, results_path):
    results = {
        'Model': title,
        'Animation': animation_name,
        'Best Parameters': str(best_params),
        'Best CV Accuracy': best_cv_score,
        'Train Accuracy': train_score,
        'Test Accuracy': test_score
    }
    df = pd.DataFrame([results])
    if not os.path.exists(results_path):
        df.to_csv(results_path, index=False)
    else:
        df.to_csv(results_path, mode='a', header=False, index=False)

'''RUN THE MODELS FOR EACH ANIMATION'''

model_list = [
    ("Naive Bayes", get_nb_pipeline),
    ("KNN", get_knn_pipeline),
    ("Logistic Regression", get_logreg_pipeline),
    ("NuSVC", get_nusvc_pipeline),
    ("Random Forest", get_rf_pipeline),
    ("SVC", get_svc_pipeline),
    ("MLP", get_mlp_pipeline)
]

# Result file path
#results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Single_feature_results.csv"
# results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Single_feature_results.csv"


# Get all unique animations in the dataset
animation_names = dataset['anim_name'].unique()

for anim in animation_names:
    animation_name = anim
    subset = dataset[dataset['anim_name'] == anim].copy()
    subset['tester_id'] = subset['file_key'].apply(lambda x: x.split('_')[0])
    subset['session_id'] = subset['file_key'].apply(lambda x: x.split('_')[1])

    X = subset.loc[:, 'f0':'f82']
    y = subset['tester_id']

    train_subset = subset[subset['session_id'].isin(['S1', 'S2'])]
    test_subset = subset[subset['session_id'] == 'S3']

    X_train_sess = train_subset.loc[:, 'f0':'f82']
    y_train_sess = train_subset['tester_id']
    X_test_sess = test_subset.loc[:, 'f0':'f82']
    y_test_sess = test_subset['tester_id']

print(f"\n[Animation: {animation_name}]")
print(f"S1+S2 train size: {len(X_train_sess)}, S3 test size: {len(X_test_sess)}")

    
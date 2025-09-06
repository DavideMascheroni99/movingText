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
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from collections import defaultdict


# disable an unexpected warning on the new pandas version
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# Delete the file if already exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Create the csv if doesn't exists, otherwhise append to the existing one
def append_to_csv(df, file_path):
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

'''CONSTANTS'''
num_seed = 10

csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
#csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"

selected_features_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_st.csv"
#selected_features_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_st.csv"
delete_file_if_exists(selected_features_file)

results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"
#results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"
delete_file_if_exists(results_file)

'''LOAD THE DATASET'''

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
                  'feature_selection__k': [20, 30, 40, 50, 60, 70]}
    return pipeline, param_grid


def get_knn_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [20, 30, 40, 50, 60, 70],
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
        'feature_selection__k': [20, 30, 40, 50, 60, 70],
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
        'feature_selection__k': [20, 30, 40, 50, 60, 70],
        'nusvc__nu': [0.25, 0.5, 0.75],             
        'nusvc__kernel': ['rbf'],                  
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
        'feature_selection__k': [20, 30, 40, 50, 60, 70],
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
        'feature_selection__k': [20, 30, 40, 50, 60, 70],
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
        'feature_selection__k': [20, 30, 40, 50, 60, 70],
        'mlp__hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'mlp__activation': ['tanh', 'relu'],
        'mlp__alpha':  [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__solver': ['adam']
        }
    return pipeline, param_grid

'''GRID SEARCH FUNCTION'''

def run_grid_search(X, y, pipeline, param_grid):

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    train_score = grid_search.best_estimator_.score(X, y)

    return best_params, best_cv_score, train_score, grid_search.best_estimator_

'''WRITE RESULTS FUNCTION'''

def write_results(title, best_params, best_cv_score, train_score, test_score, results_path, animation_name):
    # Extract k
    best_k = best_params.get('feature_selection__k', None)  
    results = {
        'Model': title,
        'Animation': animation_name,
        'Best Parameters': str(best_params),
        'Best K': best_k, 
        'Best CV Accuracy': round(best_cv_score, 4),
        'Train Accuracy': round(train_score, 4),
        'Test Accuracy': round(test_score, 4)
    }
    df = pd.DataFrame([results])
    append_to_csv(df, results_path)
 
'''BEST K FEATURES'''
def save_selected_features(model_name, animation_name, best_k, selector, columns, results_path):
    # Get F-scores
    f_scores = selector.scores_
    features = np.array(columns)
    
    # Replace NaN with -inf to avoid selection issues
    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
    
    # Get indices of top-k F-scores in descending order
    top_indices = np.argsort(f_scores)[-best_k:][::-1] 
    
    selected_features = features[top_indices]
    selected_f_scores = f_scores[top_indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': model_name,
        'Animation': animation_name,
        'Best K': best_k,
        'Feature': selected_features,
        'F-score': np.round(selected_f_scores, 4)
    })
    append_to_csv(df, results_path)


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

# Get all unique animations in the dataset
animation_names = dataset['anim_name'].unique()

'''RANDOM SPLIT TRAINING & S1+S2 vs S3 PER ANIMATION'''

# Return X_train, X_test, y_train, y_test depending on split type
def split_dataset(df_anim, split_type, seed):

    if split_type == "80/20":
        X = df_anim.loc[:, 'f0':'f82']
        y = df_anim['tester_id']
        return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    elif split_type == "S1+S2 vs S3":
        train_subset = df_anim[df_anim['session_id'].isin(['S1', 'S2'])]
        test_subset = df_anim[df_anim['session_id'] == 'S3']
        X_train = train_subset.loc[:, 'f0':'f82']
        y_train = train_subset['tester_id']
        X_test = test_subset.loc[:, 'f0':'f82']
        y_test = test_subset['tester_id']
        return X_train, X_test, y_train, y_test
    
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


def evaluate_model_on_seed(df_anim, model_fn, split_type, seed):
    # Perform a new split for this seed
    X_train, X_test, y_train, y_test = split_dataset(df_anim, split_type, seed)
    
    # Run GridSearch
    pipeline, param_grid = model_fn()
    best_params, best_cv_score, train_score, best_estimator = run_grid_search(X_train, y_train, pipeline, param_grid)
    
    # Evaluate on test set
    test_score = best_estimator.score(X_test, y_test)
    
    return best_params, best_cv_score, train_score, test_score, best_estimator



def aggregate_seeds(df_anim, model_fn, split_type, num_seed=10):
    all_results = []
    param_cv_dict = defaultdict(list)
    param_estimator_dict = {}

    for seed in range(num_seed):
        best_params, best_cv_score, train_score, test_score, best_estimator = evaluate_model_on_seed(df_anim, model_fn, split_type, seed)
        all_results.append((best_params, best_cv_score, train_score, test_score, best_estimator))

        # Convert params dict to a tuple key for aggregation
        param_key = tuple(sorted(best_params.items()))
        param_cv_dict[param_key].append(best_cv_score)

        # Keep one estimator per param combo
        if param_key not in param_estimator_dict:
            param_estimator_dict[param_key] = best_estimator

    # Compute mean CV per parameter combination
    mean_cv_per_param = {k: np.mean(v) for k, v in param_cv_dict.items()}

    # Select parameters with highest mean CV
    best_param_key = max(mean_cv_per_param, key=mean_cv_per_param.get)
    best_params = dict(best_param_key)
    best_estimator = param_estimator_dict[best_param_key]

    # Average scores
    mean_cv = np.mean([x[1] for x in all_results])
    mean_train = np.mean([x[2] for x in all_results])
    mean_test = np.mean([x[3] for x in all_results])

    return best_params, mean_cv, mean_train, mean_test, best_estimator



def process_model_split(model_name, anim, df_anim, model_fn, split_label, results_file, selected_features_file):

    best_params, mean_cv, mean_train, mean_test, best_estimator = aggregate_seeds(df_anim, model_fn, split_label, num_seed=num_seed)

    write_results(f"{model_name} ({split_label})", best_params, mean_cv, mean_train, mean_test, results_file, anim)
    save_selected_features(f"{model_name} ({split_label})", anim, best_params['feature_selection__k'], best_estimator.named_steps['feature_selection'], df_anim.loc[:, 'f0':'f82'].columns, selected_features_file)



for model_name, model_fn in model_list:
    for anim in animation_names:
        df_anim = dataset[dataset['anim_name'] == anim]

        # Random 80/20 split
        process_model_split(model_name, anim, df_anim, model_fn, "80/20", results_file, selected_features_file)

        # S1+S2 vs S3 split
        process_model_split(model_name, anim, df_anim, model_fn, "S1+S2 vs S3", results_file, selected_features_file)

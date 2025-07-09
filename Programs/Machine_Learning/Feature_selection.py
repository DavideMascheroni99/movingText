import os
import pandas as pd
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
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
import warnings
import matplotlib.pyplot as plt

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

# Extract the tester id and the session id from file_key
dataset['person_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

# The X are the whole rows composed by the whole columns, while the labels are the id of the persons
X = dataset.loc[:, 'f0':'f82']
y = dataset['person_id']

'''DIFFERENT SPLIT OF THE DATA'''
# Random split (80/20) with stratification
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Session split (S1+S2 for training, S3 for test)
train_subset = dataset[dataset['session_id'].isin(['S1', 'S2'])]
test_subset = dataset[dataset['session_id'] == 'S3']
X_train_sess = train_subset.loc[:, 'f0':'f82']
y_train_sess = train_subset['person_id']
X_test_sess = test_subset.loc[:, 'f0':'f82']
y_test_sess = test_subset['person_id']

'''DEFINITION OF EACH PIPELINE WITH THEIR RESPECTIVE PARAMETER GRID'''

def get_nb_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('nb', GaussianNB())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70]
        }
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
        ('feature_selection', RFE(LogisticRegression(max_iter=1000, random_state=0), n_features_to_select=50)),
        ('logreg', LogisticRegression(max_iter=1000, random_state=0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
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
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0))),
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
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
        ('mlp' , MLPClassifier(max_iter=3000, random_state = 0))
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
    return pipeline, param_grid

'''GRID SEARCH FUNCTION'''

def run_grid_search(X_train, y_train, X_test, y_test, pipeline, param_grid, title, results_path=None):
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
    
    if results_path:
        results = {
            'Model': title,
            'Best Parameters': str(best_params),
            'Best CV Accuracy': best_cv_score,
            'Train Accuracy': train_score,
            'Test Accuracy': test_score
        }
        if not os.path.exists(results_path):
            df = pd.DataFrame([results])
            df.to_csv(results_path, index=False)
        else:
            df = pd.DataFrame([results])
            df.to_csv(results_path, mode='a', header=False, index=False)
    
    return grid_search

'''PLOT THE K BEST VALUES WITH ITS F-SCORE'''

def plot_top_features(grid_search, X_train, model_name, split_name, save_dir):

    feature_names = X_train.columns

    selector = grid_search.best_estimator_.named_steps['feature_selection']
    support_mask = selector.get_support()
    scores = selector.scores_

    selected_features = feature_names[support_mask]
    selected_scores = scores[support_mask]

    sorted_idx = selected_scores.argsort()[::-1]
    sorted_features = selected_features[sorted_idx]
    sorted_scores = selected_scores[sorted_idx]

    k = grid_search.best_params_['feature_selection__k']

    plt.figure(figsize=(12, max(6, k * 0.3)))
    plt.barh(sorted_features[:k][::-1], sorted_scores[:k][::-1], color='skyblue')
    plt.xlabel('ANOVA F-score')
    plt.title(f'Top {k} Features for {model_name} ({split_name} split)')
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.yticks(fontsize=9)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_{split_name.replace(' ', '_').replace('/', '-')}.png")
    plt.savefig(save_path)



'''RUN THE MODELS'''

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
results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_results_fs.csv"
#results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_results_fs.csv"

best_features_dir = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_KBest"
#best_features_dir = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_KBest"

# If i rerun the code I want to delete the previous results file
if os.path.exists(results_file):
    os.remove(results_file)  

# 1. Random Split (80/20)
for model_name, model_fn in model_list:
    pipeline, param_grid = model_fn()
    gs = run_grid_search(X_train_rand, y_train_rand, X_test_rand, y_test_rand, pipeline, param_grid, model_name + " (80/20)", results_path=results_file)
    plot_top_features(gs, X_train_rand, model_name, "80/20", best_features_dir)

# 2. Session Split (S1+S2 → train, S3 → test)
for model_name, model_fn in model_list:
    pipeline, param_grid = model_fn()
    gs = run_grid_search(X_train_sess, y_train_sess, X_test_sess, y_test_sess, pipeline, param_grid, model_name + " (S1+S2 vs S3)", results_path=results_file)
    plot_top_features(gs, X_train_sess, model_name, "S1+S2_vs_S3", best_features_dir)

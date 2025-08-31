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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve
import warnings
from collections import defaultdict

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

'''LOAD DATASET'''
#csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

# Extract animation name, tester id and session id
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

features_cols = [f'f{i}' for i in range(83)]
animation_names = dataset['anim_name'].unique()
people = dataset['tester_id'].unique()

'''PIPELINE DEFINITION'''
def get_nb_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB())
    ])
    param_grid = {'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()]}
    return pipeline, param_grid
'''
def get_knn_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski', 'euclidean', 'manhattan']
    }
    return pipeline, param_grid

def get_logreg_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
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
        ('nusvc', NuSVC(probability=True))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'nusvc__nu': [0.25, 0.5, 0.75],
        'nusvc__kernel': ['rbf', 'poly', 'sigmoid'],
        'nusvc__gamma': ['scale', 'auto']
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
        'rf__n_estimators': [20, 30, 50, 100, 200],
        'rf__max_features': ['sqrt'],
        'rf__max_depth': [5, 10, 20, 30]
    }
    return pipeline, param_grid

def get_svc_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('svc', SVC(probability=True))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'svc__kernel': ['rbf', 'poly']
    }
    return pipeline, param_grid

def get_mlp_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('mlp', MLPClassifier(max_iter=2000, random_state=0))
    ])
    param_grid = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'mlp__hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'mlp__activation': ['tanh', 'relu'],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__solver': ['adam']
    }
    return pipeline, param_grid'''

model_list = [
    ("Naive Bayes", get_nb_pipeline),
    #("KNN", get_knn_pipeline),
    #("Logistic Regression", get_logreg_pipeline),
    #("NuSVC", get_nusvc_pipeline),
    #("Random Forest", get_rf_pipeline),
    #("SVC", get_svc_pipeline),
    #("MLP", get_mlp_pipeline)
]

'''Prepare train and test data for a single person'''
def prepare_train_test_data(person_data, split_type, seed):
    if split_type == 'session':
        # S1+S2 for train, S3 for test
        train_data = person_data[person_data['session_id'].isin(['S1', 'S2'])]
        test_data = person_data[person_data['session_id'] == 'S3']

        # Create impostors for the current person
        impostors = dataset[dataset['tester_id'] != person_data['tester_id'].iloc[0]]

        # Randomly select impostor samples to match genuine train/test samples
        impostors_train = impostors.sample(n=len(train_data), random_state=seed)
        remaining_impostors = impostors.drop(impostors_train.index)
        impostors_test = remaining_impostors.sample(n=len(test_data), random_state=seed)

        # Combine genuine and impostor samples
        train_combined = pd.concat([train_data, impostors_train])
        test_combined = pd.concat([test_data, impostors_test])

        X_train = train_combined[features_cols]
        y_train = np.array([1]*len(train_data) + [0]*len(impostors_train))
        X_test = test_combined[features_cols]
        y_test = np.array([1]*len(test_data) + [0]*len(impostors_test))

        return X_train, y_train, X_test, y_test

    elif split_type == 'random':
        # Create impostors for the current person
        impostors = dataset[dataset['tester_id'] != person_data['tester_id'].iloc[0]]

        # Label genuine and impostor samples
        genuine = person_data.copy()
        genuine['y'] = 1
        impostors = impostors.sample(frac=1, random_state=seed).reset_index(drop=True)
        impostors['y'] = 0

        # Balance impostors to match genuine size
        num_genuine = len(genuine)
        impostors_balanced = impostors.sample(n=num_genuine, random_state=seed)

        # Combine genuine and impostor samples
        combined = pd.concat([genuine, impostors_balanced]).reset_index(drop=True)

        # Stratified split for train/test
        train_combined, test_combined = train_test_split(
            combined, test_size=0.2, stratify=combined['y'], random_state=seed
        )

        X_train = train_combined[features_cols]
        y_train = train_combined['y'].values
        X_test = test_combined[features_cols]
        y_test = test_combined['y'].values

        return X_train, y_train, X_test, y_test

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

'''RESULT WRITING'''
def write_results(model, split_name, best_params, best_cv_acc, train_acc, test_acc, metrics, results_path, anim_name):
    precision, recall, spec, roc_auc, eer = metrics
    row = {
        'Model': f"{model} ({split_name})",
        'Animation': anim_name,
        'Best Parameters': best_params,
        'Best CV Accuracy': round(best_cv_acc, 4),
        'Train Accuracy': round(train_acc, 4),
        'Test Accuracy': round(test_acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Specificity': round(spec, 4),
        'Roc Auc': round(roc_auc, 4),
        'EER': round(eer, 4)
    }
    df = pd.DataFrame([row])
    if not os.path.exists(results_path):
        df.to_csv(results_path, index=False)
    else:
        df.to_csv(results_path, mode='a', header=False, index=False)

# ------------------- TRAIN AND EVALUATE FUNCTION -------------------
def train_and_evaluate_per_person(X_train_total, y_train_total, animation_test_sets, model_fn, split_name, results_file, num_seed=10):
    for anim_name, (X_test_anim, y_test_anim) in animation_test_sets.items():
        per_person_metrics = []
        per_person_scores = []

        for person in people:
            # select only samples for this animation and person
            person_mask = (X_train_total.index.isin(dataset[(dataset['tester_id']==person) & (dataset['anim_name']==anim_name)].index))
            X_train_person = X_train_total.loc[person_mask]
            y_train_person = y_train_total.loc[person_mask]

            if len(X_train_person) == 0:
                continue

            pipeline, param_grid = model_fn()
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_person, y_train_person)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test_anim)
            if hasattr(best_model, "predict_proba"):
                y_score = best_model.predict_proba(X_test_anim)[:, 1]
            else:
                y_score = best_model.decision_function(X_test_anim)

            test_acc = accuracy_score(y_test_anim, y_pred)
            prec = precision_score(y_test_anim, y_pred)
            rec = recall_score(y_test_anim, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test_anim, y_pred).ravel()
            spec = tn / (tn + fp)
            roc_auc = roc_auc_score(y_test_anim, y_score)
            fpr, tpr, _ = roc_curve(y_test_anim, y_score)
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fnr - fpr))
            eer = fpr[eer_index]

            per_person_metrics.append((prec, rec, spec, roc_auc, eer))
            per_person_scores.append(test_acc)

        # average over all persons for this animation
        mean_test = np.mean(per_person_scores)
        mean_metrics = np.mean(per_person_metrics, axis=0)
        best_params = grid.best_params_

        write_results(
            model_name,
            split_name,
            best_params,
            grid.best_score_,
            best_model.score(X_train_total, y_train_total),
            mean_test,
            mean_metrics,
            results_file,
            anim_name
        )

# ------------------- FILE PATH -------------------
results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results.csv"
if os.path.exists(results_file):
    os.remove(results_file)

# ------------------- RANDOM SPLIT PER ANIMATION (10 seeds) -------------------
num_seed = 10
for model_name, model_fn in model_list:
    for anim in animation_names:

        # Dictionary to accumulate per-person results over seeds
        per_animation_metrics = []

        for person in dataset['tester_id'].unique():
            person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == anim)]
            if len(person_data) == 0:
                continue

            # Accumulate metrics over seeds
            per_person_test_acc = []
            per_person_metrics = []

            for seed in range(num_seed):
                X_train_p, y_train_p, X_test_p, y_test_p = prepare_train_test_data(person_data, 'random', seed=seed)
                animation_test_sets = {anim: (X_test_p, y_test_p)}

                # Train and evaluate
                train_and_evaluate_per_person(
                    X_train_p,
                    pd.Series(y_train_p),
                    animation_test_sets,
                    model_fn,
                    "Random 80/20",
                    results_file,
                    num_seed=num_seed
                )

# ------------------- SESSION SPLIT PER ANIMATION -------------------
for model_name, model_fn in model_list:
    for anim in animation_names:
        for person in dataset['tester_id'].unique():
            person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == anim)]
            if len(person_data) == 0:
                continue

            X_train_p, y_train_p, X_test_p, y_test_p = prepare_train_test_data(person_data, 'session', seed=0)
            animation_test_sets = {anim: (X_test_p, y_test_p)}

            train_and_evaluate_per_person(
                X_train_p,
                pd.Series(y_train_p),
                animation_test_sets,
                model_fn,
                "S1+S2 vs S3",
                results_file,
                num_seed=1
            )

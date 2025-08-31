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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
from collections import defaultdict
from sklearn.metrics import roc_curve

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

'''PIPELINE DEFINITION'''

def get_nb_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB())
    ])
    param_grid = {'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()]}
    return pipeline, param_grid

'''def get_knn_pipeline():
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

# Create a list of pairs of ids of a genuine and an impostor sample
def generate_impostors(X, y, seed):
    unique_ids = y.unique()
    pairs = []
    # Make the program reproducible
    rng = np.random.RandomState(seed)

    # loop over all unique ids
    for uid in unique_ids:
        # In impostor_ids I put the ids different from the current one
        impostor_ids = unique_ids[unique_ids != uid]
        if len(impostor_ids) > 0:
            # Choose a random impostor from the previous list
            impostor_id = rng.choice(impostor_ids)
            # Take a genuine sample
            idx = y[y == uid].index[0]
            # Store this pair of ids
            pairs.append((idx, impostor_id))
    return pairs

def evaluate_verification(model, X_test, y_test, impostor_pairs):
    y_true, y_scores = [], []
    # For each genuine sample, I record the probability that the model assigns to the correct tester which is marked as 1.
    for idx, true_id in zip(X_test.index, y_test):
        # Predict probability of a sample belonging to each of the classes
        prob = model.predict_proba(X_test.loc[[idx]])[0]
        # Probability assigned to the correct tester
        predicted_score = prob[list(model.classes_).index(true_id)]
        # Genuine = 1
        y_true.append(1) 
        y_scores.append(predicted_score)
    # For each genuine sample, check the probability the model assigns to a random impostor ID, mark it as 0, and store that probability for evaluation
    for idx, impostor_id in impostor_pairs:
        prob = model.predict_proba(X_test.loc[[idx]])[0]
        predicted_score = prob[list(model.classes_).index(impostor_id)]
        # Impostor = 0
        y_true.append(0)
        y_scores.append(predicted_score)

    auc = roc_auc_score(y_true, y_scores)
    # empty list to store predictions
    preds = [] 
    for score in y_scores:
        if score >= 0.5:
            # Genuine
            preds.append(1) 
        else:
            # Impostor
            preds.append(0)  
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    spec = tn / (tn + fp)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    # Compute False Negative Rate
    fnr = 1 - tpr
    # Find the point where FNR and FPR are closest
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]

    return prec, rec, spec, auc, eer 


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

#results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results.csv"
results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results.csv"
if os.path.exists(results_file):
    os.remove(results_file)

animation_names = dataset['anim_name'].unique()
features_cols = [f'f{i}' for i in range(83)]

'''RANDOM SPLIT VERIFICATION'''
num_seed = 10 

for model_name, model_fn in model_list:
    for anim in animation_names:
        subset = dataset[dataset['anim_name'] == anim]
        X = subset[features_cols]
        y = subset['tester_id']

        metrics_all = []
        cv_acc_all = []
        train_acc_all = []
        test_acc_all = []
        best_params_all = []

        for seed in range(num_seed):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )

            pipeline, param_grid = model_fn()
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            train_acc = best_model.score(X_train, y_train)
            preds = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, preds)

            impostors = generate_impostors(X_test, y_test, seed)
            metrics = evaluate_verification(best_model, X_test, y_test, impostors)

            metrics_all.append(metrics)
            cv_acc_all.append(grid.best_score_)
            train_acc_all.append(train_acc)
            test_acc_all.append(test_acc)
            best_params_all.append(grid.best_params_)

        # Take the average across all seeds instead of picking the best one
        mean_metrics = np.mean(metrics_all, axis=0)
        mean_cv_acc = np.mean(cv_acc_all)
        mean_train_acc = np.mean(train_acc_all)
        mean_test_acc = np.mean(test_acc_all)

        # Select parameters from the seed with the highest test accuracy
        best_idx = np.argmax(test_acc_all)
        best_params = best_params_all[best_idx]

        write_results(model_name, "Random 80/20", best_params,
                      mean_cv_acc, mean_train_acc, mean_test_acc,
                      mean_metrics, results_file, anim)

'''SESSION SPLIT VERIFICATION (S1+S2 → S3)'''
train_subset = dataset[dataset['session_id'].isin(['S1', 'S2'])]
X_train_sess = pd.DataFrame(train_subset[features_cols], columns=features_cols)
y_train_sess = train_subset['tester_id']

for model_name, model_fn in model_list:
    pipeline, param_grid = model_fn()
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_sess, y_train_sess)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_acc = grid.best_score_
    train_acc = best_model.score(X_train_sess, y_train_sess)

    for anim in animation_names:
        test_subset = dataset[(dataset['anim_name'] == anim) & (dataset['session_id'] == 'S3')]
        if not test_subset.empty:
            X_test_sess = pd.DataFrame(test_subset[features_cols], columns=features_cols)
            y_test_sess = test_subset['tester_id']

            impostors = generate_impostors(X_test_sess, y_test_sess, 1)
            metrics = evaluate_verification(best_model, X_test_sess, y_test_sess, impostors)

            # Compute test accuracy for S3
            preds = best_model.predict(X_test_sess)
            test_acc = accuracy_score(y_test_sess, preds)

            write_results(model_name, "S1+S2 vs S3", best_params, best_cv_acc, train_acc,
              test_acc=test_acc, metrics=metrics, results_path=results_file, anim_name=anim)



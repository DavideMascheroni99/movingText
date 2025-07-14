import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from collections import defaultdict

# disable an unexpected warning on the new pandas version
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# Load dataset
csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

dataset['person_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

# List that contains f0 to f82
features_cols = [f'f{i}' for i in range(83)]
# Get all the different person number
people = dataset['person_id'].unique()
NUM_TRIALS = 10

# Define models and hyperparameter grids
def get_classifiers_with_grid():
    # Naive Bayes
    nb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', GaussianNB())
    ])
    nb_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70]
    }

    # K-Nearest Neighbors
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', KNeighborsClassifier())
    ])
    knn_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__n_neighbors': [3, 5, 7, 9, 11],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['minkowski', 'euclidean', 'manhattan']
    }

    # Logistic Regression
    logreg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', LogisticRegression(max_iter=1000, random_state=0))
    ])
    logreg_params = {
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # NuSVC
    nusvc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', NuSVC(probability=True))
    ])
    nusvc_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__nu': [0.25, 0.5, 0.75],
        'clf__kernel': ['rbf', 'poly', 'sigmoid'],
        'clf__gamma': ['scale', 'auto']
    }

    # Random Forest
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', RandomForestClassifier(random_state=0))
    ])
    rf_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__n_estimators': [20, 30, 50, 100, 200],
        'clf__max_features': ['sqrt'],
        'clf__max_depth': [5, 10, 20, 30]
    }

    # SVC
    svc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', SVC(probability=True))
    ])
    svc_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['rbf', 'poly']
    }

    # MLP Classifier
    mlp_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', MLPClassifier(max_iter=3000, random_state=0))
    ])
    mlp_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'clf__activation': ['tanh', 'relu'],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate_init': [0.001, 0.01],
        'clf__solver': ['adam']
    }

    # Return a list of tuples (name, pipeline, param_grid)
    return [
        ("Naive Bayes", nb_pipeline, nb_params),
        ("KNN", knn_pipeline, knn_params),
        ("Logistic Regression", logreg_pipeline, logreg_params),
        ("NuSVC", nusvc_pipeline, nusvc_params),
        ("Random Forest", rf_pipeline, rf_params),
        ("SVC", svc_pipeline, svc_params),
        ("MLP", mlp_pipeline, mlp_params),
    ]

# Prepare data for a single person
def prepare_train_test_data(person_data, split_type, seed):
    if split_type == 'session':
        train_data = person_data[person_data['session_id'].isin(['S1', 'S2'])]
        test_data = person_data[person_data['session_id'] == 'S3']
    elif split_type == 'random':
        train_data, test_data = train_test_split(person_data, test_size=0.2, stratify=person_data['session_id'], random_state=seed)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    impostors = dataset[dataset['person_id'] != person_data['person_id'].iloc[0]]
    impostors_shuffled = impostors.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Balance impostor samples to match genuine samples
    impostor_train = impostors_shuffled.iloc[:len(train_data)]
    impostor_test = impostors_shuffled.iloc[len(train_data):len(train_data)+len(test_data)]

    # Make balanced train and test sets with equal genuine and impostor samples
    train_combined = pd.concat([train_data, impostor_train])
    test_combined = pd.concat([test_data, impostor_test])
    
    # I extract only the 83 features leaving the file key out cause possible leaking
    X_train = train_combined[features_cols]
    X_test = test_combined[features_cols]
    # Label as 1 the data of the current person and with zero the others
    y_train = np.array([1] * len(train_data) + [0] * len(impostor_train))
    y_test = np.array([1] * len(test_data) + [0] * len(impostor_test))

    return X_train, y_train, X_test, y_test

# Train and evaluate a model
def train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Compute train and test accuracy
    train_accuracy = grid.best_estimator_.score(X_train, y_train)
    test_accuracy = grid.best_estimator_.score(X_test, y_test)

    # Use the best model of grid search to make prediction on the test set
    y_pred = grid.best_estimator_.predict(X_test)
    # Get the probability a sample belongs to class 1
    y_score = grid.best_estimator_.predict_proba(X_test)[:, 1]

    # Get the confusion matrix 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Compute precision, recall and specificity. Check also that denominator is greater than zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    return grid, grid.best_score_, train_accuracy, test_accuracy, grid.best_params_, precision, recall, specificity, fpr, tpr, roc_auc


def compute_mean(test_scores, train_scores, cv_scores, precisions, recalls, specificities, best_params=None):
    avg_test = np.mean(test_scores)
    avg_train = np.mean(train_scores)
    avg_cv = np.mean(cv_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_specificity = np.mean(specificities)
    k = best_params.get('feature_selection__k', 'N/A') if best_params else 'N/A'
    return avg_test, avg_train, avg_cv, avg_precision, avg_recall, avg_specificity, k


def run_random_split(person_data, pipeline, param_grid):
    param_accumulator = defaultdict(list)
    test_scores, train_scores, cv_scores = [], [], []
    precisions, recalls, specificities = [], [], []

    for seed in range(NUM_TRIALS):
        X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, "random", seed)
        grid, best_cv, train_acc, test_acc, best_params, precision, recall, specificity, *_ = train_and_evaluate_model(
            pipeline, param_grid, X_train, y_train, X_test, y_test
        )

        test_scores.append(test_acc)
        train_scores.append(train_acc)
        cv_scores.append(best_cv)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)

        # Save param -> accuracy for global best
        params_tuple = tuple(sorted(grid.best_params_.items()))
        param_accumulator[params_tuple].append(test_acc)

    return test_scores, train_scores, cv_scores, precisions, recalls, specificities, param_accumulator, best_params


def run_session_split(person_data, pipeline, param_grid):
    X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, "session", seed=0)
    grid, best_cv, train_acc, test_acc, best_params, precision, recall, specificity, *_ = train_and_evaluate_model(
        pipeline, param_grid, X_train, y_train, X_test, y_test
    )

    return [test_acc], [train_acc], [best_cv], [precision], [recall], [specificity], None, best_params


def save_results(results_path, name, split_type, best_params, final_metrics):
    split_label = "(S1+S2 vs S3)" if split_type == "session" else "(80/20)"
    model_with_split = f"{name} {split_label}"
    final_test, final_train, final_cv, final_precision, final_recall, final_specificity, selected_k = final_metrics

    result = pd.DataFrame([{
        "Model": model_with_split,
        "Best Parameters": str(best_params),
        "Best CV Accuracy": final_cv,
        "Train Accuracy": final_train,
        "Test Accuracy": final_test,
        "Selected k": selected_k,
        "Precision": final_precision,
        "Recall": final_recall,
        "Specificity": final_specificity
    }])

    if not os.path.exists(results_path):
        result.to_csv(results_path, index=False)
    else:
        result.to_csv(results_path, mode='a', header=False, index=False)


def run_verification(split_type, results_path):
    classifiers = get_classifiers_with_grid()

    for name, pipeline, param_grid in classifiers:
        print(f"\n=== {name} ===")
        all_test, all_train, all_cv = [], [], []
        all_precisions, all_recalls, all_specificities = [], [], []
        param_accumulator_total = defaultdict(list)
        last_best_params = None

        for person in people:
            person_data = dataset[dataset['person_id'] == person]

            if split_type == "random":
                test_scores, train_scores, cv_scores, precisions, recalls, specificities, param_accumulator, best_params = run_random_split(
                    person_data, pipeline, param_grid
                )

                # Accumulate all param sets
                for p, accs in param_accumulator.items():
                    param_accumulator_total[p].extend(accs)

            elif split_type == "session":
                test_scores, train_scores, cv_scores, precisions, recalls, specificities, _, best_params = run_session_split(
                    person_data, pipeline, param_grid
                )

            # Compute per-person average (random: over seeds, session: just one value)
            avg_test, avg_train, avg_cv, avg_precision, avg_recall, avg_specificity, _ = compute_mean(
                test_scores, train_scores, cv_scores, precisions, recalls, specificities, best_params
            )

            # Accumulate across people
            all_test.append(avg_test)
            all_train.append(avg_train)
            all_cv.append(avg_cv)
            all_precisions.append(avg_precision)
            all_recalls.append(avg_recall)
            all_specificities.append(avg_specificity)
            last_best_params = best_params

        # Global average across all people
        final_metrics = compute_mean(all_test, all_train, all_cv, all_precisions, all_recalls, all_specificities)

        if split_type == "random":
            avg_param_performance = {k: np.mean(v) for k, v in param_accumulator_total.items()}
            best_param_tuple = max(avg_param_performance.items(), key=lambda x: x[1])[0]
            best_params = dict(best_param_tuple)
        else:
            best_params = last_best_params

        save_results(results_path, name, split_type, best_params, final_metrics)


# File paths
results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_results.csv"

# Remove old results file if exists
if os.path.exists(results_file):
    os.remove(results_file)

# Run verification for random split
run_verification(split_type='random', results_path=results_file)
# Run verification for session split, save ROC curve
run_verification(split_type='session', results_path=results_file)

import os
import pandas as pd
import numpy as np
import warnings
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# disable an unexpected warning on the new pandas version
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

# Load dataset
#csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

dataset['person_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

# List that contains f0 to f82
features_cols = [f'f{i}' for i in range(83)]
# Get all the different person number
people = dataset['person_id'].unique()
NUM_TRIALS = 10

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
    }

    # K-Nearest Neighbors
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', KNeighborsClassifier())
    ])
    knn_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'clf__n_neighbors': [3, 5, 7, 9, 11],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['minkowski', 'euclidean', 'manhattan']
    }

    # Logistic Regression
    logreg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=0))
    ])
    logreg_params = {
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # NuSVC
    nusvc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', NuSVC(probability=True))
    ])
    nusvc_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'clf__nu': [0.25, 0.5, 0.75],
        'clf__kernel': ['rbf', 'poly', 'sigmoid'],
        'clf__gamma': ['scale', 'auto']
    }

    # Random Forest
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', RandomForestClassifier(random_state=0))
    ])
    rf_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'clf__n_estimators': [20, 30, 50, 100, 200],
        'clf__max_features': ['sqrt'],
        'clf__max_depth': [5, 10, 20, 30]
    }

    # SVC
    svc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', SVC(probability=True))
    ])
    svc_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['rbf', 'poly']
    }

    # MLP Classifier
    mlp_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', MLPClassifier(max_iter=3000, random_state=0))
    ])
    mlp_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
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

# Prepare data for a single person. The parameter person data contains all the samples related to the current person
def prepare_train_test_data(person_data, split_type, seed):
    if split_type == 'session':
        # S1 and S2 in train while S3 on test
        train_data = person_data[person_data['session_id'].isin(['S1', 'S2'])]
        test_data = person_data[person_data['session_id'] == 'S3']
        
        # Create impostors for the current person
        impostors = dataset[dataset['person_id'] != person_data['person_id'].iloc[0]]

        # Randomly selects impostor samples to match the number of genuine training samples
        impostors_train = impostors.sample(n=len(train_data), random_state=seed)
        # Does the same thing after removing the samples used for the train
        remaining_impostors = impostors.drop(impostors_train.index)
        impostors_test = remaining_impostors.sample(n=len(test_data), random_state=seed)

        train_combined = pd.concat([train_data, impostors_train])
        test_combined = pd.concat([test_data, impostors_test])

        X_train = train_combined[features_cols]
        y_train = np.array([1] * len(train_data) + [0] * len(impostors_train))
        X_test = test_combined[features_cols]
        y_test = np.array([1] * len(test_data) + [0] * len(impostors_test))

        return X_train, y_train, X_test, y_test

    elif split_type == 'random':
        # Create impostors for the current person
        impostors = dataset[dataset['person_id'] != person_data['person_id'].iloc[0]]
        
        # Label genuine and impostor samples
        genuine = person_data.copy()
        genuine['y'] = 1
        # Shuffle the impostors (frac=1) and reset the old row indexes
        impostors = impostors.sample(frac=1, random_state=seed).reset_index(drop=True)
        impostors['y'] = 0

        # Balance impostors to match genuine size approximately 
        num_genuine = len(genuine)
        impostors_balanced = impostors.sample(n=num_genuine, random_state=seed)


        # Combine genuine and impostor samples
        combined = pd.concat([genuine, impostors_balanced]).reset_index(drop=True)

        # Stratify on label 1 and 0 to have a balanced distribution
        train_combined, test_combined = train_test_split(
            combined,
            test_size=0.2,
            stratify=combined['y'],
            random_state=seed
        )

        X_train = train_combined[features_cols]
        y_train = train_combined['y'].values
        X_test = test_combined[features_cols]
        y_test = test_combined['y'].values

        return X_train, y_train, X_test, y_test

    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    

def train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Use the best estimator fitted by GridSearchCV
    best_model = grid.best_estimator_

    cv_accuracy = grid.best_score_

    # Compute train and test accuracy
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    # Use the best model to make prediction on the test set
    y_pred = best_model.predict(X_test)

    # Get the probability a sample belongs to class 1. Right now all classifiers have predict proba, but I add this to be flexible on future classifiers implementation
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    else:
        y_score = best_model.decision_function(X_test)

    # Get the confusion matrix 
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Compute precision, recall and specificity. Check also that denominator is greater than zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fpr, tpr, _ = roc_curve(y_test, y_score)

    # Compute EER
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    roc_auc = auc(fpr, tpr)
    
    return grid, cv_accuracy, train_accuracy, test_accuracy, grid.best_params_, precision, recall, specificity, fpr, tpr, roc_auc, eer


def compute_mean(test_scores, train_scores, cv_scores, precisions, recalls, specificities, roc_aucs, eers, best_params=None):
    avg_test = np.mean(test_scores)
    avg_train = np.mean(train_scores)
    avg_cv = np.mean(cv_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_specificity = np.mean(specificities)
    avg_auc = np.mean(roc_aucs)
    avg_eer = np.mean(eers)
    k = best_params.get('feature_selection__k', 'N/A') if best_params else 'N/A'

    return avg_test, avg_train, avg_cv, avg_precision, avg_recall, avg_specificity, avg_auc, avg_eer, k



def run_random_split(person_data, pipeline, param_grid):
    # Dictionary that maps each key to a list. The goal is to collect multiple accuracy values for each unique parameter combination
    param_accumulator = defaultdict(list)
    test_scores, train_scores, cv_scores = [], [], []
    precisions, recalls, specificities = [], [], []
    roc_aucs, eers = [], []

    for seed in range(NUM_TRIALS):
        X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, "random", seed)
        grid, best_cv, train_acc, test_acc, best_params, precision, recall, specificity, _, _, roc_auc, eer  = train_and_evaluate_model(
            pipeline, param_grid, X_train, y_train, X_test, y_test
        )

        test_scores.append(test_acc)
        train_scores.append(train_acc)
        cv_scores.append(best_cv)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        roc_aucs.append(roc_auc)
        eers.append(eer)

        # Create an hashtable with the sorted key value returned by grid.best_params_.items()
        params_tuple = tuple(sorted(grid.best_params_.items()))
        # Using the parameter tuple as the key, append the current test accuracy to the list associated with that parameter set.
        param_accumulator[params_tuple].append(test_acc)

    return test_scores, train_scores, cv_scores, precisions, recalls, specificities, roc_aucs, eers, param_accumulator, best_params


def run_session_split(person_data, pipeline, param_grid):
    X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, "session", seed=0)
    grid, best_cv, train_acc, test_acc, best_params, precision, recall, specificity, fpr, tpr, roc_auc, eer = train_and_evaluate_model(
        pipeline, param_grid, X_train, y_train, X_test, y_test
    )
    
    # Return a none because in this case the best parameters are obtained only for one iteration. No need for further computations.
    # I also return the single elements inside a list for code consistency in the run_verification function
    return [test_acc], [train_acc], [best_cv], [precision], [recall], [specificity], [roc_auc], [eer], None, best_params

def save_results(results_path, name, split_type, best_params, final_metrics):
    split_label = "(S1+S2 vs S3)" if split_type == "session" else "(80/20)"
    model_with_split = f"{name} {split_label}"
    
    # Unpack final metrics
    final_test, final_train, final_cv, final_precision, final_recall, final_specificity, final_roc_auc, final_eer, selected_k = final_metrics

    result = pd.DataFrame([{
        "Model": model_with_split,
        "Best Parameters": str(best_params),
        "Best CV Accuracy": round(final_cv, 4),
        "Train Accuracy": round(final_train, 4),
        "Test Accuracy": round(final_test, 4),
        "Selected k": selected_k,
        "Precision": round(final_precision, 4),
        "Recall": round(final_recall, 4),
        "Specificity": round(final_specificity, 4),
        "Roc Auc": round(final_roc_auc, 4),
        "EER": round(final_eer, 4)
    }])

    file_exists = os.path.exists(results_path)
    result.to_csv(results_path, mode='a', header=not file_exists, index=False)


def evaluate_with_random_split(classifiers):
    results = []

    for name, pipeline, param_grid in classifiers:
        print(f"\n=== {name} ===")
        seed_metrics = []
        param_accumulator_total = defaultdict(list)
        best_params_example = None
        
        for person in people:
            person_data = dataset[dataset['person_id'] == person]

            test_scores, train_scores, cv_scores, precisions, recalls, specificities, roc_aucs, eers, param_accumulator, best_params = run_random_split(
                person_data, pipeline, param_grid
            )

            per_person_mean = compute_mean(test_scores, train_scores, cv_scores, precisions, recalls, specificities, roc_aucs, eers, best_params)            
            seed_metrics.append(per_person_mean)

            for param, scores in param_accumulator.items():
                param_accumulator_total[param].extend(scores)

            if best_params_example is None:
                best_params_example = best_params

        final_metrics = tuple(np.mean([np.array(metric[:8]) for metric in seed_metrics], axis=0))
        selected_k = best_params_example.get('feature_selection__k', 'N/A') if best_params_example else 'N/A'
        final_metrics += (selected_k,)

        avg_param_performance = {k: np.mean(v) for k, v in param_accumulator_total.items()}
        best_param_tuple = max(avg_param_performance.items(), key=lambda x: x[1])[0]
        best_params = dict(best_param_tuple)

        results.append((name, best_params, final_metrics))

    return results


def evaluate_with_session_split(classifiers):
    results = []

    for name, pipeline, param_grid in classifiers:
        print(f"\n=== {name} ===")
        all_test, all_train, all_cv = [], [], []
        all_precisions, all_recalls, all_specificities = [], [], []
        all_auc, all_eer = [], []
        last_best_params = None

        for person in people:
            person_data = dataset[dataset['person_id'] == person]

            test_scores, train_scores, cv_scores, precisions, recalls, specificities, roc_aucs, eers, _, best_params = run_session_split(person_data, pipeline, param_grid)

            avg_test, avg_train, avg_cv, avg_precision, avg_recall, avg_specificity, avg_auc, avg_eer, k= compute_mean(
                test_scores, train_scores, cv_scores, precisions, recalls, specificities, roc_aucs, eers, best_params
            )


            all_test.append(avg_test)
            all_train.append(avg_train)
            all_cv.append(avg_cv)
            all_precisions.append(avg_precision)
            all_recalls.append(avg_recall)
            all_specificities.append(avg_specificity)
            all_auc.append(avg_auc)
            all_eer.append(avg_eer)
            last_best_params = best_params

        final_metrics = compute_mean(
            all_test, all_train, all_cv,
            all_precisions, all_recalls, all_specificities,
            all_auc, all_eer,
            best_params=last_best_params
        )
        best_params = last_best_params

        results.append((name, best_params, final_metrics))

    return results

def run_verification(split_type, results_path):
    classifiers = get_classifiers_with_grid()

    if split_type == "random":
        for name, best_params, final_metrics in evaluate_with_random_split(classifiers):
            save_results(results_path, name, split_type, best_params, final_metrics)

    elif split_type == "session":
        results = evaluate_with_session_split(classifiers)
        for name, best_params, final_metrics in results:
            save_results(results_path, name, split_type, best_params, final_metrics)

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

# File paths
results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_results_nofs.csv"
#results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_results_nofs.csv"

# Remove old results file if exists
if os.path.exists(results_file):
    os.remove(results_file)

def main():
    run_verification("random", results_file)
    run_verification("session", results_file)

if __name__ == "__main__":
    main()

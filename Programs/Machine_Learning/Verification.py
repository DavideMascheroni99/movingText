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
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif

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

#List that contains f0 to f82
features_cols = [f'f{i}' for i in range(83)]
#Get all the different person number
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

# Plot and save top features
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

    plt.figure(figsize=(12, max(6, k//2)))
    plt.barh(sorted_features, sorted_scores)
    plt.xlabel('ANOVA F-value')
    plt.title(f'Top {k} features for {model_name} ({split_name} split)')
    plt.gca().invert_yaxis()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{model_name}_{split_name}_top_features.png'))
    plt.close()


def run_verification(split_type, results_path, roc_save_path=None, feature_plot_dir=None):
    classifiers = get_classifiers_with_grid()
    roc_data = []

    for person in people:
        person_data = dataset[dataset['person_id'] == person]

        # To store metrics per model
        model_results = {}

        for name, pipeline, param_grid in classifiers:
            trial_test_accs = []
            trial_train_accs = []
            trial_cv_scores = []
            trial_precisions = []
            trial_recalls = []
            trial_specificities = []
            trial_aucs = []

            # Store feature plot and ROC data for best trial later
            best_grid = None
            best_fpr = None
            best_tpr = None
            best_auc = None
            best_test_acc = -np.inf

            for trial_seed in range(NUM_TRIALS):
                X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, split_type, trial_seed)
                grid, best_cv, train_acc, test_acc, best_params, precision, recall, specificity, fpr, tpr, roc_auc = train_and_evaluate_model(
                    pipeline, param_grid, X_train, y_train, X_test, y_test
                )

                trial_test_accs.append(test_acc)
                trial_train_accs.append(train_acc)
                trial_cv_scores.append(best_cv)
                trial_precisions.append(precision)
                trial_recalls.append(recall)
                trial_specificities.append(specificity)
                trial_aucs.append(roc_auc)

                # Keep track of best trial to plot later
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_grid = grid
                    best_fpr = fpr
                    best_tpr = tpr
                    best_auc = roc_auc
                    best_params_for_model = best_params

            # Compute average metrics over all trials for this model and person
            avg_test = np.mean(trial_test_accs)
            avg_train = np.mean(trial_train_accs)
            avg_cv = np.mean(trial_cv_scores)
            avg_precision = np.mean(trial_precisions)
            avg_recall = np.mean(trial_recalls)
            avg_specificity = np.mean(trial_specificities)
            avg_auc = np.mean(trial_aucs)

            model_results[name] = {
                "avg_test": avg_test,
                "avg_train": avg_train,
                "avg_cv": avg_cv,
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_specificity": avg_specificity,
                "avg_auc": avg_auc,
                "best_grid": best_grid,
                "best_fpr": best_fpr,
                "best_tpr": best_tpr,
                "best_auc": best_auc,
                "best_params": best_params_for_model
            }

        # Identify best model for the person by highest avg_test accuracy
        best_model_name = max(model_results, key=lambda m: model_results[m]["avg_test"])
        best_model_info = model_results[best_model_name]

        # Save results to CSV for all models
        for model_name, metrics in model_results.items():
            k = metrics["best_params"].get('feature_selection__k', 'N/A')
            split_label = "(S1+S2 vs S3)" if split_type == "session" else "(80/20)"
            model_with_split = f"{model_name} {split_label}"

            result = pd.DataFrame([{
                "Model": model_with_split,
                "Best Parameters": str(metrics["best_params"]),
                "Best CV Accuracy": metrics["avg_cv"],
                "Train Accuracy": metrics["avg_train"],
                "Test Accuracy": metrics["avg_test"],
                "Selected k": k,
                "Precision": metrics["avg_precision"],
                "Recall": metrics["avg_recall"],
                "Specificity": metrics["avg_specificity"],
                "AUC": metrics["avg_auc"]
            }])

            if not os.path.exists(results_path):
                result.to_csv(results_path, index=False)
            else:
                result.to_csv(results_path, mode='a', header=False, index=False)

        # Plot top features only for best model
        if feature_plot_dir and best_model_info["best_grid"] is not None:
            plot_top_features(best_model_info["best_grid"], X_train, best_model_name, split_type, feature_plot_dir)

        # Store ROC data only for best model (session split)
        if split_type == 'session' and best_model_info["best_fpr"] is not None:
            roc_data.append((best_model_name, best_model_info["best_fpr"], best_model_info["best_tpr"], best_model_info["best_auc"]))

    # Create the ROC curve plot for the session split
    if split_type == 'session' and roc_save_path and len(roc_data) > 0:
        plt.figure(figsize=(10, 8))
        for name, fpr, tpr, auc_score in roc_data:
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Session Split)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(roc_save_path)
        plt.close()

results_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_results.csv"
roc_output_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\roc_verification_ss.png"
feature_plot_dir = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_KBest"

if os.path.exists(results_file):
    os.remove(results_file)

run_verification(split_type='random', results_path=results_file, feature_plot_dir=feature_plot_dir)
run_verification(split_type='session', results_path=results_file, roc_save_path=roc_output_path, feature_plot_dir=feature_plot_dir)

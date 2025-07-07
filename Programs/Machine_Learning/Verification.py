import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

dataset['person_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

features_cols = [f'f{i}' for i in range(83)]
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
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [StandardScaler(), MinMaxScaler()],
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
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        ('clf', MLPClassifier(max_iter=2000, random_state=0))
    ])
    mlp_params = {
        'scaler': [MinMaxScaler(), StandardScaler()],
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
def prepare_train_test_data(person_data, split_type, trial):
    if split_type == 'session':
        train_data = person_data[person_data['session_id'].isin(['S1', 'S2'])]
        test_data = person_data[person_data['session_id'] == 'S3']
    elif split_type == 'random':
        train_data, test_data = train_test_split(person_data, test_size=0.2, stratify=person_data['session_id'], random_state=trial)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    impostors = dataset[dataset['person_id'] != person_data['person_id'].iloc[0]]
    impostors_shuffled = impostors.sample(frac=1, random_state=trial).reset_index(drop=True)
    impostor_train = impostors_shuffled.iloc[:len(train_data)]
    impostor_test = impostors_shuffled.iloc[len(train_data):len(train_data)+len(test_data)]

    train_combined = pd.concat([train_data, impostor_train])
    test_combined = pd.concat([test_data, impostor_test])

    X_train = train_combined[features_cols]
    X_test = test_combined[features_cols]
    y_train = np.array([1] * len(train_data) + [0] * len(impostor_train))
    y_test = np.array([1] * len(test_data) + [0] * len(impostor_test))

    return X_train, y_train, X_test, y_test

# Train and evaluate a model
def train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    y_score = grid.best_estimator_.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    return grid, grid.best_score_, grid.best_estimator_.score(X_train, y_train), grid.best_estimator_.score(X_test, y_test), grid.best_params_, precision, recall, specificity, fpr, tpr, roc_auc

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

    plt.figure(figsize=(12, max(6, k * 0.3)))
    plt.barh(sorted_features[:k][::-1], sorted_scores[:k][::-1], color='skyblue')
    plt.xlabel('ANOVA F-score')
    plt.title(f'Top {k} Features for {model_name} ({split_name} split)')
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)
    plt.yticks(fontsize=9)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_{split_name.replace(' ', '_').replace('/', '-')}_top_features.png")
    plt.savefig(save_path)
    plt.close()

# Run full evaluation and save results including k and plots
def run_verification(split_type, results_path, roc_save_path=None, feature_plot_dir=None):
    classifiers = get_classifiers_with_grid()
    roc_data = []

    for name, pipeline, param_grid in classifiers:
        print(f"\n=== {name} ===")
        person_accuracies, person_train_scores, person_cv_scores = [], [], []

        for person in people:
            person_data = dataset[dataset['person_id'] == person]
            X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, split_type, 0)
            grid, best_cv, train_acc, test_acc, best_params, precision, recall, specificity, fpr, tpr, roc_auc = train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test)

            person_accuracies.append(test_acc)
            person_train_scores.append(train_acc)
            person_cv_scores.append(best_cv)

            if feature_plot_dir:
                plot_top_features(grid, X_train, name, split_type, feature_plot_dir)

            if split_type == 'session':
                roc_data.append((name, fpr, tpr, roc_auc))

        # Save average results
        avg_test = np.mean(person_accuracies)
        avg_train = np.mean(person_train_scores)
        avg_cv = np.mean(person_cv_scores)
        k = best_params.get('feature_selection__k', 'N/A')

        if results_path:
            split_label = "(S1+S2 vs S3)" if split_type == "session" else "(80/20)"
            model_with_split = f"{name} {split_label}"
            result = pd.DataFrame([{
                "Model": model_with_split,
                "Best Parameters": str(best_params),
                "Best CV Accuracy": avg_cv,
                "Train Accuracy": avg_train,
                "Test Accuracy": avg_test,
                "Selected k": k
            }])

            if not os.path.exists(results_path):
                result.to_csv(results_path, index=False)
            else:
                result.to_csv(results_path, mode='a', header=False, index=False)

    if split_type == 'session' and roc_save_path:
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

# Paths
results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_results.csv"
roc_output_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\roc_verification_ss.png"
feature_plot_dir = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Feature_Plots"

if os.path.exists(results_file):
    os.remove(results_file)

run_verification(split_type='random', results_path=results_file, feature_plot_dir=feature_plot_dir)
run_verification(split_type='session', results_path=results_file, roc_save_path=roc_output_path, feature_plot_dir=feature_plot_dir)

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

#disable an unexpected warning on the new pandas version
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
        ('clf', GaussianNB())
    ])
    nb_params = {
        'scaler': [MinMaxScaler(), StandardScaler()]
    }

    # K-Nearest Neighbors
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', KNeighborsClassifier())
    ])
    knn_params = {
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [StandardScaler(), MinMaxScaler()],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # NuSVC
    nusvc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', NuSVC(probability=True))
    ])
    nusvc_params = {
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [MinMaxScaler(), StandardScaler()],
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
        'scaler': [MinMaxScaler(), StandardScaler()],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['rbf', 'poly']
    }

    # MLP Classifier
    mlp_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', MLPClassifier(max_iter=2000, random_state=0))
    ])
    mlp_params = {
        'scaler': [MinMaxScaler(), StandardScaler()],
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

def prepare_train_test_data(person_data, split_type, trial):
    #S1, S2 in the train set while S3 in the test set
    if split_type == 'session':
        train_data = person_data[person_data['session_id'].isin(['S1', 'S2'])]
        test_data = person_data[person_data['session_id'] == 'S3']
    #Random split (80-20) with stratification
    elif split_type == 'random':
        #Make sure to obtain a fair distribution of sessions after the split
        train_data, test_data = train_test_split(person_data, test_size=0.2, stratify=person_data['session_id'], random_state=trial)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    #Get a subset of persons different from the current one. These are used to test the system
    impostors = dataset[dataset['person_id'] != person_data['person_id'].iloc[0]]
    #Shuffle and sample to get every iteration different selections
    impostors_shuffled = impostors.sample(frac=1, random_state=trial).reset_index(drop=True)

    #For Train take the same number of impostors as the real persons
    impostor_train = impostors_shuffled.iloc[:len(train_data)]
    #Insert in the test the same number of impostors, but starting from the next row to avoid data leakage
    impostor_test = impostors_shuffled.iloc[len(train_data):len(train_data) + len(test_data)]

    #Combine genuine and impostor samples
    train_combined = pd.concat([train_data, impostor_train])
    test_combined = pd.concat([test_data, impostor_test])

    #Extract only the features
    X_train = train_combined[features_cols]
    X_test = test_combined[features_cols]

    #I label with 1 the correct data and with 0 the impostors
    y_train = np.array([1] * len(train_data) + [0] * len(impostor_train))
    y_test = np.array([1] * len(test_data) + [0] * len(impostor_test))

    return X_train, y_train, X_test, y_test

def train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test):
    #Train model with GridSearchCV
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    y_score = grid.best_estimator_.predict_proba(X_test)[:, 1]

    best_params = grid.best_params_
    best_cv_score = grid.best_score_
    train_score = grid.best_estimator_.score(X_train, y_train)
    test_score = grid.best_estimator_.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    return test_score, train_score, best_cv_score, best_params, precision, recall, specificity, fpr, tpr, roc_auc

def evaluate_person_model(person, name, pipeline, param_grid, split_type):
    accuracy_list = []
    train_score_list = []
    cv_score_list = []

    #Create the subset containing the current person data 
    person_data = dataset[dataset['person_id'] == person]

    for trial in range(NUM_TRIALS):
        X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, split_type, trial)

        test_acc, train_acc, best_cv_acc, best_params, _, _, _, _, _, _ = train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test)

        accuracy_list.append(test_acc)
        train_score_list.append(train_acc)
        cv_score_list.append(best_cv_acc)

    avg_test = np.mean(accuracy_list)
    avg_train = np.mean(train_score_list)
    avg_cv = np.mean(cv_score_list)

    print(f"Person {person} → CV: {avg_cv:.3f}, Train: {avg_train:.3f}, Test: {avg_test:.3f}")

    return avg_test, avg_train, avg_cv, best_params

def run_verification(split_type, results_path, roc_save_path=None):
    classifiers = get_classifiers_with_grid()
    roc_data = []

    for name, pipeline, param_grid in classifiers:
        print(f"\n=== {name} ===")
        person_accuracies = []
        person_cv_scores = []
        person_train_scores = []
        best_params_for_model = None

        for person in people:
            person_data = dataset[dataset['person_id'] == person]
            X_train, y_train, X_test, y_test = prepare_train_test_data(person_data, split_type, 0)

            test_acc, train_acc, best_cv_acc, best_params, precision, recall, specificity, fpr, tpr, roc_auc = train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test)

            person_accuracies.append(test_acc)
            person_train_scores.append(train_acc)
            person_cv_scores.append(best_cv_acc)
            best_params_for_model = best_params

            if split_type == 'session':
                roc_data.append((name, fpr, tpr, roc_auc))

        #Print and save overall result
        overall_acc = np.mean(person_accuracies) 
        overall_train = np.mean(person_train_scores) 
        overall_cv = np.mean(person_cv_scores) 

        print(f"{name} → Overall Accuracy: {overall_acc:.3f}")

        if results_path:
            # Format split strategy nicely
            split_label = "(S1+S2 vs S3)" if split_type == "session" else "(80/20)"

            # Combine model name and split info
            model_with_split = f"{name} {split_label}"

            # Save the results
            result = pd.DataFrame([{
                "Model": model_with_split,
                "Best Parameters": str(best_params_for_model),
                "Best CV Accuracy": overall_cv,
                "Train Accuracy": overall_train,
                "Test Accuracy": overall_acc
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

results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_results.csv"
roc_output_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\roc_verification_ss.png"

if os.path.exists(results_file):
    os.remove(results_file)

#Run the verification
run_verification(split_type='random', results_path=results_file)
run_verification(split_type='session', results_path=results_file, roc_save_path=roc_output_path)

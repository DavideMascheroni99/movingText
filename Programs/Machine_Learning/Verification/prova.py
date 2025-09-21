import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from collections import defaultdict

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

# List that contains f0 to f82
features_cols = [f'f{i}' for i in range(83)]
# Get all the different person number
people = dataset['person_id'].unique()
NUM_TRIALS = 10

def get_classifiers_with_grid():
    # SVC
    svc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', SVC())
    ])
    svc_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()],
        'feature_selection__k': [30, 40, 50, 60, 70],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['rbf', 'poly']
    }

    # Return a list of tuples (name, pipeline, param_grid)
    return [("SVC", svc_pipeline, svc_params)]

def prepare_train_test_data(person_data, seed=0):
    # S1 and S2 in train while S3 on test
    train_data = person_data[person_data['session_id'].isin(['S1', 'S2'])]
    test_data = person_data[person_data['session_id'] == 'S3']
    
    # Create impostors for the current person
    impostors = dataset[dataset['person_id'] != person_data['person_id'].iloc[0]]
    impostors_train = impostors.sample(n=len(train_data), random_state=seed)
    impostors_test = impostors.drop(impostors_train.index).sample(n=len(test_data), random_state=seed)

    train_combined = pd.concat([train_data, impostors_train])
    test_combined = pd.concat([test_data, impostors_test])

    X_train = train_combined[features_cols]
    y_train = np.array([1] * len(train_data) + [0] * len(impostors_train))
    X_test = test_combined[features_cols]
    y_test = np.array([1] * len(test_data) + [0] * len(impostors_test))

    return X_train, y_train, X_test, y_test

def train_and_evaluate_model(pipeline, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    y_pred = best_model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return grid.best_params_, train_accuracy, test_accuracy, precision, recall, specificity

def save_results(results_path, name, best_params, final_metrics, tester):
    model_with_split = f"{name} (S1+S2 vs S3)"
    
    final_test, final_train, final_precision, final_recall, final_specificity, selected_k = final_metrics

    result = pd.DataFrame([{
        "Model": model_with_split,
        "Tester": tester,
        "Best Parameters": str(best_params),
        "Train Accuracy": round(final_train,4),
        "Test Accuracy": round(final_test,4),
        "Selected k": selected_k,
        "Precision": round(final_precision,4),
        "Recall": round(final_recall,4),
        "Specificity": round(final_specificity,4)
    }])

    file_exists = os.path.exists(results_path)
    result.to_csv(results_path, mode='a', header=not file_exists, index=False)

def run_verification(results_path):
    classifiers = get_classifiers_with_grid()
    for name, pipeline, param_grid in classifiers:
        for person in people:
            person_data = dataset[dataset['person_id']==person]

            X_train, y_train, X_test, y_test = prepare_train_test_data(person_data)
            best_params, train_acc, test_acc, precision, recall, specificity = train_and_evaluate_model(
                pipeline, param_grid, X_train, y_train, X_test, y_test
            )
            
            selected_k = best_params.get('feature_selection__k', 'N/A')
            final_metrics = (test_acc, train_acc, precision, recall, specificity, selected_k)
            save_results(results_path, name, best_params, final_metrics, tester=person)

def main():
    results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_results\Verification_prova.csv"

    if os.path.exists(results_file):
        os.remove(results_file)

    run_verification(results_file)
    print(f"Results saved in {results_file}")

if __name__ == "__main__":
    main()

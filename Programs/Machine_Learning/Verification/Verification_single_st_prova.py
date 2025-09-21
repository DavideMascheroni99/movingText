import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.base import clone

# -----------------------------
# Helper functions
# -----------------------------
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def write_results(results_df, results_path):
    results_df.to_csv(results_path, index=False)

def load_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
    dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
    dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])
    return dataset

def get_feature_columns():
    return [f'f{i}' for i in range(83)]

def prepare_train_test_data(dataset, person_data, seed, features_cols):
    tester_id = person_data['tester_id'].iloc[0]
    train_genuine = person_data[person_data['session_id'].isin(['S1', 'S2'])]
    test_genuine = person_data[person_data['session_id'] == 'S3']

    impostors_train_pool = dataset[(dataset['tester_id'] != tester_id) & (dataset['session_id'].isin(['S1', 'S2']))]
    impostors_test_pool = dataset[(dataset['tester_id'] != tester_id) & (dataset['session_id'] == 'S3')]

    impostors_train = impostors_train_pool.sample(n=len(train_genuine), random_state=seed)
    impostors_test = impostors_test_pool.sample(n=len(test_genuine), random_state=seed)

    X_train = pd.concat([train_genuine[features_cols], impostors_train[features_cols]])
    y_train = np.array([1]*len(train_genuine) + [0]*len(impostors_train))
    X_test = pd.concat([test_genuine[features_cols], impostors_test[features_cols]])
    y_test = np.array([1]*len(test_genuine) + [0]*len(impostors_test))

    return X_train, y_train, X_test, y_test

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    test_acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return test_acc, prec, rec, spec

# -----------------------------
# Paths and constants
# -----------------------------
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
results_path_animation = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results\Verification_per_animation.csv"
results_path_tester = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results\Verification_per_tester.csv"

delete_file_if_exists(results_path_animation)
delete_file_if_exists(results_path_tester)

num_seed = 20

# -----------------------------
# Main execution
# -----------------------------
dataset = load_dataset(csv_path)
features_cols = get_feature_columns()

# SVC pipeline
svc_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('svc', SVC(probability=True))
])

# -----------------------------
# Metrics per tester per animation
# -----------------------------
animation_results = []

for tester_id in sorted(dataset['tester_id'].unique()):
    tester_data = dataset[dataset['tester_id'] == tester_id]

    for animation in sorted(tester_data['anim_name'].unique()):
        person_data = tester_data[tester_data['anim_name'] == animation]
        metrics_list = []

        for seed in range(20, num_seed + 20):
            X_train, y_train, X_test, y_test = prepare_train_test_data(dataset, person_data, seed, features_cols)
            model = clone(svc_pipeline)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc, prec, rec, spec = evaluate_model(model, X_test, y_test)
            metrics_list.append([train_acc, test_acc, prec, rec, spec])

        avg_metrics = np.mean(metrics_list, axis=0)
        animation_results.append([tester_id, animation] + list(avg_metrics))

animation_df = pd.DataFrame(animation_results, columns=['tester_id', 'Animation', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'Specificity'])
animation_df.sort_values(by=['tester_id', 'Animation'], inplace=True)
write_results(animation_df, results_path_animation)
print(f"Results per tester per animation saved to {results_path_animation}")

# -----------------------------
# Metrics per tester (average over animations)
# -----------------------------
tester_results = []

for tester_id, group in animation_df.groupby('tester_id'):
    avg_metrics = group[['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'Specificity']].mean().values
    tester_results.append([tester_id] + list(avg_metrics))

tester_df = pd.DataFrame(tester_results, columns=['tester_id', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'Specificity'])
tester_df.sort_values(by='tester_id', inplace=True)
write_results(tester_df, results_path_tester)
print(f"Average results per tester saved to {results_path_tester}")

# -----------------------------
# Optional: testers with average test accuracy > threshold
# -----------------------------
threshold = 0.96
qualified_testers = tester_df[tester_df['Test Accuracy'] > threshold]['tester_id'].tolist()
print(f"Testers with average test accuracy > {threshold}: {qualified_testers}")

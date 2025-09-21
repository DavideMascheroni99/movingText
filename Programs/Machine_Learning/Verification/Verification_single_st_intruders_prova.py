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

def get_open_set_split(dataset, n_known=24, split_seed=0):
    testers = sorted(dataset['tester_id'].unique())
    rng = np.random.default_rng(split_seed)  # <--- use this seed to select known testers
    train_ids = rng.choice(testers, size=n_known, replace=False)
    test_ids = [t for t in testers if t not in train_ids]
    return list(train_ids), list(test_ids)

def prepare_open_set_person_data(dataset, person_data, train_ids, test_ids, seed, features_cols):
    tester_id = person_data['tester_id'].iloc[0]

    # Train set
    train_genuine = person_data[person_data['session_id'].isin(['S1', 'S2'])]
    impostors_train_pool = dataset[
        (dataset['tester_id'].isin(train_ids)) &
        (dataset['tester_id'] != tester_id) &
        (dataset['session_id'].isin(['S1', 'S2']))
    ]
    impostors_train = impostors_train_pool.sample(n=len(train_genuine), random_state=seed, replace=False)

    X_train = pd.concat([train_genuine[features_cols], impostors_train[features_cols]], ignore_index=True)
    y_train = np.array([1]*len(train_genuine) + [0]*len(impostors_train))

    # Test set
    test_genuine = person_data[person_data['session_id'] == 'S3']
    impostors_test_pool = dataset[
        (dataset['tester_id'].isin(test_ids)) &
        (dataset['session_id'] == 'S3')
    ]
    impostors_test = impostors_test_pool.sample(n=len(test_genuine), random_state=seed, replace=False)

    X_test = pd.concat([test_genuine[features_cols], impostors_test[features_cols]], ignore_index=True)
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
results_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_intruders_results\Verification_open_set_svc.csv"
delete_file_if_exists(results_path)

dataset = load_dataset(csv_path)
features_cols = get_feature_columns()

num_seed = 20
# CHANGE split_seed HERE to pick different known persons / intruders
split_seed = 42   # <-- you can change this number
train_ids, test_ids = get_open_set_split(dataset, n_known=24, split_seed=split_seed)

results_list = []

# SVC pipeline only
svc_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler()),
    ('svc', SVC(probability=True))
])

# Iterate per known tester and per animation
for tester_id in sorted(train_ids):
    tester_data = dataset[dataset['tester_id'] == tester_id]
    for animation in sorted(tester_data['anim_name'].unique()):
        person_data = tester_data[tester_data['anim_name'] == animation]
        metrics_list = []

        for seed in range(num_seed):
            X_train, y_train, X_test, y_test = prepare_open_set_person_data(
                dataset, person_data, train_ids, test_ids, seed, features_cols
            )
            model = clone(svc_pipeline)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc, prec, rec, spec = evaluate_model(model, X_test, y_test)
            metrics_list.append([train_acc, test_acc, prec, rec, spec])

        avg_metrics = np.mean(metrics_list, axis=0)
        results_list.append([tester_id, animation] + list(avg_metrics))

# Save results CSV ordered by tester and animation
results_df = pd.DataFrame(results_list, columns=['tester_id', 'Animation', 'Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'Specificity'])
results_df.sort_values(by=['tester_id', 'Animation'], inplace=True)
write_results(results_df, results_path)

print(f"Results per tester per animation saved to {results_path}")

# Print testers with test accuracy > 0.98 for all animations
threshold = 0.96
qualified_testers = [tid for tid, g in results_df.groupby('tester_id') if (g['Test Accuracy'] > threshold).all()]
print(f"Testers with test accuracy > {threshold} for all animations: {qualified_testers}")

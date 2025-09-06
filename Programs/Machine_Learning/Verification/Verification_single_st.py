import os
import pandas as pd
import numpy as np
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


# Delete the file if already exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Create the csv if doesn't exists, otherwhise append to the existing one
def append_to_csv(df, file_path):
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

'''CONSTANTS'''

num_seed = 20

#csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"

#results_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results\Verification_single_results_st.csv"
results_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results\Verification_single_results_st.csv"
delete_file_if_exists(results_path)

#best_k_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_st.csv"
best_k_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_st.csv"
delete_file_if_exists(best_k_file)


# ---------------------------
# Load Dataset
# ---------------------------
def load_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
    dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
    dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])
    return dataset

def get_feature_columns():
    return [f'f{i}' for i in range(83)]

def load_best_k_features(csv_path):
    df = pd.read_csv(csv_path)
    best_k_features = {}

    for _, row in df.iterrows():
        model = row['Model'].strip()
        animation = row['Animation'].strip()
        k = int(row['Best K'])
        feature = row['Feature'].strip()

        if model not in best_k_features:
            best_k_features[model] = {}
        if animation not in best_k_features[model]:
            best_k_features[model][animation] = {'k': k, 'features': []}

        best_k_features[model][animation]['features'].append(feature)

    return best_k_features


# ---------------------------
# Train/Test Split
# ---------------------------
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

# ---------------------------
# Evaluation
# ---------------------------
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    return fpr[eer_index]

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    test_acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spec = tn / (tn + fp)
    roc_auc = roc_auc_score(y, y_score)
    eer = compute_eer(y, y_score)

    return test_acc, prec, rec, spec, roc_auc, eer

# ---------------------------
# Write Results
# ---------------------------
def write_results(model, best_params, train_acc, test_acc, metrics, results_path, anim_name):
    precision, recall, spec, roc_auc, eer = metrics
    row = {
        'Model': model,
        'Animation': anim_name,
        'Best Parameters': best_params,
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

# ---------------------------
# Classifier Pipelines
# ---------------------------
def get_classifiers_with_best_params():
    return [
        (
            "Naive Bayes",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),  # will be overwritten
                ('clf', GaussianNB())
            ])
        ),
        (
            "KNN",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),  # will be overwritten
                ('clf', KNeighborsClassifier())
            ])
        ),
        # Add other classifiers similarly
    ]

# ---------------------------
# Best Parameters Dictionary
# ---------------------------
best_params_all = {
    "Naive Bayes": {
        "VB_SL_LIT": {
            'scaler': MinMaxScaler()
        },
        "Other_Animation": {
            'scaler': StandardScaler(),
        }
    },
    "KNN": {
        "VB_SL_LIT": {
            'scaler': MinMaxScaler(),
            'clf__n_neighbors': 5
        }
    }
    # Add other classifiers and their per-animation parameters here
}

# ---------------------------
# Train & Evaluate
# ---------------------------
def train_and_evaluate(dataset, animation, clf_name, clf_pipeline, features_cols, best_params, num_seed, results_path):
    for seed in range(num_seed):
        all_X_train, all_y_train, all_X_test, all_y_test = [], [], [], []
        for person in dataset['tester_id'].unique():
            person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == animation)]
            X_train, y_train, X_test, y_test = prepare_train_test_data(dataset, person_data, seed, features_cols)
            all_X_train.append(X_train)
            all_y_train.append(y_train)
            all_X_test.append(X_test)
            all_y_test.append(y_test)

        clf_pipeline.set_params(**best_params)
        clf_pipeline.fit(pd.concat(all_X_train), np.concatenate(all_y_train))

        test_acc, prec, rec, spec, roc_auc, eer = evaluate_model(clf_pipeline, pd.concat(all_X_test), np.concatenate(all_y_test))
        train_acc = clf_pipeline.score(pd.concat(all_X_train), np.concatenate(all_y_train))

        write_results(clf_name, best_params, train_acc, test_acc, (prec, rec, spec, roc_auc, eer), results_path, animation)

# ---------------------------
# Main Execution
# ---------------------------

dataset = load_dataset(csv_path)
best_k_features = load_best_k_features(best_k_file)

for clf_name, clf_pipeline in get_classifiers_with_best_params():
    for animation in dataset['anim_name'].unique():
        if clf_name not in best_k_features or animation not in best_k_features[clf_name]:
            print(f"No feature selection info for {clf_name} and animation {animation}. Skipping.")
            continue

        k_info = best_k_features[clf_name][animation]
        selected_features = k_info['features'][:k_info['k']]  # ensure only top k

        train_and_evaluate(
            dataset=dataset,
            animation=animation,
            clf_name=clf_name,
            clf_pipeline=clf_pipeline,
            features_cols=selected_features,
            best_params=best_params_all.get(clf_name, {}).get(animation, {}),
            num_seed=num_seed,
            results_path=results_path
        )


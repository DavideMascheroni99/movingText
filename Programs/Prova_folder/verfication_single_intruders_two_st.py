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
from sklearn.base import clone 
import ast
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Delete the file if already exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Create the csv if doesn't exists, otherwise append to the existing one
def append_to_csv(df, file_path):
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

'''CONSTANTS'''

# seed used to make the split between persons and impostors reproducible
RANDOM_SEED = 0 
# Number of iterations per model
num_seed = 20
roc_data_dict = {}

csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"
#csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"

results_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_single_intruders_results\Verification_single_intruders_results_st.csv"
#results_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_intruders_results\Verification_single_intruders_results_st.csv"
delete_file_if_exists(results_path)

best_k_file = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_st.csv"
#best_k_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_st.csv"

best_params_file_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"
#best_params_file_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"

roc_save_path =  r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_single_intruders_results\best_animation_roc_curves_st.png"
#roc_save_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_intruders_results\best_animation_roc_curves_st.png"
delete_file_if_exists(roc_save_path)

def load_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
    dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
    dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])
    return dataset

def get_feature_columns():
    return [f'f{i}' for i in range(83)]

# Genarate the persons and the impostors ids
def get_open_set_split(dataset, n_known=24, seed=RANDOM_SEED):
    testers = sorted(dataset['tester_id'].unique())
    rng = np.random.default_rng(seed)
    train_ids = rng.choice(testers, size=n_known, replace=False)
    test_ids = [t for t in testers if t not in train_ids]
    return list(train_ids), list(test_ids)


# Read from the identification csv the top k selected features
def load_best_k_features(csv_path):
    df = pd.read_csv(csv_path)
    # Keep only rows with "(S1+S2 vs S3)"
    df = df[df['Model'].str.contains(r'\(S1\+S2 vs S3\)')]

    best_k_features = {}
    for _, row in df.iterrows():
        model = row['Model'].split("(")[0].strip() 
        animation = row['Animation'].strip()
        k = int(row['Best K'])
        feature = row['Feature'].strip()

        if model not in best_k_features:
            best_k_features[model] = {}
        if animation not in best_k_features[model]:
            best_k_features[model][animation] = {'k': k, 'features': []}

        best_k_features[model][animation]['features'].append(feature)

    return best_k_features


# Map strings to the actual sklearn scaler objects
scaler_mapping = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

# Parse the parameter in a way that can be used in the pipeline
def parse_best_params(params_str):
    # Extract the scaler using regex 
    scaler_match = re.search(r"scaler': (\w+)\(\)", params_str)
    scaler_name = None
    if scaler_match:
        scaler_name = scaler_match.group(1)
        # Replace the scaler call with just its name as a string
        params_str = re.sub(rf"{scaler_name}\(\)", f"'{scaler_name}'", params_str)

    try:
        params_dict = ast.literal_eval(params_str)
        if scaler_name and scaler_name in scaler_mapping:
            params_dict['scaler'] = scaler_mapping[scaler_name]
        return params_dict
    except Exception as e:
        print(f"Error parsing params: {params_str} -> {e}")
        return None


# Read from the previous identification file the best parameters 
def load_best_params_from_file(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Model'].str.contains(r'\(S1\+S2 vs S3\)')]

    best_params = {}
    for _, row in df.iterrows():
        model_name = row['Model'].split("(")[0].strip()
        animation = row['Animation'].strip()
        params_str = row['Best Parameters']

        params_dict = parse_best_params(params_str)

        if model_name not in best_params:
            best_params[model_name] = {}
        best_params[model_name][animation] = params_dict

    return best_params


def prepare_open_set_person_data(dataset, person_data, train_ids, test_ids, seed, features_cols):
    tester_id = person_data['tester_id'].iloc[0]

    # Train set
    train_genuine = person_data[person_data['session_id'].isin(['S1', 'S2'])]
    impostors_train_pool = dataset[
        (dataset['tester_id'].isin(train_ids)) &
        (dataset['tester_id'] != tester_id) &
        (dataset['session_id'].isin(['S1', 'S2']))
    ]
    impostors_train = impostors_train_pool.sample(
        n=len(train_genuine), random_state=seed, replace=False
    )

    X_train = pd.concat([train_genuine[features_cols], impostors_train[features_cols]], ignore_index=True)
    y_train = np.array([1]*len(train_genuine) + [0]*len(impostors_train))

    # Test set
    test_genuine = person_data[person_data['session_id'] == 'S3']
    impostors_test_pool = dataset[
        (dataset['tester_id'].isin(test_ids)) &
        (dataset['session_id'] == 'S3')
    ]
    # Balance test impostors to have same number as genuine
    impostors_test = impostors_test_pool.sample(
        n=len(test_genuine), random_state=seed, replace=False
    )

    X_test = pd.concat([test_genuine[features_cols], impostors_test[features_cols]], ignore_index=True)
    y_test = np.array([1]*len(test_genuine) + [0]*len(impostors_test))

    return X_train, y_train, X_test, y_test



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
    prec = precision_score(y, y_pred, zero_division=0)  
    rec = recall_score(y, y_pred, zero_division=0)      
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    roc_auc = roc_auc_score(y, y_score) if len(np.unique(y)) > 1 else 0
    eer = compute_eer(y, y_score) if len(np.unique(y)) > 1 else 0

    return test_acc, prec, rec, spec, roc_auc, eer


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
    append_to_csv(df, results_path)


def get_classifiers():
    return [
        (
            "Naive Bayes",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('nb', GaussianNB())
            ])
        ),
        (
            "KNN",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('knn', KNeighborsClassifier())
            ])
        ),
        (
            "Logistic Regression",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('logreg', LogisticRegression(max_iter=1000, random_state=0))
            ])
        ),
        (
            "NuSVC",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('nusvc', NuSVC())
            ])
        ),
        (
            "Random Forest",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('rf', RandomForestClassifier(random_state=0))
            ])
        ),
        (
            "SVC",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('svc', SVC())
            ])
        ),
        (
            "MLP",
            Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler()),
                ('mlp', MLPClassifier(max_iter=4000, random_state=0))
            ])
        )
    ]



def update_roc_data(clf_name, animation, y_true, y_score, eer):
  
    if clf_name not in roc_data_dict or eer < roc_data_dict[clf_name]['eer']:
        roc_data_dict[clf_name] = {
            'animation': animation,
            'y_true': np.array(y_true),
            'y_score': np.array(y_score),
            'auc': roc_auc_score(y_true, y_score),
            'eer': eer
        }

# Train classifier, evaluate metrics, and collect ROC data for the best animation per classifier
def train_and_evaluate(dataset, animation, clf_name, clf_pipeline, features_cols, best_params, train_ids, test_ids, num_seed, results_path):

    all_metrics = []
    all_y_true = []
    all_y_score = []

    for seed in range(num_seed):
        seed_metrics = []
        seed_y_true = []
        seed_y_score = []

        # Train for each person separately
        for person in train_ids:
            person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == animation)]

            X_train, y_train, X_test, y_test = prepare_open_set_person_data(dataset, person_data, train_ids, test_ids, seed, features_cols)

            model_params = {k: v for k, v in best_params.items() if not k.startswith("feature_selection")}
            model = clone(clf_pipeline).set_params(**model_params)

            model.fit(X_train, y_train)

            test_acc, prec, rec, spec, roc_auc, eer = evaluate_model(model, X_test, y_test)
            seed_metrics.append([test_acc, prec, rec, spec, roc_auc, eer])

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            seed_y_true.extend(y_test)
            seed_y_score.extend(y_score)

        # Average over all people for this seed
        mean_metrics = np.mean(seed_metrics, axis=0)
        all_metrics.append(mean_metrics)
        all_y_true.append(seed_y_true)
        all_y_score.append(seed_y_score)

    # Average results across seeds
    mean_metrics_across_seeds = np.mean(all_metrics, axis=0)
    test_acc, prec, rec, spec, roc_auc, eer = mean_metrics_across_seeds

    # Combine y_true and y_score from all seeds for ROC
    combined_y_true = np.concatenate(all_y_true)
    combined_y_score = np.concatenate(all_y_score)

    write_results(clf_name, best_params, np.nan, test_acc, (prec, rec, spec, roc_auc, eer), results_path, animation)

    return {
        'animation': animation,
        'y_true': combined_y_true,
        'y_score': combined_y_score,
        'auc': roc_auc_score(combined_y_true, combined_y_score),
        'eer': eer
    }



# Save the roc plot as a png
def save_roc_curves(roc_data_dict, save_path):

    plt.figure(figsize=(12, 9), dpi=300)

    colors = plt.colormaps['tab10'].resampled(len(roc_data_dict))
    line_styles = ['-', '--', '-.', ':']

    for i, (animation, data) in enumerate(roc_data_dict.items()):
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_score'])
        plt.plot(fpr, tpr,
                 color=colors(i),
                 linestyle=line_styles[i % len(line_styles)],
                 lw=2,
                 label=f"{animation} ({data['classifier']}), AUC={data['auc']:.2f}")

    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Verification: ROC curves (best classifier per animation)')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


'''EXECUTION'''

dataset = load_dataset(csv_path)
best_k_features = load_best_k_features(best_k_file)
best_params_all = load_best_params_from_file(best_params_file_path)

roc_data_dict = {}

# Split the testers
train_ids, test_ids = get_open_set_split(dataset, n_known=24, seed=RANDOM_SEED)

for animation in dataset['anim_name'].unique():
    best_roc_info = None
    best_clf_name = None

    for clf_name, clf_pipeline in get_classifiers():
        k_info = best_k_features[clf_name][animation]
        selected_features = k_info['features'][:k_info['k']]

        roc_info = train_and_evaluate(
            dataset=dataset,
            animation=animation,
            clf_name=clf_name,
            clf_pipeline=clf_pipeline,
            features_cols=selected_features,
            best_params=best_params_all.get(clf_name, {}).get(animation, {}),
            train_ids=train_ids,
            test_ids=test_ids,
            num_seed=num_seed,
            results_path=results_path
        )

        if best_roc_info is None or roc_info['eer'] < best_roc_info['eer']:
            best_roc_info = roc_info
            best_clf_name = clf_name

    roc_data_dict[animation] = {'classifier': best_clf_name, **best_roc_info}

save_roc_curves(roc_data_dict, roc_save_path)





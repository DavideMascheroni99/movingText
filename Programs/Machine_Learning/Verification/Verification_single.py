import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve
import warnings

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples."
)

'''LOAD DATASET'''
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

# Extract animation name, tester id and session id
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])

features_cols = [f'f{i}' for i in range(83)]
animation_names = dataset['anim_name'].unique()
people = dataset['tester_id'].unique()

'''PIPELINE DEFINITION'''
def get_nb_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB())
    ])
    param_grid = {'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()]}
    return pipeline, param_grid

model_list = [
    ("Naive Bayes", get_nb_pipeline),
]

'''Prepare train and test data for a single person using session split only'''
def prepare_train_test_data(person_data, seed):
    tester_id = person_data['tester_id'].iloc[0]

    # Genuine samples
    train_genuine = person_data[person_data['session_id'].isin(['S1', 'S2'])]
    test_genuine = person_data[person_data['session_id'] == 'S3']

    # Impostor pools: same sessions, other testers
    impostors_train_pool = dataset[
        (dataset['tester_id'] != tester_id) &
        (dataset['session_id'].isin(['S1', 'S2']))
    ]
    impostors_test_pool = dataset[
        (dataset['tester_id'] != tester_id) &
        (dataset['session_id'] == 'S3')
    ]

    # Sample impostors with same number as genuine
    impostors_train = impostors_train_pool.sample(
        n=len(train_genuine), random_state=seed,
        replace=len(impostors_train_pool) < len(train_genuine)
    )
    impostors_test = impostors_test_pool.sample(
        n=len(test_genuine), random_state=seed,
        replace=len(impostors_test_pool) < len(test_genuine)
    )

    # Combine genuine + impostors
    X_train = pd.concat([train_genuine[features_cols], impostors_train[features_cols]])
    y_train = np.array([1] * len(train_genuine) + [0] * len(impostors_train))
    X_test = pd.concat([test_genuine[features_cols], impostors_test[features_cols]])
    y_test = np.array([1] * len(test_genuine) + [0] * len(impostors_test))

    return X_train, y_train, X_test, y_test

'''RESULT WRITING'''
def write_results(model, split_name, best_params, best_cv_acc, train_acc, test_acc, metrics, results_path, anim_name):
    precision, recall, spec, roc_auc, eer = metrics
    row = {
        'Model': f"{model} ({split_name})",
        'Animation': anim_name,
        'Best Parameters': best_params,
        'Best CV Accuracy': round(best_cv_acc, 4),
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

# ------------------- TRAIN AND EVALUATE FUNCTION -------------------
def train_and_evaluate_per_person(X_train_total, y_train_total, animation_test_sets, model_fn, split_name, results_file, num_seed=20):
    for anim_name, (X_test_anim, y_test_anim) in animation_test_sets.items():
        per_person_metrics = []
        per_person_scores = []

        for person in people:
            # select only samples for this animation and person
            person_mask = (X_train_total.index.isin(dataset[(dataset['tester_id']==person) & (dataset['anim_name']==anim_name)].index))
            X_train_person = X_train_total.loc[person_mask]
            y_train_person = y_train_total.loc[person_mask]

            if len(X_train_person) == 0:
                continue

            pipeline, param_grid = model_fn()
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_person, y_train_person)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test_anim)
            if hasattr(best_model, "predict_proba"):
                y_score = best_model.predict_proba(X_test_anim)[:, 1]
            else:
                y_score = best_model.decision_function(X_test_anim)

            test_acc = accuracy_score(y_test_anim, y_pred)
            prec = precision_score(y_test_anim, y_pred)
            rec = recall_score(y_test_anim, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test_anim, y_pred).ravel()
            spec = tn / (tn + fp)
            roc_auc = roc_auc_score(y_test_anim, y_score)
            fpr, tpr, _ = roc_curve(y_test_anim, y_score)
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fnr - fpr))
            eer = fpr[eer_index]

            per_person_metrics.append((prec, rec, spec, roc_auc, eer))
            per_person_scores.append(test_acc)

        # average over all persons for this animation
        mean_test = np.mean(per_person_scores)
        mean_metrics = np.mean(per_person_metrics, axis=0)
        best_params = grid.best_params_

        write_results(
            model_name,
            split_name,
            best_params,
            grid.best_score_,
            best_model.score(X_train_total, y_train_total),
            mean_test,
            mean_metrics,
            results_file,
            anim_name
        )

# ------------------- FILE PATH -------------------
results_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results.csv"
if os.path.exists(results_file):
    os.remove(results_file)

# ------------------- SESSION SPLIT PER ANIMATION -------------------
num_seed = 3
for model_name, model_fn in model_list:
    for anim in animation_names:
        best_model_overall = None
        best_avg_test_acc = -np.inf
        best_X_train_overall = None
        best_y_train_overall = None
        best_seed = None

        # Train 20 different S1+S2 splits
        for seed in range(num_seed):
            all_person_train = []
            all_person_y_train = []
            all_person_test_acc = []

            # Prepare training data for all persons for this animation
            for person in dataset['tester_id'].unique():
                person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == anim)]
                if len(person_data) == 0:
                    continue

                X_train_p, y_train_p, X_test_p, y_test_p = prepare_train_test_data(person_data, seed=seed)
                all_person_train.append(X_train_p)
                all_person_y_train.append(y_train_p)

                # Train temporary model per person to compute test accuracy
                pipeline, param_grid = model_fn()
                grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train_p, y_train_p)
                y_pred = grid.best_estimator_.predict(X_test_p)
                test_acc = accuracy_score(y_test_p, y_pred)
                all_person_test_acc.append(test_acc)

            # Aggregate train data across persons
            X_train_total = pd.concat(all_person_train)
            y_train_total = pd.Series(np.concatenate(all_person_y_train))

            # Compute average test accuracy for this iteration
            avg_test_acc = np.mean(all_person_test_acc)

            # Train on full train set with GridSearchCV
            pipeline, param_grid = model_fn()
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_total, y_train_total)

            # Update best model if average test accuracy is higher
            if avg_test_acc > best_avg_test_acc:
                best_avg_test_acc = avg_test_acc
                best_model_overall = grid.best_estimator_
                best_X_train_overall = X_train_total
                best_y_train_overall = y_train_total
                best_params_overall = grid.best_params_
                best_seed = seed

        # Evaluate the best model on 20 different S3 test sets
        metrics_accum = []
        for seed in range(num_seed):
            all_person_test = []
            all_person_y_test = []

            for person in dataset['tester_id'].unique():
                person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == anim)]
                if len(person_data) == 0:
                    continue

                _, _, X_test_p, y_test_p = prepare_train_test_data(person_data, seed=seed)
                all_person_test.append(X_test_p)
                all_person_y_test.append(y_test_p)

            X_test_total = pd.concat(all_person_test)
            y_test_total = np.concatenate(all_person_y_test)

            # Evaluate best model
            y_pred = best_model_overall.predict(X_test_total)
            if hasattr(best_model_overall, "predict_proba"):
                y_score = best_model_overall.predict_proba(X_test_total)[:, 1]
            else:
                y_score = best_model_overall.decision_function(X_test_total)

            test_acc = accuracy_score(y_test_total, y_pred)
            prec = precision_score(y_test_total, y_pred)
            rec = recall_score(y_test_total, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test_total, y_pred).ravel()
            spec = tn / (tn + fp)
            roc_auc = roc_auc_score(y_test_total, y_score)
            fpr, tpr, _ = roc_curve(y_test_total, y_score)
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fnr - fpr))
            eer = fpr[eer_index]

            metrics_accum.append((test_acc, prec, rec, spec, roc_auc, eer))

        # Average metrics across 20 S3 test sets
        avg_metrics = np.mean(metrics_accum, axis=0)
        test_acc_avg, prec_avg, rec_avg, spec_avg, roc_auc_avg, eer_avg = avg_metrics

        write_results(
            model_name,
            f"S1+S2 vs S3",
            best_params_overall,                  
            best_avg_test_acc,
            best_model_overall.score(best_X_train_overall, best_y_train_overall),
            test_acc_avg,                          
            (prec_avg, rec_avg, spec_avg, roc_auc_avg, eer_avg),  
            results_file,
            anim
        )



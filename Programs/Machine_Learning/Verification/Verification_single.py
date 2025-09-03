import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
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

def load_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    # Get all the animation names
    dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))
    # Get all the tester names
    dataset['tester_id'] = dataset['file_key'].apply(lambda x: x.split('_')[0])
    # Get all the session names
    dataset['session_id'] = dataset['file_key'].apply(lambda x: x.split('_')[1])
    return dataset

def get_feature_columns():
    return [f'f{i}' for i in range(83)]

def get_unique_animations(dataset):
    return dataset['anim_name'].unique()

def get_unique_people(dataset):
    return dataset['tester_id'].unique()

'''PIPELINE DEFINITION'''
def get_classifiers_with_grid():
    # Naive Bayes
    nb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('clf', GaussianNB())
    ])
    nb_params = {
        'scaler': [MinMaxScaler(), StandardScaler(), RobustScaler()]
    }

    '''# K-Nearest Neighbors
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
    '''

    # Return a list of tuples (name, pipeline, param_grid)
    return [
        ("Naive Bayes", nb_pipeline, nb_params),
        #("KNN", knn_pipeline, knn_params),
        #("Logistic Regression", logreg_pipeline, logreg_params),
        #("NuSVC", nusvc_pipeline, nusvc_params),
        #("Random Forest", rf_pipeline, rf_params),
        #("SVC", svc_pipeline, svc_params),
        #("MLP", mlp_pipeline, mlp_params),
    ]

'''PREPARE THE DATA FOR VERIFICATION'''
def prepare_train_test_data(dataset, person_data, seed, features_cols):
    # Get the id of the current genuine person
    tester_id = person_data['tester_id'].iloc[0]

    train_genuine = person_data[person_data['session_id'].isin(['S1', 'S2'])]
    test_genuine = person_data[person_data['session_id'] == 'S3']
    
    # Subset of S1 and S2 of all impostors
    impostors_train_pool = dataset[
        (dataset['tester_id'] != tester_id) & 
        (dataset['session_id'].isin(['S1', 'S2']))
    ]
    # Subset of S3 of all impostors
    impostors_test_pool = dataset[
        (dataset['tester_id'] != tester_id) & 
        (dataset['session_id'] == 'S3')
    ]

    # Sample from impostor S1 and S2 random data with the same lenght as the genuine training set 
    impostors_train = impostors_train_pool.sample(
        n=len(train_genuine), random_state=seed
    )
    # Sample from impostor S3 random data with the same lenght as the genuine test set 
    impostors_test = impostors_test_pool.sample(
        n=len(test_genuine), random_state=seed
    )
    
    # Label genuine with 1 and impostors with 0
    X_train = pd.concat([train_genuine[features_cols], impostors_train[features_cols]])
    y_train = np.array([1] * len(train_genuine) + [0] * len(impostors_train))
    X_test = pd.concat([test_genuine[features_cols], impostors_test[features_cols]])
    y_test = np.array([1] * len(test_genuine) + [0] * len(impostors_test))

    return X_train, y_train, X_test, y_test

'''WRITE RESULTS'''
def write_results(model, split_name, best_params, best_cv_acc, train_acc, test_acc, metrics, results_path, anim_name):
    precision, recall, spec, roc_auc, eer = metrics
    row = {
        'Model': f"{model}",
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

'''EVALUATE THE MODEL'''
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    return fpr[eer_index]

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    
    # Some models don’t have predict_proba
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

    # Return individual metrics
    return test_acc, prec, rec, spec, roc_auc, eer


'''TRAIN THE MODEL'''
def train_best_model(dataset, animation, pipeline, param_grid, features_cols, num_seed):
    best_avg_test_acc = -np.inf
    best_seed = None

    # Find the best seed without GridSearchCV to reduce computation load
    for seed in range(num_seed):
        test_accs = []
        for person in dataset['tester_id'].unique():
            person_data = dataset[
                (dataset['tester_id'] == person) &
                (dataset['anim_name'] == animation)
            ]
            X_train_p, y_train_p, X_test_p, y_test_p = prepare_train_test_data(
                dataset, person_data, seed, features_cols
            )

            # Train pipeline with default parameters only to reduce computation time
            pipeline.fit(X_train_p, y_train_p)
            y_pred = pipeline.predict(X_test_p)
            test_accs.append(accuracy_score(y_test_p, y_pred))

        avg_test_acc = np.mean(test_accs)
        if avg_test_acc > best_avg_test_acc:
            best_avg_test_acc = avg_test_acc
            best_seed = seed

    # Train final GridSearchCV model on the best seed 
    X_train_all, y_train_all = [], []
    for person in dataset['tester_id'].unique():
        person_data = dataset[
            (dataset['tester_id'] == person) &
            (dataset['anim_name'] == animation)
        ]
        X_train_p, y_train_p, _, _ = prepare_train_test_data(
            dataset, person_data, best_seed, features_cols
        )

        # Combine all the different tester train sets into a single train test
        X_train_all.append(X_train_p)
        y_train_all.append(y_train_p)
    
    # Train only the best model with hyperparameter tuning
    best_model = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    best_model.fit(pd.concat(X_train_all), np.concatenate(y_train_all))

    return best_model, X_train_all, y_train_all, best_model.best_params_, best_avg_test_acc


def test_best_model(dataset, animation, best_model, features_cols, num_seed):
    metrics_accum = []

    for seed in range(num_seed):
        all_X_test, all_y_test = [], []
        for person in dataset['tester_id'].unique():
            person_data = dataset[(dataset['tester_id'] == person) & (dataset['anim_name'] == animation)]
    
            _, _, X_test_p, y_test_p = prepare_train_test_data(dataset, person_data, seed, features_cols)
            all_X_test.append(X_test_p)
            all_y_test.append(y_test_p)

        X_test_total = pd.concat(all_X_test)
        y_test_total = np.concatenate(all_y_test)

        test_acc, prec, rec, spec, roc_auc, eer = evaluate_model(best_model, X_test_total, y_test_total)
        metrics_accum.append((test_acc, prec, rec, spec, roc_auc, eer))


    # Average over all seeds
    avg_metrics = np.mean(metrics_accum, axis=0)
    return avg_metrics 

'''EXECUTE THE FUNCTIONS'''

csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
#csv_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Feature_csv\feature_vector.csv"

results_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results.csv"
#results_path = r"C:\Users\Davide Mascheroni\Desktop\movingText\movingText\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results.csv"

# Delete existing results file if it exists
if os.path.exists(results_path):
    os.remove(results_path)

dataset = load_dataset(csv_path)
features_cols = get_feature_columns()
num_seed = 20

for clf_name, clf_pipeline, clf_params in get_classifiers_with_grid():
    for animation in get_unique_animations(dataset):
        # Train the best model
        best_model, X_train_all, y_train_all, best_params, best_cv_acc = train_best_model(
            dataset, animation, clf_pipeline, clf_params, features_cols, num_seed
        )

        # Test the best model across all seeds and get average metrics
        test_acc, precision, recall, spec, roc_auc, eer = test_best_model(
            dataset, animation, best_model, features_cols, num_seed
        )

        write_results(
            model=clf_name,
            split_name="Final",
            best_params=best_params,
            best_cv_acc=best_cv_acc,
            train_acc=best_model.score(pd.concat(X_train_all), np.concatenate(y_train_all)),
            test_acc=test_acc,
            metrics=(precision, recall, spec, roc_auc, eer),
            results_path=results_path,
            anim_name=animation
        )


import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Feature_csv\feature_vector.csv"
dataset = pd.read_csv(csv_path)

# Extract animation name
dataset['anim_name'] = dataset['file_key'].apply(lambda x: '_'.join(x.split('_')[-3:]))

# Get all unique animations
animation_names = dataset['anim_name'].unique()

for anim in animation_names:
    animation_name = anim
    subset = dataset[dataset['anim_name'] == anim].copy()
    subset['tester_id'] = subset['file_key'].apply(lambda x: x.split('_')[0])
    subset['session_id'] = subset['file_key'].apply(lambda x: x.split('_')[1])

    X = subset.loc[:, 'f0':'f82']
    y = subset['tester_id']

    # ---- Session-based split ----
    train_subset = subset[subset['session_id'].isin(['S1', 'S2'])]
    test_subset = subset[subset['session_id'] == 'S3']

    X_train_sess = train_subset.loc[:, 'f0':'f82']
    y_train_sess = train_subset['tester_id']
    X_test_sess = test_subset.loc[:, 'f0':'f82']
    y_test_sess = test_subset['tester_id']

    print(f"\n[Session Split] Animation: {animation_name}")
    print(f"Train (S1+S2): {X_train_sess.shape}, Test (S3): {X_test_sess.shape}")
    print(f"Train label distribution: {y_train_sess.value_counts().to_dict()}")
    print(f"Test label distribution: {y_test_sess.value_counts().to_dict()}")

    if X_train_sess.empty or X_test_sess.empty:
        print("⚠️ Skipping due to empty train or test set.")
        continue

    unseen_labels = set(y_test_sess) - set(y_train_sess)
    if unseen_labels:
        print(f"⚠️ Skipping due to unseen labels in test set: {unseen_labels}")
        continue

    # ---- Random 80/20 split ----
    try:
        X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n[Random Split] Animation: {animation_name}")
        print(f"Train: {X_train_rand.shape}, Test: {X_test_rand.shape}")
        print(f"Train label distribution: {y_train_rand.value_counts().to_dict()}")
        print(f"Test label distribution: {y_test_rand.value_counts().to_dict()}")

    except ValueError as e:
        print(f"⚠️ Split failed (likely due to class imbalance): {e}")
        continue

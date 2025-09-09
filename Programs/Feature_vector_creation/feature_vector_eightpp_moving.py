import os
import pandas as pd
import numpy as np
from scipy.stats import gmean, skew, kurtosis, median_abs_deviation, iqr
from scipy.spatial import distance
import duckdb
import fv_constant

# Statistical functions
stat_funcs = [np.min, np.max, np.mean, gmean, np.median, np.std,
              lambda x: median_abs_deviation(x, scale=1), skew, iqr, kurtosis]
#No gmean for difference between left and right pupil diameter
diff_funcs = [np.min, np.max, np.mean, np.median, np.std,
              lambda x: median_abs_deviation(x, scale=1), skew, iqr, kurtosis]

import os
import pandas as pd
import numpy as np
from scipy.stats import gmean, skew, kurtosis, median_abs_deviation, iqr
from scipy.spatial import distance
import duckdb
import fv_constant

# Statistical functions
stat_funcs = [np.min, np.max, np.mean, gmean, np.median, np.std,
              lambda x: median_abs_deviation(x, scale=1), skew, iqr, kurtosis]
diff_funcs = [np.min, np.max, np.mean, np.median, np.std,
              lambda x: median_abs_deviation(x, scale=1), skew, iqr, kurtosis]

# Restrict to specific testers
selected_testers = [1, 2, 4, 5, 19, 30, 31, 35]

# Prepare file paths
expected = []
for tester in selected_testers:  # Only use selected testers
    for session in range(1, 4):
        for trial in range(1, 4):
            for f in fv_constant.filenames:
                for anim in fv_constant.animnames:
                    path = os.path.join(
                        fv_constant.BASE_PATH,
                        f"Tester{tester}", f"Session{session}", f"Trial{trial}",
                        f"T{tester}-S{session}-TRY{trial}-{anim}_{f}.csv"
                    )
                    expected.append((path, f"T{tester}_S{session}_TRY{trial}_{anim}_{f}"))

# Load data and extract fixations
fpogs_vector = []
all_dfs = {}
for path, key in expected:
    if not os.path.isfile(path):
        continue
    df = pd.read_csv(path)
    all_dfs[key] = df
    result = duckdb.query("""
        SELECT AVG(FPOGX) AS FPOGX, AVG(FPOGY) AS FPOGY,
               MAX(FPOGS) AS FPOGS, MAX(FPOGD) AS FPOGD, FPOGID
        FROM df WHERE FPOGV = '1'
        GROUP BY FPOGID ORDER BY FPOGID
    """).to_df()
    if result.empty:
        continue
    fpogs_vector.append({
        'file_key': key,
        'FPOGX': result['FPOGX'].tolist(),
        'FPOGY': result['FPOGY'].tolist(),
        'FPOGS': result['FPOGS'].tolist(),
        'FPOGD': result['FPOGD'].tolist(),
        'FPOGID': result['FPOGID'].tolist()
    })

# Overwrite fixation vector file if it exists
if os.path.exists(fv_constant.FIX_VECTOR_PATH_EIGHT_M):
    os.remove(fv_constant.FIX_VECTOR_PATH_EIGHT_M)

# Save fixation vector
pd.DataFrame(fpogs_vector).to_csv(fv_constant.FIX_VECTOR_PATH_EIGHT_M, index=False)

features = []

# f0: number of fixations
f0_rows = []
for key, df in all_dfs.items():
    result = duckdb.query("""
        SELECT COUNT(DISTINCT FPOGID) AS number_of_fixations
        FROM df
    """).to_df()
    f0_rows.append({'file_key': key, 'f0': result['number_of_fixations'][0]})
features.append(pd.DataFrame(f0_rows))

# f1–f10: fixation duration stats
f1_rows = []
for d in fpogs_vector:
    row = {'file_key': d['file_key']}
    for i, func in enumerate(stat_funcs):
        row[f'f{i+1}'] = func(d['FPOGD']) if len(d['FPOGD']) > 0 else 0
    f1_rows.append(row)
features.append(pd.DataFrame(f1_rows))

# f11–f20: distance between fixations
f11_rows = []
for d in fpogs_vector:
    distances = []
    for i in range(len(d['FPOGX']) - 1):
        x1, y1 = d['FPOGX'][i], d['FPOGY'][i]
        x2, y2 = d['FPOGX'][i+1], d['FPOGY'][i+1]
        dist = distance.euclidean([x1, y1], [x2, y2])
        distances.append(dist)
    row = {'file_key': d['file_key']}
    for i, func in enumerate(stat_funcs):
        row[f'f{11+i}'] = func(distances) if len(distances) > 0 else 0
    f11_rows.append(row)
features.append(pd.DataFrame(f11_rows))

# f21–f30: saccade speed
spd_rows = []
for d in fpogs_vector:
    speeds = []
    for i in range(len(d['FPOGX']) - 1):
        dist = distance.euclidean([d['FPOGX'][i], d['FPOGY'][i]],
                                  [d['FPOGX'][i+1], d['FPOGY'][i+1]])
        time = d['FPOGS'][i+1] - d['FPOGS'][i] - d['FPOGD'][i]
        if time > 0:
            speeds.append(dist / time)
    row = {'file_key': d['file_key']}
    for i, func in enumerate(stat_funcs):
        row[f'f{21+i}'] = func(speeds) if len(speeds) > 0 else 0
    spd_rows.append(row)
features.append(pd.DataFrame(spd_rows))

# f31: scanpath length
f31_rows = []
for d in fpogs_vector:
    total = np.sum([distance.euclidean([d['FPOGX'][i], d['FPOGY'][i]],
                                       [d['FPOGX'][i+1], d['FPOGY'][i+1]])
                    for i in range(len(d['FPOGX']) - 1)])
    f31_rows.append({'file_key': d['file_key'], 'f31': total})
features.append(pd.DataFrame(f31_rows))

# f32–f41: LPD stats
lpd_rows = []
for key, df in all_dfs.items():
    lpd = duckdb.query("SELECT LPD FROM df WHERE LPV = '1'").to_df()['LPD'].tolist()
    row = {'file_key': key}
    for i, func in enumerate(stat_funcs):
        row[f'f{32+i}'] = func(lpd) if len(lpd) > 0 else 0
    lpd_rows.append(row)
features.append(pd.DataFrame(lpd_rows))

# f42–f51: RPD stats
rpd_rows = []
for key, df in all_dfs.items():
    rpd = duckdb.query("SELECT RPD FROM df WHERE RPV = '1'").to_df()['RPD'].tolist()
    row = {'file_key': key}
    for i, func in enumerate(stat_funcs):
        row[f'f{42+i}'] = func(rpd) if len(rpd) > 0 else 0
    rpd_rows.append(row)
features.append(pd.DataFrame(rpd_rows))

# f52–f61: ratio LPD/RPD
ratio_rows = []
for key, df in all_dfs.items():
    ratio = duckdb.query("SELECT LPD/RPD AS R FROM df WHERE LPV = '1' AND RPV = '1'").to_df()['R'].tolist()
    row = {'file_key': key}
    for i, func in enumerate(stat_funcs):
        row[f'f{52+i}'] = func(ratio) if len(ratio) > 0 else 0
    ratio_rows.append(row)
features.append(pd.DataFrame(ratio_rows))

# f62–f70: difference LPD - RPD (no gmean)
diff_rows = []
for key, df in all_dfs.items():
    diff = duckdb.query("SELECT LPD - RPD AS D FROM df WHERE LPV = '1' AND RPV = '1'").to_df()['D'].tolist()
    row = {'file_key': key}
    for i, func in enumerate(diff_funcs):
        row[f'f{62+i}'] = func(diff) if len(diff) > 0 else 0
    diff_rows.append(row)
features.append(pd.DataFrame(diff_rows))

# f71: Number of blinks
f71_rows = []
for key, df in all_dfs.items():
    result = duckdb.query("""
        SELECT COUNT(DISTINCT BKID) AS BKID_COUNT
        FROM df
        WHERE CAST(BKID AS INTEGER) > 0
    """).to_df()
    f71_rows.append({'file_key': key, 'f71': result['BKID_COUNT'][0]})
features.append(pd.DataFrame(f71_rows))

# f72–f74: blink duration
blink_rows = []
for key, df in all_dfs.items():
    bkdur = duckdb.query("SELECT BKDUR FROM df WHERE BKDUR IS NOT NULL AND BKDUR != 0").to_df()['BKDUR'].tolist()
    row = {'file_key': key}
    row['f72'] = np.mean(bkdur) if len(bkdur) > 0 else 0
    row['f73'] = np.min(bkdur) if len(bkdur) > 0 else 0
    row['f74'] = np.max(bkdur) if len(bkdur) > 0 else 0
    blink_rows.append(row)
features.append(pd.DataFrame(blink_rows))

# f75–f82: directional movement distances and counts
direction_rows = []
for d in fpogs_vector:
    file_key = d['file_key']
    dx_up = dx_down = dx_left = dx_right = 0.0
    cnt_up = cnt_down = cnt_left = cnt_right = 0
    for i in range(len(d['FPOGX']) - 1):
        dx = d['FPOGX'][i+1] - d['FPOGX'][i]
        dy = d['FPOGY'][i+1] - d['FPOGY'][i]
        if dy < 0:
            dx_up += abs(dy)
            cnt_up += 1
        elif dy > 0:
            dx_down += abs(dy)
            cnt_down += 1
        if dx > 0:
            dx_right += abs(dx)
            cnt_right += 1
        elif dx < 0:
            dx_left += abs(dx)
            cnt_left += 1
    row = {
        'file_key': file_key,
        'f75': dx_up, 'f76': dx_down, 'f77': dx_left, 'f78': dx_right,
        'f79': cnt_up, 'f80': cnt_down, 'f81': cnt_left, 'f82': cnt_right
    }
    direction_rows.append(row)
features.append(pd.DataFrame(direction_rows))

# Merge all features
final = features[0]
for df in features[1:]:
    final = final.merge(df, on='file_key')

# Overwrite output file if it exists
if os.path.exists(fv_constant.EIGHT_M_OUT_PATH):
    os.remove(fv_constant.EIGHT_M_OUT_PATH)

final.to_csv(fv_constant.EIGHT_M_OUT_PATH, index=False)
print("Saved:", fv_constant.EIGHT_M_OUT_PATH)

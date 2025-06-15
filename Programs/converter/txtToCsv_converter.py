import re
import csv
import os
from pathlib import Path

# Define filenames
filenames = ["SL_LIT", "SL_BIG", "FA_LIT", "FA_BIG"]
animname = ["VB", "HS"]

total_testers = 33

# Iterate over testers, sessions, and trials
for tester in range(1, total_testers + 1):
    for session in range(1, 4):  # Sessions 1 to 3
        for trial in range(1, 4):  # Trials 1 to 3
            for fname in filenames:
                for anim_name in animname:
                    # Input .txt file path
                    txt_file = os.path.join(
                        "Results",
                        f"Tester{tester}",
                        f"Session{session}",
                        f"Trial{trial}",
                        f"T{tester}-S{session}-TRY{trial}-{anim_name}_{fname}.txt"
                    )

                    if not os.path.exists(txt_file):
                        continue  # Skip if file doesn't exist

                    # Output directory and file
                    output_dir = os.path.join(
                        "Results_csv",
                        f"Tester{tester}",
                        f"Session{session}",
                        f"Trial{trial}"
                    )
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                    csv_file = os.path.join(
                        output_dir,
                        f"T{tester}-S{session}-TRY{trial}-{anim_name}_{fname}.csv"
                    )

                    # Regex to match <REC ... />
                    rec_pattern = re.compile(r'<REC (.+?) />')

                    records = []
                    with open(txt_file, 'r') as file:
                        for line in file:
                            match = rec_pattern.search(line)
                            if match:
                                attributes_str = match.group(1)
                                attributes = dict(re.findall(r'(\w+)="(.*?)"', attributes_str))
                                records.append(attributes)

                    if records:
                        headers = records[0].keys()
                        with open(csv_file, 'w', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=headers)
                            writer.writeheader()
                            writer.writerows(records)


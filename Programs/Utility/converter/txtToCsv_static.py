import re
import csv
import os
from pathlib import Path

# Filenames only include the animation type now
filenames = ["ST_LIT", "ST_BIG"]

total_testers = 8

# Regex to match <REC ... /> robustly
rec_pattern = re.compile(r'<REC\s+(.*?)\s*/?>', re.DOTALL)

for tester in range(1, total_testers + 1):
    for session in range(1, 4):
        for trial in range(1, 4):
            for fname in filenames:
                txt_file = os.path.join(
                    "Results_Static",
                    f"Tester{tester}",
                    f"Session{session}",
                    f"Trial{trial}",
                    f"T{tester}-S{session}-TRY{trial}-{fname}.txt"
                )

                if not os.path.exists(txt_file):
                    continue

                output_dir = os.path.join(
                    "Results_Static_csv",
                    f"Tester{tester}",
                    f"Session{session}",
                    f"Trial{trial}"
                )
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                csv_file = os.path.join(
                    output_dir,
                    f"T{tester}-S{session}-TRY{trial}-{fname}.csv"
                )

                records = []

                # Read the full file content
                with open(txt_file, 'r') as file:
                    content = file.read()
                    for match in rec_pattern.finditer(content):
                        attributes_str = match.group(1)
                        attributes = dict(re.findall(r'([\w-]+)="(.*?)"', attributes_str))
                        records.append(attributes)

                if records:
                    # Use keys from the first record to preserve original column order
                    headers = list(records[0].keys())
                    with open(csv_file, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(records)

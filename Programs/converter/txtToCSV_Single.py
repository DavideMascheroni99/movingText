import re
import csv
import os
from pathlib import Path
import tkinter
from tkinter import simpledialog

# Create a tkinter window (hidden)
application_window = tkinter.Tk()
application_window.withdraw()  # Hide the root window

# Ask for user input
tester_number = simpledialog.askstring("Input", "Input tester number", parent=application_window)
session_number = simpledialog.askstring("Input", "Input session number", parent=application_window)
trial_number = simpledialog.askstring("Input", "Input trial number", parent=application_window)

# Convert to integers
tester = int(tester_number)
session = int(session_number)  
trial = int(trial_number)

# Define filenames
filenames = ["SL_LIT", "SL_BIG", "FA_LIT", "FA_BIG"]
animname = ["VB", "HS"]

# Iterate over testers, sessions, and trials
for fname in filenames:
    for anim_name in animname:
        # Input .txt file path
        txt_file = os.path.join(
            #"movingText",
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
            #"movingText",
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


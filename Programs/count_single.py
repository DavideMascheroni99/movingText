import os
import tkinter
from tkinter import simpledialog

# Create a tkinter window (hidden)
application_window = tkinter.Tk()
application_window.withdraw()  # Hide the root window

# Ask for user input
tester_number = simpledialog.askstring("Input", "Input tester number", parent=application_window)
session_number = simpledialog.askstring("Input", "Input session number", parent=application_window)
trial_number = simpledialog.askstring("Input", "Input trial number", parent=application_window)

# Define filenames
filenames = ["SL_LIT", "SL_BIG", "FA_LIT", "FA_BIG"]
animname = ["VB", "HS"]

for fname in filenames:
    for anim_name in animname:
        file_path = os.path.join(
            #"movingText",
            "Results",
            f"Tester{tester_number}",
            f"Session{session_number}",
            f"Trial{trial_number}",
            f"T{tester_number}-S{session_number}-TRY{trial_number}-{anim_name}_{fname}.txt"
            )


        try:
            with open(file_path, "r") as f:
                data = f.read()
                word = 'FPOGX="0.00000"'
                count = data.count(word)
                print(f"{word} occurred {count} times in {file_path}.")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
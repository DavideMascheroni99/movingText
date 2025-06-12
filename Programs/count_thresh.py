import os
import tkinter
from tkinter import simpledialog

# Create a tkinter window (hidden)
application_window = tkinter.Tk()
application_window.withdraw()  # Hide the root window

# Ask for user input
total_testers = simpledialog.askstring("Input", "Input number of testers", parent=application_window)
thresh_number = simpledialog.askstring("Input", "Input a threshold", parent=application_window)

# Convert to integers
total_testers_int = int(total_testers)
thresh_number_int = int(thresh_number)  

# Define filenames
filenames = ["SL_LIT", "SL_BIG", "FA_LIT", "FA_BIG"]
animname = ["VB", "HS"]

i = 0

# Iterate over testers, sessions, and trials
for tester in range(1, total_testers_int + 1):
    for session in range(1, 4):  # Sessions 1 to 3
        for trial in range(1, 4):  # Trials 1 to 3
            for fname in filenames:
                for anim_name in animname:
                    file_path = os.path.join(
                        "movingText",
                        "Results",
                        f"Tester{tester}",
                        f"Session{session}",
                        f"Trial{trial}",
                        f"T{tester}-S{session}-TRY{trial}-{anim_name}_{fname}.txt"
                    )

                    # Skip file if it doesn't exist
                    if not os.path.exists(file_path):
                        continue

                    # Read file and count target string
                    with open(file_path, "r") as f:
                        data = f.read()
                        word = 'FPOGX="0.00000"'
                        count = data.count(word)
                        if(count > thresh_number_int):
                            print(f"{word} occurred {count} times in {file_path}.")
                            i += 1
print(i)

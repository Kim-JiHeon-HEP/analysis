processes = [
    "signal",
    "ttbarG",
    "WG",
    "WW",
    "WWW",
    "WWZ", 
    "WZ",
    "WZZ",
    "ZG",
    "ZZ",
    "ZZZ"
]

subdirs = [
     "Original",
     "Second",
     "Third",
     "Fourth",
     "Fifth",
     "Sixth",
     "Seventh",
     "Eighth",
     "Nineth",
     "Tenth"
]


output_file = "jobs.txt"
with open(output_file, "w") as f:
    for process in processes:
        for subdir in subdirs:
            f.write(f"{process} {subdir}\n")

import os
import re
import csv

# Get current working directory
ROOT_DIR = os.getcwd()

FIRST_PATH = os.path.join(ROOT_DIR, "mytrain.csv")
SECOND_PATH = os.path.join(ROOT_DIR, "conv_text.csv")
DATASET_PATH = os.path.join(ROOT_DIR, "combined_dataset2.csv")

with open(FIRST_PATH, 'r') as infile1:
    with open(SECOND_PATH, 'r') as infile2:
        with open(DATASET_PATH, 'w') as outfile:
            filereader1 = csv.reader(infile1)
            filereader2 = csv.reader(infile2)
            filewriter = csv.writer(outfile)
            for row in filereader1:
                filewriter.writerow(row)
            for row in filereader2:
                filewriter.writerow(row)

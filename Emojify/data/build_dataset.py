import os
import re
import csv

# Get current working directory
ROOT_DIR = os.getcwd()

ORIGINAL_PATH = os.path.join(ROOT_DIR, "text_emotion.csv")
DATASET_PATH = os.path.join(ROOT_DIR, "dataset.csv")

emoji_dictionary = {"love": "0",
                    "fun": "2",
                    "sadness": "3",
                    "neutral": "5",
                    "empty": "5",
                    "surprise": "6",
                    "anger": "8",
                    "worry": "9",
                    "relief": "11",
                    "enthusiasm": "12",
                    "happiness": "13",
                    "hate": "14",
                    "boredom": "15"
}

with open(ORIGINAL_PATH, 'r') as infile:
    with open(DATASET_PATH, 'w') as outfile:
        filereader = csv.reader(infile)
        filewriter = csv.writer(outfile)
        i = 0
        for row in filereader:
            if (i == 0):
                i+=1
                continue
            content = re.sub('@[^ ]*', '', row[3])
            correct_emoji = emoji_dictionary[row[1]]
            filewriter.writerow([content, correct_emoji])
        print(i)

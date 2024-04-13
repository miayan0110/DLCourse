import os
import csv
import random

class LeukemiaDataSet:
    def __init__(self, path):
        self.path = path
        self.dataset = []

    def list_dir(self):
        for _, _, files in os.walk(self.path):
            for filename in files:
                print(filename)

    def generate_data(self):
        for _, _, files in os.walk(self.path):
            for filename in files:
                if filename.endswith(".bmp"):
                    label = (filename.find("all") > 0)*1
                    self.dataset.append([filename, label])
        random.shuffle(self.dataset)
        print(f"> Found {len(self.dataset)} .bmp files...")

    def divide_data(self, filenames=["training.csv", "validation.csv", "testing.csv"], data_amount=[5, 4, 1]):

        for i in range(len(filenames)):
            with open(filenames[i], "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Path", "label"])

                for _ in range(data_amount[i]):
                    writer.writerow(self.dataset.pop())

if __name__ == "__main__":
    path = "training_data"
    filenames = ["Path to train.csv", "Path to valid.csv", "Path to test.csv"]
    data_amount = [150, 50, 50]

    leukemia = LeukemiaDataSet(path)
    # leukemia.list_dir()
    leukemia.generate_data()
    leukemia.divide_data(filenames, data_amount)
    
import pandas as pd
import ResNet as rn
import dataloader as dl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# hyperparameters
hps = {
    "learning_rate": 0.01,
    "batch_size": 4,
    "n_epochs": 10
}

# history
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}


def evaluate():
    print("evaluate() not defined")

def test():
    print("test() not defined")

def train(dataLoader, model):
    dataset_size = dataLoader.__len__()
    n_epochs = hps["n_epochs"]
    lack_size = dataset_size%hps["batch_size"]
    is_lack = (lack_size > 0)
    n_batches = dataset_size//hps["batch_size"] + is_lack
    optimizer = optim.Adam(model.parameters(), lr=hps["learning_rate"])
    crossEntropy = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        correct = 0.0
        cost = 0.0

        for batch in range(n_batches):
            features = []
            labels = []

            for i in range(hps["batch_size"]):
                if is_lack and batch == n_batches-1 and len(features) == lack_size:
                    break
                feature, label = dataLoader.__getitem__(batch*hps["batch_size"]+i)
                features.append(feature)
                labels.append(label)
            features = torch.from_numpy(np.array(features)).float()
            labels = torch.from_numpy(np.array(labels)).float()

            model.train()

            optimizer.zero_grad()

            pred = model(features)
            loss = crossEntropy(pred, labels.long())
            loss.backward()

            optimizer.step()

            correct += (pred.argmax(dim=1) == labels).sum()
            cost += loss.item()
        epoch_acc = correct/dataset_size
        epoch_loss = cost/n_batches
        history["train_acc"].append(epoch_acc)
        history["train_loss"].append(epoch_loss)

        print(f"[ Train | {epoch+1:3d}/{n_epochs:3d} ] loss = {epoch_loss:.5f}, acc = {epoch_acc:.5f}")


def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

if __name__ == "__main__":
    dataLoader = dl.LeukemiaLoader("./Labs/Lab3/Lab3-Leukemia_Classification/training_data/", "train")

    res18 = rn.ResNet("18", rn.ResBlock, 3, [2, 2, 2, 2])
    train(dataLoader, res18)
import pandas as pd
import ResNet.py as rn
import dataloader.py as dl
import torch
import torch.nn as nn
import torch.optim as optim

# hyperparameters
hps = {
    "learning_rate": 0.01,
    "batch_size": 10,
    "n_epochs": 2
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
    n_batches = dataset_size//hps["batch_size"] + (dataset_size%hps["batch_size"] > 0)
    optimizer = optim.Adam(model.parameters(), lr=hps["learning_rate"])

    for epoch in range(n_epochs):
        features = []
        labels = []
        correct = 0.0
        cost = 0.0

        for batch in range(n_batches):

            for i in range(hps["batch_size"]):
                feature, label = dataLoader.__getitem__(i+epoch*hps["batch_size"])
                features.append(feature)
                labels.append(label)

            model.train()

            optimizer.zero_grad()

            pred = model(torch.FloatTensor(features))
            loss = nn.CrossEntropyLoss(pred, torch.FloatTensor(labels).long())
            loss.backward()

            optimizer.step()

            correct += (pred == labels).sum()
            cost += loss.item()
        epoch_acc = correct/dataset_size
        epoch_loss = cost/n_batches
        history["train_acc"].append(epoch_acc)
        history["train_loss"].append(epoch_loss)

        print(f"[ Train | {epoch:3d}/{n_epochs:3d} ] loss = {epoch_loss:.5f}, acc = {epoch_acc:.5f}")


        
    print("train() not defined")

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

if __name__ == "__main__":
    print("Good Luck :)")
    dataLoader = dl.LeukemiaLoader("./training_data/", "train")

    res18 = rn.ResNet("18", rn.ResBlock, 3, [2, 2, 2, 2])
    train(dataLoader, res18)
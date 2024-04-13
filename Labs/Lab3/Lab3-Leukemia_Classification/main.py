import pandas as pd
import ResNet as rn
import dataloader as dl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters
hps = {
    "learning_rate": 0.001,
    "batch_size": 50,
    "n_epochs": 10,
    "model": "50"
}

# history
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}


def evaluate(dataLoader, model):
    dataset_size = dataLoader.__len__()
    crossEntropy = nn.CrossEntropyLoss()

    features = []
    labels = []

    for i in range(dataset_size):
        feature, label = dataLoader.__getitem__(i)
        features.append(feature)
        labels.append(label)
    features = torch.from_numpy(np.array(features)).float()
    labels = torch.from_numpy(np.array(labels)).long()
    model.eval()

    pred = model(features)
    loss = crossEntropy(pred, labels)

    correct = (pred.argmax(dim=1) == labels).sum()

    valid_loss = loss.item()
    valid_acc = correct/dataset_size

    print(f"validation loss = {valid_loss:5f}, validation acc = {valid_acc:.5f}")


def test(dataLoader, model, max_acc, min_loss):
    dataset_size = dataLoader.__len__()
    crossEntropy = nn.CrossEntropyLoss()

    features = []
    labels = []

    for i in range(dataset_size):
        feature, label = dataLoader.__getitem__(i)
        features.append(feature)
        labels.append(label)
    features = torch.from_numpy(np.array(features)).float()
    labels = torch.from_numpy(np.array(labels)).long()
    model.eval()

    pred = model(features)
    loss = crossEntropy(pred, labels)

    correct = (pred.argmax(dim=1) == labels).sum()

    test_loss = loss.item()
    test_acc = correct/dataset_size

    history["test_acc"].append(test_acc)
    history["test_loss"].append(test_loss)

    print(f"test loss = {test_loss:5f}, test acc = {test_acc:.5f}")

    if test_acc >= max_acc and test_loss <= min_loss: 
        save_model(model)
        return test_acc, test_loss
    else:
        return max_acc, min_loss


def train(dataLoader, model):
    dataset_size = dataLoader.__len__()
    n_epochs = hps["n_epochs"]
    lack_size = dataset_size%hps["batch_size"]
    is_lack = (lack_size > 0)
    n_batches = dataset_size//hps["batch_size"] + is_lack
    optimizer = optim.Adam(model.parameters(), lr=hps["learning_rate"])
    crossEntropy = nn.CrossEntropyLoss()
    max_test_acc = 0.0
    min_test_loss = 10.0

    for epoch in range(n_epochs):
        correct = 0.0
        cost = 0.0

        for batch in range(n_batches):
            print(f"[ Train | {epoch+1:02d}/{n_epochs:02d} | batch {batch+1} ]")
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

        print(f"[ Train | {epoch+1:02d}/{n_epochs:02d} ] loss = {epoch_loss:.5f}, acc = {epoch_acc:.5f}")

        testloader = dl.LeukemiaLoader("./training_data/", "valid")
        max_test_acc, min_test_loss = test(testloader, model, max_test_acc, min_test_loss)


def save_model(model):
    path = "./models/res"+hps["model"]+".pt"
    print("saving model...")
    torch.save(model.state_dict(), path)


def plot(data, n_epochs):
    epochs = [x for x in range(n_epochs)]
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    
    plt.plot(epochs, data["train_acc"], label="ResNet18 Train")
    plt.plot(epochs, data["test_acc"], label="ResNet18 Test")

    plt.legend()
    plt.show()


def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

if __name__ == "__main__":
    dataLoader = dl.LeukemiaLoader("./training_data/", "train")

    res = rn.ResNet(hps["model"], rn.ResBottleneck, 3, [3, 4, 6, 3])
    print(res)
    train(dataLoader, res)
    # plot(history, hps["n_epochs"])
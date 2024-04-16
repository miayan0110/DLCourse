import ResNet as rn
import dataloader as dl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import seaborn as sn


def evaluate(valid_loader, model, max_acc=0.0, min_loss=10.0):
    device = hps["device"]
    crossEntropy = nn.CrossEntropyLoss()

    batch_valid_cost = []
    batch_valid_correct = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            features, labels = batch
            features = features.float().to(device)
            labels = labels.long().to(device)

            pred = model(features)
            loss = crossEntropy(pred, labels)

            batch_valid_cost.append(loss.item())
            batch_valid_correct.append((pred.argmax(dim=1) == labels).float().mean())

        valid_loss = sum(batch_valid_cost)/len(batch_valid_cost)
        valid_acc = sum(batch_valid_correct)/len(batch_valid_correct)
        
        history[hps["model"]+"_test_loss"].append(valid_loss)
        history[hps["model"]+"_test_acc"].append(valid_acc.item())
        
        print(f"validation loss = {valid_loss:5f}, validation acc = {valid_acc:.5f}")        
        if valid_acc >= max_acc and valid_loss <= min_loss: 
            save_model(model)
            return valid_acc, valid_loss
        else:
            return max_acc, min_loss        


def test(test_loader, model):
    device = hps["device"]
    crossEntropy = nn.CrossEntropyLoss()

    batch_test_cost = []
    tp, tn, fp, fn = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            features, labels = batch
            features = features.float().to(device)
            labels = labels.long().to(device)

            preds = model(features)
            loss = crossEntropy(preds, labels)

            batch_test_cost.append(loss.item())
            for _, (pred, label) in enumerate(zip(preds.argmax(dim=1), labels)):
                if label == 0 and pred == 0: fp += 1
                elif label == 0 and pred == 1: fn += 1
                elif label == 1 and pred == 0: tn += 1
                else: tp += 1
        
        test_loss = sum(batch_test_cost)/len(batch_test_cost)
        test_acc = (tp+fp)/(fp+fn+tp+tn)
        confusion_mat = [[fp/(fp+fn), fn/(fp+fn)], [tn/(tp+tn), tp/(tp+tn)]]

        print(f"test loss = {test_loss:5f}, test acc = {test_acc:.5f}")
        plot_confusion_mat(confusion_mat)

def train(train_loader, valid_loader, model):
    device = hps["device"]
    n_epochs = hps["n_epochs"]
    optimizer = optim.Adam(model.parameters(), lr=hps["learning_rate"])
    crossEntropy = nn.CrossEntropyLoss()
    max_valid_acc = 0.0
    min_valid_loss = 10.0

    for epoch in range(n_epochs):
        batch_train_cost = []
        batch_train_correct = []

        model.train()

        for batch in tqdm(train_loader):
            features, labels = batch
            features = features.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()

            pred = model(features)
            loss = crossEntropy(pred, labels)
            loss.backward()
            optimizer.step()

            batch_train_cost.append(loss.item())
            batch_train_correct.append((pred.argmax(dim=1) == labels).float().mean())

        epoch_loss = sum(batch_train_cost)/len(batch_train_cost)
        epoch_acc = sum(batch_train_correct)/len(batch_train_correct)
        history[hps["model"]+"_train_acc"].append(epoch_acc.item())
        history[hps["model"]+"_train_loss"].append(epoch_loss)
        
        print(f"[ Train | {epoch+1:02d}/{n_epochs:02d} ] loss = {epoch_loss:.5f}, acc = {epoch_acc:.5f}")
        max_valid_acc, min_valid_loss = evaluate(valid_loader, model, max_valid_acc, min_valid_loss)
        

def save_model(model):
    path = hps["save_path"]+hps["model"]+".pt"
    print("saving model...")
    torch.save(model.state_dict(), path)
    
def load_model(path, model):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def plot(data, n_epochs):
    epochs = [x for x in range(n_epochs)]
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    
    plt.plot(epochs, data["18_train_acc"], label="ResNet18 Train", marker=".")
    plt.plot(epochs, data["18_test_acc"], label="ResNet18 Test", marker=".")
    plt.plot(epochs, data["50_train_acc"], label="ResNet50 Train", marker=".")
    plt.plot(epochs, data["50_test_acc"], label="ResNet50 Test", marker=".")
    plt.plot(epochs, data["152_train_acc"], label="ResNet152 Train", marker=".")
    plt.plot(epochs, data["152_test_acc"], label="ResNet152 Test", marker=".")

    plt.legend()
    plt.show()
    

def plot_confusion_mat(confusion_mat):
    df = pd.DataFrame(confusion_mat, [0, 1], [0, 1])
    sn.heatmap(df, annot=True, cmap='Blues')


def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

# history
history = {"18_train_loss": [], "18_train_acc": [], "18_test_loss": [], "18_test_acc": [],
          "50_train_loss": [], "50_train_acc": [], "50_test_loss": [], "50_test_acc": [],
          "152_train_loss": [], "152_train_acc": [], "152_test_loss": [], "152_test_acc": []}

# confusion matrix
confusion_mat = []


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # hyperparameters
    hps = {
        "learning_rate": 0.01,
        "batch_size": 128,  # 8/4
        "n_epochs": 10,
        "model": "18",  #"50"/"152"
        "device": device,
        "data_path": "./training_data/",
        "save_path": "./models/new_res"
    }

    train_loader = DataLoader(dl.LeukemiaLoader(hps["data_path"], "train"), batch_size=hps["batch_size"], shuffle=True)
    test_loader = DataLoader(dl.LeukemiaLoader(hps["data_path"], "valid"), batch_size=hps["batch_size"], shuffle=False)

    res = rn.ResNet(hps["model"], rn.ResBlock, 3, [2, 2, 2, 2]).to(device)

    gc.collect()
    train(train_loader, test_loader, res)
    plot(history, hps["n_epochs"])

    test_loader = DataLoader(dl.LeukemiaLoader(hps["data_path"], "test"), batch_size=hps["batch_size"], shuffle=False)
    gc.collect()
    load_path = hps["save_path"]+hps["model"]+".pt"
    test(test_loader, load_model(load_path, res))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=512):
        super(SiameseNetwork, self).__init__()

        self.shared_nn = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size)  
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
       
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        img1_embedding = img1#self.shared_nn(img1)
        img2_embedding =img2 #self.shared_nn(img2)

        concatenated = torch.cat((img1_embedding, img2_embedding), dim=1)
        
      
        output = self.fc(concatenated)
        return output




class CustomDataset(Dataset):
    def __init__(self, data ):
        self.data = data
        self.data["img1"] = self.data["img1"].apply(lambda x: ast.literal_eval(x))
        self.data["img1"] = self.data["img1"].apply(lambda x: np.array(x,dtype="float32"))

        self.data["img2"] = self.data["img2"].apply(lambda x: ast.literal_eval(x) )
        self.data["img2"] = self.data["img2"].apply(lambda x: np.array(x,dtype="float32"))




    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row= self.data.iloc[index]
        img1,img2,label = row["img1"],row["img2"],row["label"]
        # img1,img2 = np.array(img1,dtype="float32"),np.array(img2,dtype="float32")
        # print(img1.shape,img1)
        # print(img1,type(img1))
        return img1, img2, label


def train(model, train_loader, criterion, optimizer):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for img1, img2, labels in tqdm(train_loader,desc= "loading Training Data"):
            img1, img2, labels = img1, img2, labels.float()#torch.tensor(labels, dtype=torch.float32)

            img1 = img1.to("cuda" if torch.cuda.is_available() else "cpu")
            img2 = img2.to("cuda" if torch.cuda.is_available() else "cpu")
            labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")

            optimizer.zero_grad()

            outputs = model(img1, img2)
            loss = criterion(outputs, labels.unsqueeze(1))  
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = (outputs > 0.5).float()  
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        return accuracy,epoch_loss

def val(model, val_loader, criterion):
 
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for img1, img2, labels in tqdm(val_loader,desc= "Loading Validation Data"):
        img1, img2, labels = img1, img2, labels.float()#torch.tensor(labels, dtype=torch.float32)
        # print(labels.shape)

        img1 = img1.to("cuda" if torch.cuda.is_available() else "cpu")
        img2 = img2.to("cuda" if torch.cuda.is_available() else "cpu")
        labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = model(img1, img2)
        loss = criterion(outputs, labels.unsqueeze(1))  


        running_loss += loss.item()

        predicted = (outputs > 0.5).float()  
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

    epoch_loss = running_loss / len(train_loader)

    accuracy = 100 * correct / total
    return accuracy,epoch_loss

if __name__ == "__main__":
   
    df = pd.read_csv("../data/train_embeddings.csv")
    train_df,val_df = train_test_split(df,test_size=0.2,random_state=42,shuffle= True)
    print("DataFrame Loaded Succesfully")
    train_dataset = CustomDataset(train_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = CustomDataset(val_df)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print("Dqataset Loaded Succesfully")
    embedding_size = 512
    model = SiameseNetwork(embedding_size)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-9,eps=1e-2)
    num_epochs = 2000
    best = 0
    last_epoch = 0
    thresh = 200
    print("Model Trainining Started...")
    for epoch in tqdm(range(num_epochs),desc = "Epochs"):

        accuracy,epoch_loss = train(model, train_loader, criterion, optimizer)
        val_accuracy,val_epoch_loss =val(model, val_loader, criterion)
        if val_accuracy>best:
            best =val_accuracy
            print("Best Model")
            torch.save(model.state_dict(),"./model/clssifier2.pt")
            last_epoch = epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        if epoch-last_epoch>thresh:
            break

# import torch

# def predict(model, img1, img2, threshold=0.5):
#     model.eval()
#     with torch.no_grad():
#         img1, img2 = img1.float(), img2.float()
#         output = model(torch.cat((img1, img2), dim=1))
#         prediction = (output > threshold).float()
#     return prediction.item()
# if __name__ == "__main__":
   
#     model = ... 
#     img1 = torch.randn(1, 512) 
#     img2 = torch.randn(1, 512)  

#     prediction = predict(model, img1, img2)
#     print("Prediction:", prediction)


import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'device available is {device}')

image_path = []
labels = []

for i in os.listdir("/content/animal-faces/afhq"):
    for label in os.listdir(f"/content/animal-faces/afhq/{i}"):
        for image in os.listdir(f"/content/animal-faces/afhq/{i}/{label}"):
            image_path.append(f"/content/animal-faces/afhq/{i}/{label}/{image}")
            labels.append(label)
df = pd.DataFrame(zip(image_path, labels), columns = ["image_path", "labels"])
df.head()

train = df.sample(frac = 0.7)
test = df.drop(train.index)

val = test.sample(frac = 0.5)
test = test.drop(val.index)

label_encoder = LabelEncoder()

label_encoder.fit(df['labels'])

transform = transforms.Compose([
     transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
     transforms.Resize((128, 128)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(15),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
     transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe['labels']))
    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).to(device)
        return image, label

train_dataset = CustomImageDataset(dataframe = train, transform = train_transform)
val_dataset = CustomImageDataset(dataframe = val, transform = test_transform)
test_dataset = CustomImageDataset(dataframe = test, transform = test_transform)

# Plot random sample images to check if the dataset is loaded correctly
n_rows = 3
n_cols = 3

f, axarr = plt.subplots(n_rows, n_cols)

for row in range(n_rows):
    for col in range(n_cols):
        image = Image.open(df.sample(n = 1)["image_path"].iloc[0]).convert("RGB")
        axarr[row, col].imshow(image)
        axarr[row, col].axis("off")
plt.show()

LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 50

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pooling = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear((128*16*16), 128)
        self.output = nn.Linear(128, len(df['labels'].unique()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.linear(x))
        x = self.output(x)

        return x

model = Net().to(device)

from torchsummary import summary
summary(model, input_size=(3, 128 , 128))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

total_loss_train_plot = []
total_loss_val_plot = []
total_acc_train_plot = []
total_acc_val_plot = []

for epoch in range(EPOCHS):
    model.train()
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()
    model.eval()
    with torch.no_grad():
       for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()
            val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
            total_acc_val += val_acc
    total_loss_train_plot.append(round(total_loss_train / 1000, 4))
    total_loss_val_plot.append(round(total_loss_val / 1000, 4))
    total_acc_train_plot.append(round((total_acc_train/train_dataset.__len__())*100, 4))
    total_acc_val_plot.append(round((total_acc_val/val_dataset.__len__())*100, 4))

    print(f""" Epoch {epoch+1}, Train loss: {round(total_loss_train / 1000, 4)}, Val loss: {round(total_loss_val /1000, 4)} Train Accuracy = {round((total_acc_train/train_dataset.__len__())*100, 4)} %, Val Accuracy = {round((total_acc_val/val_dataset.__len__())*100, 4)} %""")

    scheduler.step(total_loss_val)

torch.save(model.state_dict(), "my_model.pth")
print("Model saved safely!")

with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)

        acc = (torch.argmax(predictions, axis = 1) == labels).sum().item()
        loss = criterion(predictions, labels)
        total_loss_test += loss.item()
        total_acc_test += acc
print(f"Accuracy score is : {round((total_acc_test/test_dataset.__len__()) * 100, 4)} and Loss is {round(total_loss_test/1000, 4)}")

fig, axs = plt.subplots(1, 2, figsize = (15, 5))
axs[0].plot(total_loss_train_plot, label = "Train Loss")
axs[0].plot(total_loss_val_plot, label = "Val Loss")
axs[0].set_title("Training and Validation Loss over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(total_acc_train_plot, label = "Train Accuracy")
axs[1].plot(total_acc_val_plot, label = "Val Accuracy")
axs[1].set_title("Training and Validation Accuracy over Epochs")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
plt.show()

# 1. reading the image
# 2. transform the image with the transform object
# 3. predict through the model
# 4. inverse transform by label encoder
def predict_image(image_path):
    model.eval()  # Set to evaluation mode
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).to(device)  # Use test_transform with normalization
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        output = torch.argmax(output, axis=1).item()
    return label_encoder.inverse_transform([output])[0]

predict_image("/content/cat115.jpg")
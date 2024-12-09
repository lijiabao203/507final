import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datasets import load_dataset
from timm import create_model
from torchvision import models

# Here, I will define lots of parameters for changing the model. Switching models are based on parts of them.
load_size = 50 # It will be used in data loader, and the train data set will load 3 times of this while test data set load this number at once.
classes = 30 # Because of the device restriction, I only train first 30 classes. Change it if you have the condition.
ModelType = "CNN"
assert ModelType in ("CNN", # Basical CNN model
                     "VGG", # VGG-CNN model
                     "ViT", # ViT-FC model
                     "SWIM") # SWIM-FC model
num_epochs = 20 # number of epochs

# Some regular things to define
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define of models
# CNN
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 256 * 14 * 14)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# VGG-CNN
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)  # After pooling, the size becomes 256x14x14 (assuming input 224x224)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ViT
class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ViT, self).__init__()
        vit = create_model('vit_base_patch16_224', pretrained=True)
        self.features = vit
        for param in self.features.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1000,  512)  # After pooling, the size becomes 256x14x14 (assuming input 224x224)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# SWIM
class SWIM(nn.Module):
    def __init__(self, num_classes=10):
        super(SWIM, self).__init__()
        swim = create_model('swin_base_patch4_window7_224', pretrained=True)
        self.features = swim
        for param in self.features.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1000,  512)  # After pooling, the size becomes 256x14x14 (assuming input 224x224)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define preprocessing and data augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.Resampling.LANCZOS),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = []
        self.labels = []

        for i in range(len(labels)):
            if labels[i] < classes and images[i].mode == 'RGB':
                self.images.append(images[i])
                self.labels.append(labels[i])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset
ds = load_dataset("ethz/food101")
print(f"Data set construction start...")
train_dataset = CustomDataset(ds["train"]["image"], ds["train"]["label"], transform)
test_dataset = CustomDataset(ds["validation"]["image"], ds["validation"]["label"], transform)

print(f"Data loader construction start...")
train_loader = DataLoader(train_dataset, batch_size=3*load_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=load_size, shuffle=False)

print(f"Data preparation finished...")

# Initialize the model, loss function, and optimizer
if ModelType == "CNN":
    model = CNN(num_classes=classes)
elif ModelType == "VGG":
    model = VGG(num_classes=classes)
elif ModelType == "ViT":
    model = ViT(num_classes=classes)
else:
    model = SWIM(num_classes=classes)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = loss_fn.to(device)

# Training start
print(f"Model construction finish, staring training...")
acc_lis = []
los_lis = []
tim_lis = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    tim_rec = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        start_time = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels.long())
        loss.backward()
        optimizer.step()

        tim_rec += time.time() - start_time
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if i % 30 == 29:
            acc_rec = 100 * correct / total
            loss_rec = running_loss / (150*(i +1))
            acc_lis.append(acc_rec)
            los_lis.append(loss_rec)
            print(f"Batch {i + 1}/{int(len(train_dataset)/150)}, Loss: {loss_rec}, Accuracy: {acc_rec}%")
    # Print training stats
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataset)}, Accuracy: {100 * correct / total}%")
    print(f"In this epoch, model used {tim_rec} seconds for training")
    tim_lis.append(tim_rec)
    # if epoch > 10:
    #     break
    # if correct / total > 0.91:
    #     print("ACC is bigger than 0.91, early stop!")
    #     break
torch.save(model, 'E:/temp_comandother/programing/507 final/models/'+ModelType+classes+'classes'+num_epochs+'epochs'+'.pth')

# Evaluation loop
model.eval()
correct = 0
total = 0

print(f"Average training time for each epoch is: {sum(tim_lis) / len(tim_lis)} seconds")
print("Starting test model...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {100 * correct / total}%")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(los_lis, color='tab:blue', label='Training Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:orange')
ax2.plot(acc_lis, color='tab:orange', label='Training Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Training Loss and Accuracy')
plt.tight_layout()
plt.show()
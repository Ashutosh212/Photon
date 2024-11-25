import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from sigth2sound import *

def get_audio_torch(input_path, fixed_length=160000):
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] == 2:  # Convert stereo to mono
        audio = audio.mean(axis=0).unsqueeze(0)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    audio = torchaudio.functional.mu_law_encoding(audio, 16)

    if audio.shape[1] < fixed_length:
        audio = torch.nn.functional.pad(audio, (0, fixed_length - audio.shape[1]))
    else:
        audio = audio[:, :fixed_length]
    
    return audio.float()

# Label mapping
label_map = {'ballet': 0, 'bangra': 1, 'salsa': 2}

# Custom dataset class
class AudioDataset(Dataset):
    def __init__(self, data_dir, label_map, transform=None):
        self.data_dir = data_dir
        self.label_map = label_map
        self.transform = transform
        self.audio_files = []
        self.labels = []
        
        # Iterate through each class folder and load file paths and labels
        for label, idx in label_map.items():
            class_dir = os.path.join(data_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(class_dir, file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        audio = get_audio_torch(audio_path)  # Use provided get_audio_torch function
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio, label

# Define training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_predictions.double() / len(dataloader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Initialize and configure dataset and dataloader
data_dir = "F:\\Sample\\audio_dataset"  # Root folder containing 'salsa', 'ballet', 'bangra' subfolders
dataset = AudioDataset(data_dir=data_dir, label_map=label_map)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn_1d_soudnet(num_class=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, device, num_epochs=100)

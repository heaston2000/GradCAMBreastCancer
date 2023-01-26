# This is meant to be the CNN for Breast Cancer Detection competition run by Kaggle and the Radiological Society of North America
#   as an extension GradCAM is implemented on the dataset such that 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile

# Load the data (only csvs containing metadata) and stuff:
"""
Idea for loading the data: we have a training and a testing csv already downloaded,
The idea is to partition the indiviudal .csv's into a training and a testing set just based on the label
of each .csv and once we load in those, we go through them one batch at a time and:
- Download the samples from a batch
- Preprocess all the samples (need a function for this)
- Train the network on the batch
- Delete the batch from memory
"""
# Define data set of the metadata to make them iterable
class PatientMetaData():
    def __init__(self):
        self.all_meta_data  = pd.read_csv("train.csv")
        # We only have 29,427 of the images 
        self.patient_image_ids = self.all_meta_data.iloc[:29427, 1:3] # Columns for patient ID, imageID, left "L" or right "R" breast
        self.cancer_labels  = self.all_meta_data.iloc[:29427,6]
        #print(self.patient_image_ids)
        self.n_samples = len(self.patient_image_ids)
        #print(self.n_samples)
        
    def __getitem__(self, index):
        # Return patient ID, image ID, cancer bool
        patientID = self.patient_image_ids.iloc[index, 0]
        imageID = self.patient_image_ids.iloc[index, 1]
        filename = "Data/" + str(patientID) + "_" + str(imageID) + ".png"
        return filename, self.cancer_labels.iloc[index]
    
    def __len__(self):
        return self.n_samples
    
#     def features(self):
#         return self.n_features_x, self.n_features_y
    
#     def testSets(self):
#         return self.x_test, self.y_test
    

# hyper parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Pre-training stuff
classes = (1, 0) # this is wrong, change this later once we load in the data

# Create the training data loader:
meta_dataset = PatientMetaData()
data_loader   = DataLoader(dataset=meta_dataset, batch_size = batch_size, shuffle=True)

# NOW DEFINE THE ARCHITECTURE OF THE CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, 5), # Assuming 1 in channel (inputs are b&w), output channel size is 6 and kernels are 5x5
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Kernel size is 2x2 with stride of 2
            nn.Conv2d(6, 16, 5), # input channel size is equal to the last channel size, kernels are still 5x5
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(1000000, 64), #16 * 5*5
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    # We need to apply the hook before the last pooling layer and after all the convolutional layers
    
    def forward(self, x):
        x_internal = self.block1(x) 
        x_internal = torch.flatten(x_internal)
        y_pred = self.block2(x_internal)
        return y_pred
   
def onehot(c):
    #Vector is one hot encoded with v1 = negative , v2 = positive
    v = torch.zeros(2)
    v[c] = 1
    return v



   
   

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
transform = transforms.Compose([transforms.ToTensor()])
#tensor_frog = transform(frog)



# TRAIN LOOP
# Images can be found here: "https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs?select=images_as_pngs"
num_steps = len(data_loader) # Need to implement the train loader depending on what the data is
for epoch in range(num_epochs):
    total_loss = 0
    for i, (filepaths, c) in enumerate(data_loader):
        #images = images.to(device)
        #labels = labels.to(device)
        
        avg_batch_loss = 0
        for j, filepath in enumerate(filepaths):
            #print(filepath)
            # Image format is patientID_imageID.png
            current_mammo = Image.open(filepath)
            image = np.array(current_mammo).shape
            
            
        
            image_tensor = transform(current_mammo)
            #print(image_tensor.shape)
            # Forward pass
            output = model(image_tensor)
            loss = criterion(output, onehot(c[j]))
            avg_batch_loss += loss / len(filepath)
        total_loss += avg_batch_loss
        
        # Forward pass
        #outputs = model(images)
        #loss = criterion(outputs, labels)
        
        if i > 1000:
            break
        elif i % 200 == 0:
            print(f'Step: {i}, Batch Loss: {avg_batch_loss}')
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_loss = total_loss / 1000
    print(f'epoch: {epoch}, loss: {total_loss}')
        
        
# Evaluate the model
with network.eval():
    total_correct = 0
    total_guessed = 0
    
    class_correct = [0 for i in range(10)]
    class_total = [0 for i in range(10)]
    
    for images, labels in test_loader:
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        total_guessed += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            prediction = predicted[i]
            if (label == predicted):
                class_correct[label] += 1
            class_total[label] += 1
            
    accuracy = 100 * total_correct / total_guessed # Accuracy of the total network
    


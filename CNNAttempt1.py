# This is meant to be the CNN for Breast Cancer Detection competition run by Kaggle and the Radiological Society of North America
#   as an extension GradCAM is implemented on the dataset such that we can see which parts of the image trigger a positive cancer detection

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
        # We only have 29,425 of the images  (total dataset is around 57,000)
        patient_image_ids = self.all_meta_data.iloc[:29425, 1:3] # Columns for patient ID, imageID, left "L" or right "R" breast
        cancer_labels  = self.all_meta_data.iloc[:29425,6]
        
        # Now convert the incoming data into tensors:
        self.all_x = torch.tensor(patient_image_ids.to_numpy(), dtype=torch.int)
        self.all_y = torch.tensor(cancer_labels.to_numpy(), dtype=torch.long) # since the probability output wil be in long format
        
        # And partition the data into testing and training sets:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.all_x, self.all_y, test_size=0.2)
        self.n_samples = self.x_train.shape[0]
        
        #print(self.x_train)
        #print(self.y_train.shape)
        
    def __getitem__(self, index):
        # Return patient ID, image ID, cancer bool for the training set
        patientID = self.x_train[index, 0].numpy()
        imageID = self.x_train[index, 1].numpy()
        #print(str(patientID))
        filename = "Data/" + str(patientID) + "_" + str(imageID) + ".png"
        return filename, self.y_train[index]
    
    def __len__(self):
        return self.n_samples
    
    def test_sets(self):
        return self.x_test.numpy(), self.y_test.numpy()
    

# hyper parameters
num_epochs = 1
batch_size = 128
learning_rate = 0.0001

# Pre-training stuff
classes = (1, 0) # this is wrong, change this later once we load in the data

# Create the training data loader:
meta_dataset = PatientMetaData()
data_loader  = DataLoader(dataset=meta_dataset, batch_size = batch_size, shuffle=True)

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
            nn.Linear(1000000, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    # We need to apply the hook before the last pooling layer and after all the convolutional layers
    
    # Code for the hook method of GradCAM is from:
    #   https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
    
    
    def forward(self, x):
        x_internal = self.block1(x)
        # Where the hook goes
        h = x_internal.register_hook(self.activations_hook)
        #print(x_internal.shape)
        
        
        x_internal = x_internal.view(-1, 1000000)
        #print(x_internal.shape)
        
        y_pred = self.block2(x_internal)
        #print(y_pred)
        return y_pred
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.block1(x)
   


def onehot(c):
    #Vector is one hot encoded with v_1 = negative , v_2 = positive
    v = torch.zeros(2)
    v[c] = 1
    return v



   
############## TRAINING ######################
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
transform = transforms.Compose([transforms.ToTensor()])



# TRAIN LOOP
# Images can be found here: "https://www.kaggle.com/datasets/radek1/rsna-mammography-images-as-pngs?select=images_as_pngs"
num_steps = len(data_loader) # Need to implement the train loader depending on what the data is
#print(num_steps)
for epoch in range(num_epochs):
    total_loss = 0
    num_iters = 0
    for i, (filepaths, c) in enumerate(data_loader):
        image_tensors = torch.empty([len(filepaths), 512, 512]) # len(filepaths) usually = batch size, but not when we reach the end of the data
        for j, filepath in enumerate(filepaths):
            # Forward pass:
            # Image format is patientID_imageID.png
            current_mammo = Image.open(filepath)
            image_tensor = transform(current_mammo).squeeze()
            image_tensors[j] = image_tensor
            
        # Put image tensors in format batch, channels, height, width
        image_tensors = image_tensors.unsqueeze(1).to(device)
        #c = c.to(device) # don't need to do this
        outputs = model(image_tensors)
        #print(outputs) # both are tensors
        #print(c)
        loss = criterion(outputs, c)
        num_iters += 1
        total_loss += loss.detach()
        
        
        if i % 2 == 0:
            print(f'Step: {i}, Batch Loss: {loss}')
        if i > 40:
           break
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    total_loss = total_loss / num_iters
    print(f'epoch: {epoch}, loss: {total_loss}')
        
   
   
################# GRADCAM ####################   

model.eval()

# This is an example of a positive mammo:
patientID = '236'
imageID = '1531879119'
path = "Data/" + patientID + "_" + imageID + ".png"
image = Image.open(filepath)
image_tensor = transform(image).unsqueeze(0)

pred = model(image_tensor).squeeze() # This is the channel for positive value of cancer (takes argument dim=1 in the reference
print(pred)
#pred = pred.argmax(dim=1, requires_grad=True)
#print(pred)

pred[1].backward()
gradients = model.get_activations_gradient()
pooled_gradients = torch.mean(gradients, dim = 0)

activations = model.get_activations(image_tensor).detach()

for i in range(16): #our images are 512 x 512, but the pooled imagesare 16
    activations[:, i] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
print(heatmap)
plt.matshow(heatmap.squeeze())
plt.show()

import cv2

for i in range(29425):
    img = cv2.imread(Dataset[i])
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint(512 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('./map.jpg', superimposed_img)



################### TESTING #####################
# Return patient ID, image ID, cancer bool for the training set
#patientID = self.x_train[index, 0].numpy()
#imageID = self.x_train[index, 1].numpy()
#print(str(patientID))
#filename = "Data/" + str(patientID) + "_" + str(imageID) + ".png"
#return filename, self.y_train[index]
x_test, y_test = PatientMetaData.test_sets()
with model.eval():
    total_correct = 0
    total_guessed = 0
    
    class_correct = [0 for i in range(2)]
    class_total = [0 for i in range(2)]
    
    for i, (patientID, imageID) in enumerate(x_test):
        filename = "Data/" + str(patientID) + "_" + str(imageID) + ".png"
        
        # Forward pass:
        current_mammo = Image.open(filepath)
        image_tensor = transform(current_mammo).squeeze()
        image_tensors[i] = image_tensor
        label = y_test[i]
            
            
        # Put image tensors in format batch, channels, height, width
        image_tensors = image_tensors.unsqueeze(1).to(device)
        outputs = model(image_tensors)
        
        _, predicted = torch.max(outputs, 1)
        total_guessed += labels.size(0)
        total_correct += (predicted == label).sum().item()
        
        for i in range(batch_size):
            #label = labels[i]
            prediction = predicted[i]
            if (label == predicted):
                class_correct[label] += 1
            class_total[label] += 1
            
    accuracy = 100 * total_correct / total_guessed # Accuracy of the total network
    


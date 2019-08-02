#libraries
from os.path import isdir
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
from PIL import Image


#functions
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", action = 'store', default="/home/workspace/ImageClassifier/flowers", type=str, help="data directory of training and testing data")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="set directory to save checkpoints")
    parser.add_argument("--arch", type=str, action='store', dest = 'pretrained_model', default = 'vgg19')
    parser.add_argument("--learning_rate", type=float, default="0.01", help="learning rate")
    parser.add_argument("--epochs", type=int, default="20", help="number of pochs to train model")
    parser.add_argument("--hidden_units", type=int, default="512", help="number of hidden layers")
    parser.add_argument("--drop", type=float, default="0.5", help="enter drop size")
    parser.add_argument("--gpu", type=bool, default=True, help="use GPU or CPU to train model")
    args = parser.parse_args()
 
    return args
#functions to performs train/validation/test transformations on a dataset
def process(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])])
    #Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    #Defining the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
    class_to_idx = train_data.class_to_idx
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader, class_to_idx
#Function which makes decision on using GPU or CPU
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
        
    return device
#Initial classifier
def initial_classifier(model, input_features, hidden_units, drop):
    if hidden_units is None: 
        hidden_units = 512
        print("Number of hidden layers: 512.")
   
    #Defining classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p = drop)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim = 1))]))
    return classifier
#Function for training the model
def network_trainer(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps):
    epochs = 0
    steps = 0
    running_loss = 0
    print_every = 30
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                    
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        valid_loss = criterion(outputs, labels)
                        
                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()  
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                f"Accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                model.train()
                
                
    return model
#Function which validate model on test data images
def validate_model_on_testing_data(model, testloader, device):
    total = 0
    correct = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            



 
#Function "main" takes all functions and executes them
def main():
    print('inputing')
    input = arg_parser()
    
    save_dir = input.save_dir
    data_dir = input.data_directory
    pretrained_model = input.pretrained_model
    learning_rate = input.learning_rate
    hidden_units = input.hidden_units
    epochs = input.epochs
    drop = input.drop
    gpu = input.gpu
    #loading and processing images
    print('loading images')
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data, valid_data, test_data, trainloader, validloader, testloader, index = process(train_dir, valid_dir, test_dir)
    
    #loading predefined model
    print('loading predefined model')
    model = getattr(models, pretrained_model)(pretrained=True)
    
    #freezing parameters
    print('freezing parameters')
    for param in model.parameters():
        param.requires_grad = False
    #checking for GPU
    print('checking for gpu')
    device = check_gpu(gpu_arg=input.gpu);
    
    
    # Send model to device
    model.to(device);
    
    #sending model to device
    print('sending model to device')
    print(device)
    model.to(device);
    
    #criterion and optimizer
    input_features = model.classifier[0].in_features
    classifier = initial_classifier(model, input_features, input.hidden_units, input.drop)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #training model
    print('training model')
    print_every = 30
    steps = 0
    
    network_trainer(model, trainloader, validloader, device, criterion, optimizer, epochs, steps, print_every)
    #accuracy
    test_loss = validate_model_on_testing_data(model, testloader, device)
    print("Accuracy on test data:")
    print(test_loss)
    #saving checkpoint
    print('saving checkpoint')
    
    model.class_to_idx = index

    checkpoint = {'input_features': input_features,
                  'hidden_units': hidden_units,
                  'drop': drop,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'optimizer': optimizer,
                  'arch': pretrained_model,
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.classifier.state_dict()}
    torch.save(checkpoint, save_dir)
    
if __name__ == '__main__': 
    main()
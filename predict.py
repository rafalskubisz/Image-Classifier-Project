#importing libraries
from train import check_gpu

import numpy as np
import argparse
import json
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def arg_parser():
  
    parser = argparse.ArgumentParser()

    #checkpoint from train.py
    
    parser.add_argument("--checkpoint", type=str, default='checkpoint.pth', help='path of your saved model')
    
    #passing image for prediction
    parser.add_argument("--image_path", type=str, default="/home/workspace/ImageClassifier/flowers/test/102/image_08012.jpg")
    
    #returning top K most likely classes
    parser.add_argument("--topk", default="3", type=int)
    
    #importing category names
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help='Mapping of categories to real names.')

    #Using for inference
    parser.add_argument("--gpu", type=bool, default=True, help="use GPU or CPU to train model")
 
    args = parser.parse_args()
 
    return args


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load("checkpoint.pth")
    
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.name = "vgg19"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['arch'])
        model.name = checkpoint['arch']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    
    
    return model
                                    
def process_image(image_path):

    size = 256, 256
    crop_size = 224
    
    img = Image.open(image_path)
    
    img.thumbnail(size)

    left = (size[0] - crop_size)/2
    top = (size[1] - crop_size)/2
    right = (left + crop_size)
    bottom = (top + crop_size)

    img = img.crop((left, top, right, bottom))
    
    np_image = np.array(img)
    np_image = np_image/255
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    np_image = (np_image - means) / stds
    
    pyt_np_image = np_image.transpose(2,0,1)
    
    return pyt_np_image



def predict(image_path, model, topk):

    # Loading model
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    pyt_np_image = process_image(image_path)
    # Converting to torch tensor from Numpy array
    image_tensor = torch.from_numpy(pyt_np_image).type(torch.FloatTensor)
    
    image_unsqueezed = image_tensor.unsqueeze_(0)

    
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model.forward(image_unsqueezed)

    # Calculating probabilities
    probs = torch.exp(output)
    probs_top, index_top = probs.topk(topk)
    
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    class_to_idx = loaded_model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}
    classes_top_list = []
    
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list

def display_preds(image_path, model, topk):
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(image_path, model, topk)
    #Converting classes to names
    names = []
    for i in classes:
        names += [cat_to_name[i]]
    
    image = Image.open(image_path)
    f, ax = plt.subplots(2,figsize = (4,7))
    ax[0].imshow(image)
    ax[0].set_title(names[0])
    y_names = np.arange(len(names))
    ax[1].set_yticklabels(names)
    ax[1].set_yticks(y_names)
    
    ax[1].invert_yaxis() 
    ax[1].barh(y_names, probs, color='blue')
    plt.show()
    

    
    
def main():
    
    input = arg_parser()
    
    #loading categories to names
    with open(input.category_names, 'r') as f:
        cat_to_name = json.load(f)
            
    
    print('checking for gpu')
    device = check_gpu(gpu_arg=input.gpu)
    
    
    model = load_checkpoint(input.checkpoint)
   
    image_tensor = process_image(input.image_path)
    
    probs_top_list, classes_top_list = predict(input.image_path, model, input.topk)
    
    print(probs, classes)
    
    return probs, classes

if __name__ == '__main__':
    main()
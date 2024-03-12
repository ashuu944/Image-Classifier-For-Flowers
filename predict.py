import argparse
import json
import torch
from torch import nn
from torchvision import models, transforms
from collections import OrderedDict
import numpy as np
from PIL import Image

def arg_parser():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("--arch", default="VGG", help="Architecture of the prediction (default: VGG)")
    parser.add_argument("--image_path", help="Path to the input image")
    parser.add_argument("--checkpoint",default="model_checkpoint.pth", help="Path to the checkpoint file")
    parser.add_argument("--topk", type=int, default=5, help="Top K classes to display (default: 5)")
    parser.add_argument("--category_names", help="Path to the JSON file mapping class indices to category names")
    parser.add_argument("--device", default="cuda", help="Device for prediction (default: cuda)")  
    args = parser.parse_args()
    return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Open the image using PIL
    img = Image.open(image)
    
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply the transformations
    img = preprocess(img)
    
    # Convert to NumPy array
    np_image = np.array(img)
    
    return np_image

def load_checkpoint(file_path, arch='VGG'):
    # Load the checkpoint file
    checkpoint = torch.load(file_path)
    
    # Load the pre-trained model based on the selected architecture
    if arch == 'VGG':
        model = models.vgg16(pretrained=True)
        input_layer = 25088  # input layer size for VGG
    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        input_layer = 1024  #  input layer size for Densenet
    else:
        raise ValueError(f"Unsupported architecture: {arch} Select either VGG or Densenet ")

    # Freeze parameters to avoid backpropagation during inference
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the classifier based on the saved parameters
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_layer, checkpoint['hidden_layer'])),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=checkpoint['dropout'])),
        ('fc2', nn.Linear(checkpoint['hidden_layer'], checkpoint['hidden_layer'])),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=checkpoint['dropout'])),
        ('fc3', nn.Linear(checkpoint['hidden_layer'], len(checkpoint['class_to_idx']))),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    # Load the saved model state and class_to_idx
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model,cat_to_name=None, topk=5, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Process the image
    img = process_image(image_path)
    
    # Convert NumPy array to PyTorch tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()
    
    # Move the tensor to the appropriate device (CPU or GPU)
    img_tensor = img_tensor.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Calculate probabilities and top classes
    probs, classes = torch.topk(torch.softmax(output, dim=1), topk)
    
    # Convert tensor results to NumPy arrays
    probs = probs.cpu().numpy().squeeze()
    classes = classes.cpu().numpy().squeeze()

        # Reverse the class_to_idx mapping
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Get class labels from indices
    class_labels = [cat_to_name[idx_to_class[idx]] for idx in classes]
    
    return probs, classes, class_labels
    
    return probs, classes

def main():
    # Get Keyword Args for Training
    args = arg_parser()

    image_path = args.image_path

    # Load the checkpoint model 
    model = load_checkpoint(args.checkpoint,args.arch)
    
    #Choosing the appropriate device
    selected_device = torch.device('cuda' if torch.cuda.is_available() and args.device else 'cpu')

    # Loading JSON file if provided, else load default file name
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f,strict=False)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f,strict=False)
    

    # Perform prediction
    probs, classes, class_labels= predict(image_path, model,cat_to_name,topk=args.topk,device=selected_device)
        # Print the top classes and their corresponding class names
    for prob, cls,lbl in zip(probs, classes,class_labels):
        print(f"Class: {cls}, Flower Name: {lbl}, Probability: {prob}")
        
if __name__ == '__main__': main()

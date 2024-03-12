import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def arg_parser():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Train a new network on a flower dataset")
    parser.add_argument("--data_dir", default="flowers", help="The directory of the dataset (default: flowers)")
    parser.add_argument("--arch", default="VGG", help="Architecture of the pre-trained model (default: VGG)")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier (default: 512)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability in the classifier (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument("--device", default="cuda", help="Device for training (default: cuda)")
    args = parser.parse_args()
    return args

def load_data(data_dir='flowers'):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Define data transformations for training and validation
    training_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return train_loader, valid_loader

def build_model(arch='VGG', hidden_units=512, dropout=0.5):

    if arch == 'VGG':
        # Load a pre-trained model
        model = models.vgg16(pretrained=True)

        # Freeze parameters to avoid backpropagation during training
        for param in model.parameters():
            param.requires_grad = False
        
        # Modify the classifier
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        # Freeze parameters to avoid backpropagation during training
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    else:
        print("Unsupported architecture,please choose VGG or Densenet")
    
    model.classifier = classifier
    return model

def train_model(model, train_loader, valid_loader, epochs=5, lr=0.001, device='cuda',arch='VGG',hidden_units=512, dropout=0.5):

    # Additional information
    PATIENCE = 3  # Number of epochs to wait for improvement before early stopping
    best_loss = float('inf')
    no_improvement_count = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device)

    # Train loop
    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            accuracy = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                        
                val_loss += batch_loss.item()
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            # Average validation loss
            val_loss /= len(valid_loader)

            # Check for early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Training loss: {running_loss/len(train_loader):.3f}.. "
                f"Validation loss: {val_loss:.3f}.. "
                f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

            running_loss = 0

            if no_improvement_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} as there is no improvement in validation loss.")
                break

        # Save the trained model to a checkpoint
    checkpoint = {
            'arch': arch,
            'input_layer': 25088,
            'hidden_layer': hidden_units,
            'dropout': dropout,
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'class_to_idx': train_loader.dataset.class_to_idx,
            'optimizer_dict':optimizer.state_dict()
             }
    path = f'model_checkpoint_{arch.lower()}.pth'
    torch.save(checkpoint, path)    

def main():

    # Get Keyword Args for Training
    args = arg_parser()
     
    # Load data
    train_loader, valid_loader = load_data(data_dir=args.data_dir)

    # Build model
    model = build_model(arch=args.arch, hidden_units=args.hidden_units, dropout=args.dropout)
    
    #select the apropriate device
    selected_device = torch.device('cuda' if torch.cuda.is_available() and args.device else 'cpu')

    # Train model and save the check point
    train_model(model, train_loader, valid_loader, epochs=args.epochs, lr=args.lr, device=selected_device,arch=args.arch,hidden_units=args.hidden_units, dropout=args.dropout)

if __name__ == '__main__': main()




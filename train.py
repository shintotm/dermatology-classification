import torch
import numpy as np
import argparse
from torchvision import datasets, transforms, models
from torch import nn
from collections import OrderedDict
from torch import optim


def get_input():
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description="Training settings")
    
    parser.add_argument('image_dir', action='store', help='path to training data')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    in_args = parser.parse_args()
    
    return in_args

def create_model():
    model = models.vgg19(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias = True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p = 0.5)),
                          ('fc2', nn.Linear(4096, 3, bias = True)),
                          ('output', nn.LogSoftmax(dim = 1))
                           ]))


    model.classifier = classifier
    return model

def get_dataloaders(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_set_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    validation_set_transforms = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
    test_set_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])





    train_dataset = datasets.ImageFolder(train_dir, transform = train_set_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = validation_set_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_set_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)  
    
    return train_loader, valid_loader, test_loader

def validation(model,data_loader, criterion, device):
    loss = 0
    accuracy = 0
    
    for inputs,labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        loss += criterion(output, labels).item()

        output = torch.exp(output)
        check_equality = (labels.data == output.max(dim = 1)[1])
        accuracy += check_equality.type(torch.FloatTensor).mean()
        
    return loss, accuracy

def train(model, epochs, optimizer, criterion, train_loader, valid_loader, device):
    running_loss = 0.0
    steps = 0
    print_every = 10
    
    print(f'starting training using device: {device}...')
    for e in range(epochs):
        model.train()
        
        for inputs, labels in train_loader:
            steps += 1
            inputs , labels  = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader,criterion, device)

                print("epoch: {}/{} :".format(e+1, epochs),
                      "Train loss : {:.4f} ".format(running_loss/print_every),
                      "val loss: {:.4f} ".format(valid_loss/len(valid_loader)),
                      "val acc: {:.4f} ".format(accuracy/len(valid_loader))) 

                running_loss = 0
                model.train()    
    
    return model


def main():
    
    in_args = get_input()
    print('image_dir: ', in_args.image_dir)
    print('epochs:', in_args.epochs)
    train_loader, valid_loader, test_loader = get_dataloaders(in_args.image_dir)
    print('len of train_loader', len(train_loader))
    print('len of valid_loader', len(valid_loader))
    print('len of test_loader', len(test_loader))
    
    model = create_model()
    
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    epochs = in_args.epochs
    model = train(model, epochs, optimizer, criterion, train_loader, valid_loader, device)
    
    test_loss, test_acc = validation(model, test_loader,criterion, device) 
    print("test_loss: {:.4f}".format(test_loss/len(test_loader)),
          ", test_acc: {:.4f}".format(test_acc/len(test_loader)))

    
    
if __name__ == "__main__":
    main()
                        

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
import pickle
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
import torch.optim as optim
from torchvision import models
from sklearn import svm, preprocessing
from sklearn.metrics import f1_score
import argparse


np.random.seed(0)
torch.manual_seed(0)

class KaggleDataset(Dataset):

  def __init__(self, x_file, y_file, transform=None, device='cuda'):
        """
        Args:
            x_file : Path to the pkl file of training images.
            y_file : Path to the pkl file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(x_file, 'rb') as f:
          x_train = pickle.load(f)


        with open(y_file, 'rb') as f:
          y_train = pickle.load(f)
          self.le = preprocessing.LabelEncoder()
          self.le.fit(list(set(y_train)))
          y_train = F.one_hot(torch.tensor(self.le.transform(y_train)), num_classes=-1)


        self.X_Train = x_train
        self.Y_Train = y_train
        self.transform = transform

  def __len__(self):
      return len(self.X_Train)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      x = self.X_Train[idx]
      y = self.Y_Train[idx]

      if self.transform:
          x = self.transform(Image.fromarray(x))
          x = x.type(torch.FloatTensor)

      return x, y


def main():

  epochs = config.epoch
  mode = config.mode
  batch_size = config.batch_size
  model_type = config.model_type
  init_path = config.init_path
  
  # loading the data
    
  path_x = init_path + '/data/x_train.pkl'
  path_y = init_path + 'data/y_train.pkl'

  transform_train = transforms.Compose(
      [transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor(),
      transforms.Resize(96),
      ])

  transform_test = transforms.Compose(
      [transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor(),
      transforms.Resize(110),
      transforms.TenCrop(96),
      transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
      ])


  train_valid_set = KaggleDataset(path_x, path_y, transform=transform_train)
  nums = [int(len(train_valid_set)*0.8),  len(train_valid_set) - int(len(train_valid_set)*0.8)]
  trainset, validset = torch.utils.data.dataset.random_split(train_valid_set, nums)
          

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
  validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

  classes_names = {0: 'big_cats', 1: 'butterfly', 2: 'cat', 3: 'chicken', 4: 'cow', 5: 'dog', 6: 'elephant', 7: 'goat', 8: 'horse', 9: 'spider', 10: 'squirrel'}
  classes = list(classes_names.values())
    
  
  # building the models

  model_types = ['resnet18', 'efficientnet_b1', 'efficientnet_b7', 'mobilenet_v2']
  model_list = []



  if 'resnet18' in model_types:
      model = models.resnet18()
      model.fc = nn.Linear(512, 11)
      model_list.append(model)


  if 'efficientnet_b1' in model_types:
      model  = models.efficientnet_b1()
      model.classifier = nn.Linear(1280, 11)
      model_list.append(model)

  if 'efficientnet_b7' in model_types:
      model  = models.efficientnet_b7()
      model.classifier = nn.Linear(2560, 11)
      model_list.append(model)


  if 'mobilenet_v2' in model_types:
      model = models.mobilenet_v2()
      model.classifier[1]=nn.Linear(1280, 11)
      model_list.append(model)


  for i in range(len(model_list)):
    model_list[i] = model_list[i].to('cuda')
    model_list[i].requires_grad_(True)

  criterion = nn.CrossEntropyLoss(reduction='mean').to('cuda')

  if mode == 'test':
    # loading the trained model
    model_paths = ['/trained_models/Cls-resnet18-0.0003-400-32-1.0-(21-12-08-976563).pkl',
                  '/trained_models/Cls-efficientnet_b1-0.0003-400-32-1.0-(21-12-08-976224).pkl',
                  '/trained_models/Cls-efficientnet_b7-0.0001-400-32-1.0-(21-12-06-780261).pkl',
                  '/trained_models/Cls-mobilenet_v2-0.0003-400-32-1.0-(21-12-08-566914).pkl']

    for i in range(len(model_list)):
      path = init_path + model_paths[i]
      model_list[i].load_state_dict(torch.load(path, map_location='cuda'))

    print(test(model_list, validloader))

  elif mode == 'train':
    if model_type == 'resnet18':
      train(model_list[0], epochs, model_paths[0], trainloader, validloader, criterion)
    elif model_type == 'efficientnet_b1':
      train(model_list[1], epochs, model_paths[1], trainloader, validloader, criterion)
    elif model_type == 'efficientnet_b7':
      train(model_list[2], epochs, model_paths[2], trainloader, validloader, criterion)
    elif model_type == 'mobilenet_v2':
      train(model_list[3], epochs, model_paths[3], trainloader, validloader, criterion)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--init_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--model_type', type=str,
                        default='resnet18', help='efficientnet_b1/mobilenet_v2/resnet18/efficientnet_b7')
    parser.add_argument('--batch-size', type=int, default=18)
    
    config = parser.parse_args()
    main(config)


# training

def train(model, num_epochs, save_path, trainloader, validloader, criterion):

  model.train() 
  lr = 0.0001

  optimizer = optim.Adam([{'params': model.parameters(), 'lr':lr}])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  factor=0.5, patience=15)


  # keeping the loss after each epoch
  validation_losses = []
  training_losses = []
  best_valid_loss = np.inf
  best_model = model
  best_epoch = 1

  total_batches_train = len(trainloader)

  for epoch in range(num_epochs):


      validation_loss = validate(validloader, model, criterion)
      scheduler.step(validation_loss)
      if validation_loss < best_valid_loss:
        best_valid_loss = validation_loss
        best_model = model
        best_epoch = epoch + 1
      validation_losses = np.append(validation_losses, validation_loss)

      print('epoch: ' + str(epoch))
      running_loss = 0.0
      epoch_training_loss = 0.0

      for i, data in enumerate(trainloader, 0):

          # if i % 100 == 0:
          #   print(i)

          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          inputs = inputs.to('cuda')
          labels = labels.type(torch.FloatTensor)
          labels = labels.to('cuda')

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = torch.softmax(model(inputs), dim=1)
          outputs = outputs.to('cuda')
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 100 == 0:    # print every 100 mini-batches
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0

          # updating the overall training loss
          epoch_training_loss += loss.item()

      training_losses = np.append(training_losses, epoch_training_loss/total_batches_train)

  
  torch.save(best_model.state_dict(), save_path)
  print('Finished Training')



def validate(loader, model, criterion):
    total = 0
    correct = 0                                                                   
    running_loss = 0.0                                                                                     
    with torch.no_grad():

        for i, data in enumerate(loader, 0):
                               
            inputs, labels = data

            inputs = inputs.to('cuda') 
            labels = labels.type(torch.FloatTensor)                      
            labels = labels.to('cuda')                        
            
            outputs = torch.softmax(model(inputs), dim=1)
                                    
            loss = criterion(outputs, labels)                 
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels, axis=1)
            total += len(labels)                           
            correct += (predicted == labels).sum().item()     
            running_loss = running_loss + loss.item()         
    
    mean_val_accuracy = 100 * correct / total              
    mean_val_loss = running_loss / len(loader) 

    print('Validation Accuracy: %d %%' % (mean_val_accuracy)) 
    print('Validation Loss: ', mean_val_loss)

    return mean_val_loss


def test(model_list, testloader):

  predictions = []
  model_predictions = []
  targets = []
  
  model_list = [model.eval() for model in model_list]
  
  for i, data in enumerate(testloader, 0):

      inputs, labels = data

      # inputs = torch.transpose(inputs, 1, 0)

      inputs = inputs.to('cuda')

      outputs = []
      for model in model_list:
        model_outputs = torch.softmax(model(inputs), dim=1)
        outputs.append(model_outputs)
      outputs = np.sum([model_outputs for model_outputs in outputs])

      outputs = outputs.to('cpu')


      batch_predictions = torch.argmax(outputs, axis=1)
      
      predictions = np.append(predictions, batch_predictions)
      class_labels = torch.argmax(labels, axis=1)
      targets = np.append(targets, class_labels)


  return f1_score(targets, predictions, average='micro')



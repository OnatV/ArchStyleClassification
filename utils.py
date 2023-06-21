import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import sys
from torchvision.models.alexnet import alexnet
from torchvision.models.googlenet import googlenet
from torchvision.models.resnet import resnet50
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Images(Dataset):
    '''
    Image loading class compatiable with pytorch modules
    '''
    def __init__(self,dataset_path,transforms_=None, subset = 'train'):

        self.transform = transforms.Compose(transforms_)

        if subset == 'train':
            self.tset =  pd.read_csv("TrainingSet.csv")

        if subset == 'val':
            self.tset =  pd.read_csv("ValSet.csv")             
        
        if subset == 'test':
            self.tset =  pd.read_csv("TestSet.csv")

        self.classes = tuple(np.unique (self.tset["Classes(Folder Names)"]))
        
        
    def __getitem__(self, index):

        #Loads the image
        try : 
            im      = Image.open( self.tset['Images'][index % len(self.tset)] ).convert('RGB') 
            #Finds class number 0,1,...
            classs  = self.classes.index( self.tset["Classes(Folder Names)"][index % len(self.tset)] )
            # Creates a vector with a single one at class index eg. [0 1 0 0 ...]
            label   = np.eye(len(self.classes),dtype=np.float32)[classs] 
        except FileNotFoundError:
            print("Im not found", self.tset['Images'][index % len(self.tset)])    

        #label should be returned if MSE loss is going to be used
        return self.transform(im),classs         
    
    def __len__(self):
        return len(self.tset)




def test_train_divide(dataset_path,train_size = 0.7):
    '''
    This function is used to seperate the dataset into test validation and training sets
    '''

    training_set      = [] 
    test_set          = []
    val_set           = []
    classes_test      = []
    classes_training  = []
    classes_val       = []

    for dirpath , dirnames , filenames in os.walk(dataset_path):
                    
        if (dirnames == [])  & (filenames != []) : #If we reach to the bottom
            print('Current Path :' , dirpath)
            print()

            filenames = [dirpath + '/' + file for file in filenames]
            #We pick %70 instances from each class for training
            n_trainig = round(len(filenames)* train_size) 
            #We set test_size/2 of the samples as validation set 
            n_val = round(len(filenames)* (1-train_size)/3 )
            

            train =  random.sample(filenames, k=n_trainig)
            test_val  = list(set(filenames) - set(train))

            val =  random.sample(test_val, k=n_val)
            test = list(set(test_val) - set(val))

            classes_training.extend([dirpath] * len(train))
            classes_test.extend([dirpath] * len(test))
            classes_val.extend([dirpath] * len(val))

            training_set.extend(train)
            test_set.extend(test)
            val_set.extend(val)

    training_set_ = pd.DataFrame( {"Classes(Folder Names)": classes_training , "Images": training_set} )
    test_set_ = pd.DataFrame( {"Classes(Folder Names)": classes_test , "Images": test_set} )
    val_set_ = pd.DataFrame( {"Classes(Folder Names)": classes_val , "Images": val_set} )
    #Shuffling
    training_set_= training_set_.sample(frac=1).reset_index(drop=True)
    test_set_    = test_set_.sample(frac=1).reset_index(drop=True)
    val_set_     = val_set_.sample(frac=1).reset_index(drop=True)

    training_set_.to_csv("TrainingSet.csv", index = False)
    test_set_.to_csv("TestSet.csv", index = False)
    val_set_.to_csv("ValSet.csv", index = False)


def train_classifier(opt):
    '''
    Trains the classifier specified by the options. Prints the accuracy score on validation images at every epoch.
    '''
    trainloader = DataLoader(
        Images(opt.dataset_name, transforms_=opt.transforms_,subset = 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    valloader = DataLoader(
        Images(opt.dataset_name, transforms_=opt.transforms_,subset = 'val'),
        batch_size=100,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Model Choices
    if opt.net == 'alexnet':
        net = alexnet(pretrained = opt.use_pretrained)
        #Updating layers for transfer learning
        net.classifier[4] = nn.Linear(4096,1024)
        net.classifier[6] = nn.Linear(1024,opt.num_classes)       

    if opt.net == 'googlenet' :
        net = googlenet(pretrained = opt.use_pretrained)
        net.fc  = nn.Linear(1024,opt.num_classes)
       
    if opt.net == 'resnet' :
        net = resnet50(pretrained = opt.use_pretrained)
        net.fc  = nn.Linear(2048,opt.num_classes)
   
    #Optimizer Choices
    if opt.optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), opt.lr, momentum=opt.momentum)
    if opt.optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), opt.lr)
     
    if opt.continue_train:
        checkpoint = torch.load(opt.modelloadPath)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  

        #This should be commented out if the loaded optimizer is not Adam
        #But the training will be continued by Adam optimizer 
        if opt.optimizer_type == 'Adam':              
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    #Loss Choices
    if opt.loss_type == 'MSE':
        criterion = nn.MSELoss()
    if opt.loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()

    net.train()
    net.to(device)

        
    for epoch in range(opt.start_epoch, opt.end_epoch):
        total_loss = 0
        for i, batch in enumerate(trainloader):
            
            # batch is a tuple of (images , labels)
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss %f]"
                % (
                    epoch,
                    opt.end_epoch,
                    i,
                    len(trainloader),
                    total_loss,                  
                )
            )
        #Checking score on the validation set
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in valloader:
                images, correct_labels = batch[0].to(device), batch[1].to(device)
                outputs = net(images)

                if opt.loss_type == 'MSE':
                  _, correct_labels = torch.max(correct_labels,1)
                if opt.loss_type == 'CrossEntropy':
                  pass

                _, predicted   = torch.max(outputs.data, 1)
                total += correct_labels.size(0)
                correct += (predicted == correct_labels).sum().item()
        print('Accuracy of the network on the validation images: ', correct / total)

    print('Finished Training of ' + opt.net)   

    torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,            
            }, opt.modelsavePath)


def test_classifier(opt):
    '''
    Runs test on the classifier specified in the options.
    '''
    print('Now testing')
    testloader = DataLoader(
    Images(opt.dataset_name, transforms_=opt.transforms_,subset = 'test'),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
    )
    
    if opt.net == 'alexnet':
        net = alexnet()
        net.classifier[4] = nn.Linear(4096,1024)
        net.classifier[6] = nn.Linear(1024,opt.num_classes)

    if opt.net == 'googlenet' :
        net = googlenet(pretrained= True)
        net.fc  = nn.Linear(1024,opt.num_classes)
    
    if opt.net == 'resnet' :
        net = resnet50(pretrained= True)
        net.fc  = nn.Linear(2048,opt.num_classes)    
    

    checkpoint = torch.load(opt.modelsavePath)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    #Testing Accuracy
    correct = 0
    total = 0
    correct_labels_list= []
    predicted_labels_list = []

    with torch.no_grad():
        for batch in testloader:
            images, correct_labels = batch[0].to(device), batch[1].to(device)

            outputs = net(images)
            
            if opt.loss_type == 'MSE':
              _, correct_labels = torch.max(correct_labels,1)
            if opt.loss_type == 'CrossEntropy':
              pass
              
            _, predicted   = torch.max(outputs.data, 1)
            total += correct_labels.size(0)
            correct += (predicted == correct_labels).sum().item()
            predicted_labels_list.extend(list(np.array(predicted.cpu())))
            correct_labels_list.extend(list(np.array(correct_labels.cpu())))
            

    print('Accuracy of the network on the test images: %f %%' % (
        100 * correct / total))
    conf_matrix = find_conf_matrix(correct_labels_list,predicted_labels_list)
    return conf_matrix


def find_conf_matrix(y_test,y_pred):
    
    classnames = np.unique(y_test)
    conf_matrix_size = ( len(classnames),  len(classnames) )
    conf_matrix = pd.DataFrame( np.zeros( conf_matrix_size ), index = classnames, columns= classnames )

    for i in range(len(y_pred)):
        conf_matrix[y_pred[i]][y_test[i]] += 1

    # print("Accuracy: " ,sum(y_pred==y_test) /len(y_pred))  

    conf_matrix /= conf_matrix.to_numpy().sum() #Normalization
    return conf_matrix
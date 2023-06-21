import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.models.resnet import resnet50
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

'''
This file is for SVM + ResNet50 Approach. One needs to have a trained model to run this file.
'''

class options_():
    
    def __init__(self):
        self.num_classes = 25
        self.n_cpu       = 2
        self.batch_size  = 16
        self.epoch  = 50
        self.dataset_name =   "arcDataset"
        self.net            = 'resnet'         
        self.transforms_ = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]       
        self.modelloadPath = "model_" + self.net + "epoch_" + str(self.epoch) + "batch_" + str(self.batch_size) + ".pth"

class Images(Dataset):

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
        except FileNotFoundError:
            print("Im not found", self.tset['Images'][index % len(self.tset)])         
        # Creates a vector with a single one at class index eg. [0 1 0 0 ...]
        label   = np.eye(len(self.classes),dtype=np.float32)[classs]                           

        

        return self.transform(im),classs 
    def __len__(self):
        return len(self.tset)

def extract_features(opt,mode = 'train'):

    #Loads one of the train-val-test sets according to the mode specified
    loader = DataLoader(
        Images(opt.dataset_name, transforms_=opt.transforms_,subset = mode),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    #Initilize net
    net = resnet50(pretrained = True) #This is set to true because loading the model is faster than initializng its weights
    net.fc  = nn.Linear(2048,opt.num_classes)

    #Loads the pretrained model
    checkpoint = torch.load(opt.modelloadPath)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    device = torch.device("cpu")

    net.to(device)
    y_set = []
    x_set = []
    with torch.no_grad():
        for batch in loader:
            
            # batch is a tuple of (images , labels)
            x, labels = batch[0].to(device), batch[1].to(device)

            #We pass each input from the net and extract the feature before the last fc layer
            #This is coded for the layer arcitecture of Resnet 
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)

            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)

            x = net.avgpool(x)
            x = torch.flatten(x, 1)
            
            x_set.extend(list(np.array(x)))
            y_set.extend(list(np.array(labels)))
               
    features = pd.DataFrame({'Features':x_set,'Labels':y_set})
    features.to_pickle("features_" + mode + ".pickle")


def process_dataset_SVC(parameters):

    f_train = pd.read_pickle("features_train.pickle")
    f_val = pd.read_pickle("features_val.pickle")
    f_test = pd.read_pickle("features_test.pickle")

    X_train = np.array(list( f_train['Features']) + list(f_val['Features'] ) )
    y_train = np.array(list( f_train['Labels'])   + list(f_val['Labels']   ) )

    X_test =  np.array(list( f_test['Features'] ) )
    y_test =  np.array(list( f_test['Labels']   ) )

    del f_train, f_val, f_test
    ###################################Finds the Best Parameters for the Classifier###################################
    
    svc = SVC(gamma = 'auto')
    clf = GridSearchCV(svc, parameters,cv = 5,n_jobs=2)                              #5-fold cross_validaiton
    clf.fit(X_train,y_train)
    print(pd.DataFrame(clf.cv_results_))
    print("Best parameters: ",clf.best_params_)
    print("Best Validation Score:",clf.best_score_)

    #######################################Trains Classifier with Optimum Values#######################################
    svc = SVC(gamma = 'auto', C = clf.best_params_['C'] , kernel = clf.best_params_['kernel'])
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)      
    
    return find_conf_matrix(y_test,y_pred)

def find_conf_matrix(y_test,y_pred):
    
    classnames = np.unique(y_test)
    conf_matrix_size = ( len(classnames),  len(classnames) )
    conf_matrix = pd.DataFrame( np.zeros( conf_matrix_size ), index = classnames, columns= classnames )

    for i in range(len(y_pred)):
        conf_matrix[y_pred[i]][y_test[i]] += 1

    print("Accuracy: " ,sum(y_pred==y_test) /len(y_pred))  

    conf_matrix /= conf_matrix.to_numpy().sum() #Normalization
    return conf_matrix

opt = options_() 

if __name__ == '__main__':    
    for mode_ in ['train','test','val']:
        try:
            open("features_" + mode_ + ".pickle")
        except:
            extract_features(opt,mode = mode_)    

    parameters_svm = { 'C' : [0.3,0.5,0.7,1], 'kernel' : ('rbf','linear')}#Parameters to test for svm

    conf_matrix = process_dataset_SVC(parameters_svm)

    plt.imshow(conf_matrix, cmap ="RdYlBu")
    plt.colorbar()
    plt.xticks(range(len(conf_matrix)), conf_matrix.columns, rotation = 'vertical')
    plt.yticks(range(len(conf_matrix)), conf_matrix.index, rotation = 'horizontal')
    plt.xlabel("Correct Label")
    plt.ylabel("Predicted Label")
    plt.title(" Confusion Matrix")
    plt.show()
from utils import *
import matplotlib.pyplot as plt
class options_():
    
    def __init__(self):
        self.num_classes = 25
        self.n_cpu       = 2
        self.batch_size  = 16
        self.start_epoch  = 0
        self.end_epoch    = 5
        self.lr          = 0.001
        self.momentum    = 0.9
        self.dataset_name =   "arcDataset"
        self.optimizer_type = 'SGD'
        self.loss_type      = 'CrossEntropy'
        self.net            = 'resnet'
        self.continue_train = False
        self.use_pretrained = True         
        self.transforms_ = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.modelsavePath = "model_" + self.net + "epoch_" + str(self.end_epoch) + "batch_" + str(self.batch_size) + ".pth"
        self.modelloadPath = "model_" + self.net + "epoch_" + str(self.start_epoch) + "batch_" + str(self.batch_size) + ".pth"

opt = options_()

if __name__ == '__main__':
    try:
        open('TrainingSet.csv','r')
    except FileNotFoundError :
        print("Training Set not found. It will be created")
        test_train_divide(dataset_path = opt.dataset_name)

    train_classifier(opt)
    conf_matrix = test_classifier(opt)
    plt.imshow(conf_matrix, cmap ="RdYlBu")
    plt.colorbar()
    plt.xticks(range(len(conf_matrix)), conf_matrix.columns, rotation = 'vertical')
    plt.yticks(range(len(conf_matrix)), conf_matrix.index, rotation = 'horizontal')
    plt.xlabel("Correct Label")
    plt.ylabel("Predicted Label")
    plt.title(" Confusion Matrix")
    plt.show()









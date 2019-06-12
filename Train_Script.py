import torch
import torchvision  
import torch.nn as nn 
import numpy as np
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data 
import os 
import shutil 
import glob 
import torchvision.datasets as Data 
from Plot_accuracy import visualization,time_visualization
import torch.utils.data as TD
import time 
import pretrainedmodels as PM 
import inspect
from Shufflenet import ShuffleNetV1,ShuffleNetV2
from Nasnet.nasnet import NASNetAMobile
from Mobilenet import MobilenetV1,MobileNetV2
from Imagenet_loader import ImageFolder
from Caltech_loader import Caltech256
from Resnet_architectures import *
from data_aug import * 
import pdb 
from Resnet_architectures import * 

import os.path
import lmdb
#import caffe
from PIL import Image
import transforms as Image_Trans

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()

parser.add_argument('-lr','--learning_rate',default = 0.01,help='Specifies the initial learning rate for model',type=float)
parser.add_argument('-wd','--weight_decay',help='Specifies Weight decay (L2 regularization) for optimizing process')
parser.add_argument('-Imagenet','--Imagenet_data',default="./data/Imagenet",help='Specifies Weight decay (L2 regularization) for optimizing process')
parser.add_argument('-opt','--optimizer',default = 'adam',help='Specifies the optimizer type for training. Choose from [Adam,SGD,Adadelta,Adagrad,RMSProp,ASGD,Adamax]')
parser.add_argument('--test_interval',default = 1,help='Specifies the epoch interval after which to test the model')
parser.add_argument('--opt_policy',default = 'Plateau_LR',help='Specifies the learning rate schedule for training. Choose from [Step_LR , exp_LR ,Plateau_LR]')
parser.add_argument('--lr_step_size',default = 15,help='Specifies the step size for reducing learning rate [in epochs]',type=int)
parser.add_argument('--epochs',default = 150,help='Specifies the number of epochs to train the model for',type=int)
parser.add_argument('--exp_LR_gamma',default = 0.1,help='Specifies the decay rate for exponential LR policy',type=float)
parser.add_argument('--Plateau_patience',default = 3,help='Specifies the patience to be kept by the scheduler, while the loss does not decrease',type=int)
parser.add_argument('--batch_size',default = 48,help='Specifies the batch size of samples for forward pass',type=int)
parser.add_argument('--loss',default = 'cross_entropy',help='Specifies the batch size of samples to aggregate gradients over options = [Cross_entropy,BCE]')
parser.add_argument("--plot_accuracy",default=2,help="No of epochs after which to plot accuracies")

parser.add_argument("--method",default="Normal",help="Type of data augmentation to use.Select from [Normal,Mixup,Cutmix,Cutmix_heatmap,Cutoff]")
parser.add_argument("--Topk_size",default=6,type = int, help="Number of top patches to select from heatmaps")
parser.add_argument("--dataset",default="CIFAR10",help = "Specify the dataset to train depthnet on.")
parser.add_argument("--model",default="Resnet18",help="Type of model to train Messidor on")

args = parser.parse_args()

cutmix_2 = True if args.method == "Cutmix_heatmap" else False 
###################################################################################################
# Data Preprocessing

if args.model in ["Inceptionv3","Xception"]:
    if args.dataset == "Imagenet":
        img_transform  = transforms.Compose([Image_Trans.ToTensor(),Image_Trans.Scale((512,512)),Image_Trans.CenterCrop((299,299))])
    else:
        img_transform  = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])

elif args.model in ["squeezenet","Densenet121","Densenet169","Densenet201","Densenet264","VGG_19",
                     "Resnet18","Resnet50","Resnet34","Resnet101","Resnet152","ShufflenetV1",
                    "ShufflenetV2","MobilenetV1"]:
    if args.dataset == "Imagenet":    
        img_transform  = transforms.Compose([Image_Trans.RandomHorizontalFlip(),
                                            Image_Trans.ToTensor(),Image_Trans.Normalize(meanfile="data/Imagenet/imagenet_mean.binaryproto"),
                                            Image_Trans.CenterCrop((224,224))]) 
    elif args.dataset == "Caltech":
        img_transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        img_transform  = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) 

elif args.model in ["NASnet"]:
    if args.dataset == "Imagenet":
        img_transform = transforms.Compose([
                Image_Trans.ToTensor(),
                Image_Trans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                Image_Trans.CenterCrop((224,224))])
    else:
        img_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])       

elif args.model in ["MobilenetV2"]:
    if args.dataset == "Imagenet":
        img_transform = transforms.Compose([
                Image_Trans.ToTensor(),
                Image_Trans.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                Image_Trans.CenterCrop((224,224))])
    else:
        img_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
#########################################################################################3
# Loading Data

if args.dataset == "MNIST":   # Shape: (1,28,28)
    num_classes = 10
    Trainloader = Data.MNIST("data/MNIST",download=False,train=True,transform = img_transform)
    Testloader = Data.MNIST("data/MNIST",download=False,train=False,transform = img_transform)

elif args.dataset == "CIFAR10":  # Shape: (3,32,32)
    num_classes = 10
    Trainloader = Data.CIFAR10("data",download=False,train=True,transform = img_transform)
    Testloader = Data.CIFAR10("data",download=False,train=False,transform = img_transform)

elif args.dataset == "CIFAR100":  # Shape: (3,32,32)
    num_classes = 100
    Trainloader = Data.CIFAR100("data",download=False,train=True,transform = img_transform)
    Testloader = Data.CIFAR100("data",download=False,train=False,transform = img_transform)

elif args.dataset == "Fashion-MNIST":
    num_classes = 10
    Trainloader = Data.FashionMNIST("data/Fashion-MNIST",download=True,train=True,transform = img_transform)
    Testloader = Data.FashionMNIST("data/Fashion-MNIST",download=True,train=False,transform = img_transform)

elif args.dataset == "SVHN":
    num_classes = 10    
    Trainloader = Data.SVHN("data/SVHN",download=False,split="train",transform = img_transform)
    Testloader = Data.SVHN("data/SVHN",download=False,split="test",transform = img_transform)

elif args.dataset == "STL10":
    num_classes = 10
    Trainloader = Data.STL10("data",download=False,split="train",transform = img_transform)
    Testloader = Data.STL10("data",download=False,split="test",transform = img_transform)

elif args.dataset == "Imagenet":
    num_classes = 1000

    Trainloader = ImageFolder(data_path=args.Imagenet_data,transform=img_transform)
    Testloader = ImageFolder(data_path=args.Imagenet_data,transform=img_transform,Train=False)

elif args.dataset == "Caltech":
    num_classes = 257
    Trainloader = Caltech256("data/Caltech",train=True,transform=img_transform)
    Testloader = Caltech256("data/Caltech",train=False,transform=img_transform)

if args.dataset == "Imagenet":
    train_loader = torch.utils.data.DataLoader(dataset=Trainloader,
                                           batch_size=args.batch_size,shuffle=True,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=Testloader,
                                           batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=True)
else:
    train_loader = torch.utils.data.DataLoader(dataset=Trainloader,
                                           batch_size=args.batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=Testloader,
                                           batch_size=int(args.batch_size/2),shuffle=False)

print ('---------- Training and Test data Loaded ')

#############################################################################################################################
# Loading Model

if args.model == "Inceptionv3":
    model = PM.inceptionv3(num_classes = 1000,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = torch.DataParallel(model.to(device))  
    width,height = 299,299

elif args.model == "Xception":
    model = PM.xception(num_classes = 1000,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = model.to(device) 
    width,height = 299,299 

elif args.model == "VGG_19":
    model = PM.vgg19(num_classes = 1000)#,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = model.to(device) 
    width,height = 224,224

elif args.model == "Resnet18":

    model = resnet18(num_classes = num_classes,pretrained=False) 
    model = model.to(device) 
    width,height = 224,224

elif args.model == "Resnet50":

    model = resnet50(num_classes = num_classes,pretrained=False) 
    model = model.to(device) 
    width,height = 224,224

elif args.model == "Resnet101":

    model = resnet101(num_classes = num_classes,pretrained=False) 
    model = model.to(device) 
    width,height = 224,224

elif args.model == "Resnet152":

    model = resnet152(num_classes = num_classes,pretrained=False) 
    model = model.to(device) 
    width,height = 224,224

elif args.model == "Resnet34":

    model = resnet34(num_classes = num_classes,pretrained=False) 
    model = model.to(device) 
    width,height = 224,224

elif args.model == "squeezenet":
    model = squeezenet(num_classes=num_classes)
    model = model.to(device)
    width,height = 224,224

elif args.model == "Densenet121":
    model = PM.densenet121(num_classes = 1000,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = model.to(device)
    width,height = 224,224

elif args.model == "Densenet169":
    model = PM.densenet169(num_classes = 1000,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = model.to(device)
    width,height = 224,224

elif args.model == "Densenet201":
    model = PM.densenet201(num_classes = 1000,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = model.to(device)
    width,height = 224,224

elif args.model == "Densenet264":
    model = PM.densenet264(num_classes = 1000,pretrained=None)
    d = model.last_linear.in_features
    model.last_linear = nn.Linear(d, num_classes) 
    model = model.to(device)
    width,height = 224,224

elif args.model == "ShufflenetV1":
    model = ShuffleNetV1(num_classes=num_classes)
    model = model.to(device)
    width,height = 224,224

elif args.model == "ShufflenetV2":
    model = ShuffleNetV2(n_class=num_classes)
    model = model.to(device)
    width,height = 224,224

elif args.model == "NASnet":
    model = NASNetAMobile(num_classes=num_classes)
    model = model.to(device)

elif args.model == "MobilenetV1":
    model = MobilenetV1(num_classes=num_classes)
    model = model.to(device)
    width,height = 224,224

elif args.model == "MobilenetV2":
    model = MobileNetV2(n_class=num_classes)
    model = model.to(device)
    width,height = 224,224

print ('---------- Model Loaded')

#####################################################################################################################
# Calculating model parameters
'''
model_params = sum(p.numel() for p in model.parameters())
print("\n{0} has a total of {1} parameters\n\n".format(args.model,model_params))
'''
#########################################################################################################################
# Loading from pretrained weights
#model = nn.DataParallel(model)
'''
model.load_state_dict(torch.load("Model_save_weights/CIFAR10/Resnet18/Model_params_epoch_19.pth"))
'''
#############################################################################################################################
if args.weight_decay:
    weight_decay = float(args.weight_decay)
else:
    weight_decay = 0

if args.optimizer=='adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay,amsgrad=True)
elif args.optimizer == 'ASGD':
    optimizer = torch.optim.ASGD(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay)
elif args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay)    
elif args.optimizer == 'Adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay)    
elif args.optimizer == 'RMSProp':
    optimizer = torch.optim.RMSProp(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay)    
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay,momentum=0.9)
elif args.optimizer == 'Adamax':
    optimizer = torch.optim.Adamax(model.parameters(),lr=args.learning_rate,weight_decay=weight_decay)                        

if args.opt_policy=='Step_LR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = args.lr_step_size)
elif args.opt_policy=='exp_LR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = args.exp_LR_gamma)
elif args.opt_policy=='Plateau_LR':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = args.Plateau_patience)    

Epochs = args.epochs

if args.loss == 'cross_entropy':
    Loss_criterion = nn.CrossEntropyLoss()
elif args.loss == 'BCE':
    Loss_criterion = nn.BCEWithLogitsLoss()        
elif args.loss == 'L1':
    Loss_criterion = nn.SmoothL1Loss()  

Train_accuracies = []
Train_timing = []
Test_accuracies = []
Total_time = 0
Max_Val_Accuracy = 0.0
#F = open("Time_logs/"+str(args.dataset)+"/"+str(args.model)+".log","w+")

print ('---------- Starting Training')
for i in range(Epochs):

    model.train()

    Total_right = 0
    Total_counter = 0 

    Train_start = time.time()

    if args.opt_policy != 'Plateau_LR':
        scheduler.step()
    
    for j,(Images,Labels) in enumerate(train_loader):

        #pdb.set_trace()
        Images = Images.float().to(device)

        if (args.dataset in ["MNIST","Fashion-MNIST"]):

            Images = Images.repeat(1,3,1,1)
              
        Labels = Labels.to(device)

        #print(Images.shape)

#####################################################################
# Applying different data augmentation techniques

        if args.method == "Normal":

            Predictions = model(Images)
        
            if args.model in ["Inceptionv3"]:
                loss = Loss_criterion(Predictions[0],Labels)
                Predict_classes = torch.argmax(Predictions[0],dim=1)

            else:
                loss = Loss_criterion(Predictions,Labels)
                Predict_classes = torch.argmax(Predictions,dim=1)       
        
        elif args.method == "Mixup":

            Images,New_Labels,lam = Mixup(Images,Labels,1)
            
            Predictions = model(Images)
        
            if args.model in ["Inceptionv3"]:
                loss_1 = Loss_criterion(Predictions[0],Labels)
                loss_2 = Loss_criterion(Predictions[0],New_Labels)
                loss = (lam*loss_1) + ((1. - lam)*loss_2)

                Predict_classes = torch.argmax(Predictions[0],dim=1)
            else:
                loss_1 = Loss_criterion(Predictions,Labels)
                loss_2 = Loss_criterion(Predictions,New_Labels)
                loss = (lam*loss_1) + ((1. - lam)*loss_2)

                Predict_classes = torch.argmax(Predictions,dim=1)
        
        elif args.method == "Cutmix":

                One_hot_labels = one_hot_encoding(Labels,num_classes)

                Images,Soft_Labels = Cutmix(Images,One_hot_labels)
                Predictions = model(Images)

                if args.model in ["Inceptionv3"]:   
               
                    loss = cross_entropy(Predictions[0],Soft_Labels.to(device))
                    Predict_classes = torch.argmax(Predictions[0],dim=1)

                else:
                    loss = cross_entropy(Predictions,Soft_Labels.to(device))
                    Predict_classes = torch.argmax(Predictions,dim=1)

        elif args.method == "cutmix":

                One_hot_labels = one_hot_encoding(Labels,num_classes)

                Images,Soft_Labels = Cutmix(Images,One_hot_labels)
                Predictions = model(Images)

                if args.model in ["Inceptionv3"]:   
               
                    loss = cross_entropy(Predictions[0],Soft_Labels.to(device))
                    Predict_classes = torch.argmax(Predictions[0],dim=1)

                else:
                    loss = cross_entropy(Predictions,Soft_Labels.to(device))
                    Predict_classes = torch.argmax(Predictions,dim=1)

        elif args.method == "Cutoff":

                Images = Cutoff(Images,width,height)
                Predictions = model(Images)

                if args.model in ["Inceptionv3"]:   
               
                    loss = Loss_criterion(Predictions[0],Labels)
                    Predict_classes = torch.argmax(Predictions[0],dim=1)

                else:

                    loss = Loss_criterion(Predictions,Labels)
                    Predict_classes = torch.argmax(Predictions,dim=1)

        elif args.method == "Cutmix_heatmap":

                Indices = Learned_model(Images)
                #print(Indices[0,:,:])
                One_hot_labels = one_hot_encoding(Labels,num_classes)

                Images,Soft_Labels = Cutmix_heatmap(Images,Indices,One_hot_labels,args.Topk_size)
                
                Predictions = model(Images)

                if args.model in ["Inceptionv3"]:   
               
                    loss = cross_entropy(Predictions[0],Soft_Labels.to(device))
                    Predict_classes = torch.argmax(Predictions[0],dim=1)

                else:
                    loss = cross_entropy(Predictions,Soft_Labels.to(device))
                    Predict_classes = torch.argmax(Predictions,dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Right = torch.sum(Predict_classes==Labels)

        Total_right+=Right
        Total_counter+=Labels.size(0)

        if (j%200)==0:

            print ('Epoch: [{0}/{1}] Image Batches done: [{2}/{3}] Loss: {4:.4f}'.format(i+1,Epochs,j,len(train_loader),loss.item()))
    
    Train_accuracies.append(float(Total_right)/Total_counter)

    Train_end = time.time()

    print ("------ Training Set accuracy: {0:.4f}".format(float(Total_right)/Total_counter))

    if (i%args.test_interval==0) or (i==(Epochs-1)):

        Total_right = 0
        Total_counter = 0 

        model.eval()

        print ('---------- Validating Model on test Data')

        for z,(Images,Labels) in enumerate(test_loader):            

            Images = Images.float().to(device)

            if (args.dataset in ["MNIST","Fashion-MNIST"]):

                Images = Images.repeat(1,3,1,1)

            Labels = Labels.to(device)

            Predictions = model(Images)

            if args.model in ["Inceptionv3"]:
                Test_Loss = Loss_criterion(Predictions[0],Labels)
                Predict_classes = torch.argmax(Predictions[0],dim=1)
            else:
                Test_Loss = Loss_criterion(Predictions,Labels) 
                Predict_classes = torch.argmax(Predictions,dim=1)             
            
            Right = torch.sum(Predict_classes==Labels)

            Accuracy = float(Right)/Labels.size(0)

            Total_right+=Right
            Total_counter+=Labels.size(0)
            
            #print ('Image Batches done: [{0}/{1}] Test_Loss: {2} Accuracy: {3}'.format(z,len(D.val_loader),Test_Loss.item(),Accuracy))

            if (z%100)==0:
                print ('Image Batches done: [{0}/{1}] Accuracy: {2}'.format(z,len(test_loader),Accuracy))

            if args.opt_policy == 'Plateau_LR':
                scheduler.step(Test_Loss)

        Val_acc = float(Total_right)/Total_counter
        print ("------ Validation Set accuracy: {0}".format(Val_acc))

        Test_accuracies.append(Val_acc)

        if Val_acc > Max_Val_Accuracy:
            Max_Val_Accuracy = Val_acc 

        print("Maximum Validation Accuracy until now:{}".format(Max_Val_Accuracy))

    if (i%args.plot_accuracy == 0)  or (i==(Epochs-1)):

        visualization(np.asarray(Train_accuracies),np.asarray(Test_accuracies),args.dataset,args.model,args.method)

        np.save("accuracy_arrays/"+str(args.dataset)+"/"+str(args.model)+'_'+str(args.method)+"_train.npy",np.asarray(Train_accuracies))
        np.save("accuracy_arrays/"+str(args.dataset)+"/"+str(args.model)+'_'+str(args.method)+"_test.npy",np.asarray(Test_accuracies))

        print ("------ Accuracy visualization done !!!")

    torch.save(model.state_dict(),"Model_save_weights/"+str(args.dataset)+"/"+str(args.model)+"/"+'_'+str(args.method)+"_"+"Model_params_epoch_"+str(i)+".pth")
    #F.write("Epoch :{0} Time : {1}\n".format(i+1,Total_time))

#F.close()
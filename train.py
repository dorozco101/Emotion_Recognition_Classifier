

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import os

import torchvision.models as models
from torchvision.models.resnet import model_urls as model_url_resnet
from torchvision.models.alexnet import model_urls as model_url_alexnet
from torchvision.models.vgg import model_urls as model_url_vgg

import argparse
import logging


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=10, type=int, help='epochs (default: 10)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
#parser.add_argument('--numClasses', action='store', default=8, type=int, help='number of classes (default: 8)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.01, type=float, help='learning rate (default: 0.01)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train all layers (default: True)')
parser.add_argument('--useGPU_f', action='store_true', default=False, help='Flag to use GPU (default: False)')
parser.add_argument('--preTrained_f', action='store_false', default=True, help='Flag to whether use pretrained model (default: True)')
parser.add_argument("--net", default='AlexNet', const='AlexNet',nargs='?', choices=['AlexNet', 'ResNet', 'VGG'], help="net model(default:AlexNet)")
parser.add_argument("--dataset", default='Emotions', const='Emotions',nargs='?', choices=['Imagenet', 'Emotions'], help="Dataset (default:Emotions)")
arg = parser.parse_args()



def main():
    # create model directory to store/load old mode
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_'+arg.net+'_'+arg.dataset+'_'+'.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('===============================================================================')
    logger.info('Net: {}'.format(arg.net))
    logger.info('Dataset: {}'.format(arg.dataset))
    logger.info('Train: {}'.format(arg.train_f))

	
    batch_size = arg.batchSize
    # load the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
	
    if arg.dataset == 'Emotions':
        # change the path the model
        train_path = '/data/train'
        val_path = '/data/test'
		
        image_datasets_l = {}
        
        image_datasets_l['train'] = []
        image_datasets_l['train'].append(datasets.ImageFolder(os.path.join(train_path),data_transforms['train']))
            
        image_datasets_l['test'] = []
        image_datasets_l['test'].append(datasets.ImageFolder(os.path.join(val_path),data_transforms['test']))
        

    elif arg.dataset == 'ImageNet':
        #these paths need to be updated to something local
        train_path = '/data/imgDB/DB/ILSVRC/2012/train'
        val_path = '/data5/drone_machinelearning/imageNetData/val'
		
        image_datasets = {}
        
        image_datasets_l['train'] = list(datasets.ImageFolder(os.path.join(train_path),data_transforms['train']))
        image_datasets_l['test'] = list(datasets.ImageFolder(os.path.join(val_path),data_transforms['test']))
        
    image_datasets = {}
    for i in image_datasets_l:
        image_datasets[i] = torch.utils.data.ConcatDataset(image_datasets_l[i])
    # use the pytorch data loader
    arr = ['train', 'test'] 
    print("Data Loading......")
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                  for x in arr}

    dataset_sizes = {x: len(image_datasets[x]) for x in arr}
    print("Data Loaded: "+str(dataset_sizes))
    
    # number of classes should be 40 for modelnet
    
    class_num = len(image_datasets_l['test'][0].classes)
    
    print("Model Loading...")
    # set the pretrained parameter as true and the system will download the model by itself
    if arg.useGPU_f:
        if torch.cuda.is_available():
            use_GPU = True
        else: 
            use_GPU = False
            print("Error: NO GPU AVAILABLE, NOT USING GPU")
    else:
        use_GPU = False
        print("Not using GPU")
        
    if arg.net == 'AlexNet':
        model_url_alexnet['alexnet'] = model_url_alexnet['alexnet'].replace('https://', 'http://')
        model = models.alexnet(pretrained=arg.preTrained_f)
    elif arg.net == 'ResNet':
        model_url_resnet['resnet18'] = model_url_resnet['resnet18'].replace('https://', 'http://')
        model = models.resnet18(pretrained=arg.preTrained_f)
    elif arg.net == 'VGG':
        model_url_vgg['vgg16'] = model_url_vgg['vgg16'].replace('https://', 'http://')
        model = models.vgg16(pretrained=arg.preTrained_f)
    
    print("Model Loaded")
    # for our emotion dataset there are only 8 categories
    # so what we should do is to check the last layer of resnet
    # for example, the last layer of resNet is self.fc = nn.Linear(512, num_classes)
    # we can change the last fully connected layer as 
    print("Optimizer Setting...")
    if not arg.train_f:
        #Freeze layers if we only want to train the outer layers
        for name,param in model.named_parameters():
            param.requires_grad=False
        print("FREEZING LAYERS")
    if arg.dataset == 'Emotions':
        if arg.net == 'ResNet':
            model.fc= nn.Linear(512, out_features=class_num)
            optimizer = optim.SGD(params=[model.fc.weight, model.fc.bias], lr=arg.lr, momentum=arg.m)
        else:
            model.classifier._modules['6']= nn.Linear(4096, out_features=class_num)
            optimizer = optim.SGD(params=[model.classifier._modules['6'].weight,model.classifier._modules['6'].bias], lr=arg.lr, momentum=arg.m)
    
    print("Optimizer Set")
    # if we only want to train the last layer, we should
    # 1. set "requires_grad" in other layers as False
    # for para in list(model.parameters())[:-2]:
    #    para.requires_grad=False 

    # 2. construct an optimizer,optimizer that only cares about the last layer
    # optimizer = optim.SGD(params=[model.fc.weight, model.fc.bias], lr=1e-3)
    # In this way, only the parameter in the last layer is re-initialized
    
    # for gpu mode
    if use_GPU:
        model.cuda()
    # for cpu mode
    else:
        model
    model_path = 'model/old_model_'+arg.net+'_'+arg.dataset+'.pt'
    
    #if we have a model that we have trained before
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    
    
    # training
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs 
    for epoch in range(epochs):
        # trainning
        overall_acc = 0
        for batch_idx, (x, target) in enumerate(dataloaders['train']):

            optimizer.zero_grad()

            # for gpu mode
            if use_GPU:
                x, target = Variable(x.cuda()), Variable(target.cuda())
            # for cpu mode
            else:
                x, target = Variable(x), Variable(target)

            # use cross entropy loss
            criterion = nn.CrossEntropyLoss()
            outputs = model(x)
            loss = criterion(outputs, target)
            _, pred_label = torch.max(outputs.data, 1)
            correct = (pred_label == target.data).sum()
            overall_acc += correct
            accuracy = correct*1.0/batch_size

            loss.backward()              
            optimizer.step()             
            
            
            if batch_idx%1==0:
                print('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.data[0], accuracy))
                logger.info('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.data[0], accuracy))

        # save the model every epoch
        torch.save(model.state_dict(), model_path)
  
      
    
    
    # testing
    #model = torch.load('model/old_model.pt')
    print("Start Testing")
    logger.info("Start Testing")
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    correct, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(dataloaders['test']):
        # for gpu mode
        if use_GPU:
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            # for cpu mode
        else:
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)

            # use cross entropy loss
        criterion = nn.CrossEntropyLoss()

        outputs = model(x)
        loss = criterion(outputs, target)
        _, pred_label = torch.max(outputs.data, 1)
        correct += (pred_label == target.data).sum()
        ave_loss += loss.data[0]


    accuracy = correct*1.0/len(dataloaders['test'])/batch_size
    ave_loss /= len(dataloaders['test'])
    print('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
    logger.info('==>>> test loss:{}, accuracy:{}'.format(ave_loss, accuracy))
    
if __name__ == "__main__":
    main()
    

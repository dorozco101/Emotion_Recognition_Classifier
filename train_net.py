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
import copy
import torchvision.models as models
from torchvision.models.resnet import model_urls as model_url_resnet
from torchvision.models.alexnet import model_urls as model_url_alexnet
from torchvision.models.vgg import model_urls as model_url_vgg
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import logging


parser = argparse.ArgumentParser(description='PyTorch AlexNet Training')
parser.add_argument('-e', '--epochs', action='store', default=10, type=int, help='epochs (default: 10)')
parser.add_argument('--batchSize', action='store', default=128, type=int, help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.01, type=float, help='learning rate (default: 0.01)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--train_all_para_f', action='store_false', default=True, help='Flag to train all parameters (default: True)')
parser.add_argument('--useGPU_f', action='store_true', default=False, help='Flag to use GPU (STORE_FALSE)(default: False)')
parser.add_argument('--preTrained_f', action='store_true', default=False, help='Flag to pretrained model (default: True)')
parser.add_argument("--net", default='AlexNet', const='AlexNet',nargs='?', choices=['AlexNet', 'ResNet', 'VGG'], help="net model(default:AlexNet)")
parser.add_argument("--dataset", default='Emotions', const='Emotions',nargs='?', choices=['Emotions', 'ImageNet'], help="Dataset (default:Emotions)")
parser.add_argument('--shuffle', action='store_false', default=True, help='Flag to shuffle valid/train dataset (default: True)')
parser.add_argument('--validation_percent', action='store', default=0.1, type=float, help='float representing validation set size (default: 0.1)')
# parser.add_argument('')

arg = parser.parse_args()

class ConcatDataset(torch.utils.data.Dataset):
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple(d[i] for d in self.datasets)

	def __len__(self):
		return min(len(d) for d in self.datasets)

def main():
	# create model directory to store/load old mode
	if not os.path.exists('model'):
		os.makedirs('model')
	if not os.path.exists('log'):
		os.makedirs('log')
	# Logger Setting
	logger = logging.getLogger('netlog')
	logger.setLevel(logging.INFO)
	ch = logging.FileHandler('log/logfile_'+arg.net+'_'+arg.dataset+'.log')
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	
	batch_size = arg.batchSize
	# load the data
	data_transforms = {
		'train': transforms.Compose([
            transforms.Resize(240),
            transforms.RandomRotation(1),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(0.08),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize(240),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}
	
	if arg.dataset == 'Emotions':
		train_path = './data/train'
		val_path = './data/test'
	elif arg.dataset == 'ImageNet':
		#these paths need to be updated to something local
		train_path = '/data/imgDB/DB/ILSVRC/2012/train'
		val_path = '/data/drone_machinelearning/imageNetData/val'
		
	if arg.useGPU_f:
		if torch.cuda.is_available():
			use_GPU = True
		else: 
			use_GPU = False
			print("Error: NO GPU AVAILABLE, NOT USING GPU")
	else:
		use_GPU = False
		print("Not using GPU")
		
	image_datasets_all = {}
	results = []
	image_datasets_all['test'] = []
	#test on all angles
	for folder in os.listdir(val_path):
         
         image_datasets_all['test'].append(datasets.ImageFolder(os.path.join(val_path),data_transforms['test']))

	for numberOfRetests in range(1):
		image_datasets_all['train'] = []
		for folder in os.listdir(train_path):
			print(train_path+'/'+folder)
			image_datasets_all['train'].append(datasets.ImageFolder(os.path.join(train_path+'/'+folder),data_transforms['train']))
			
		image_datasets = {}

		for i in image_datasets_all:
			image_datasets[i] = torch.utils.data.ConcatDataset(image_datasets_all[i])
            
		arr = ['train', 'test'] if arg.train_f else ['test']
		dataset_sizes = {x: len(image_datasets[x]) for x in arr}
		print("Data Loaded")

		 # use the pytorch data loader
		valid_size = arg.validation_percent #10 percent hold out
		num_train = len(image_datasets['train'])
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))

		if arg.shuffle:
		    np.random.seed()
		    np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)
		dataloaders = {}
		dataloaders['train'] = torch.utils.data.DataLoader(
		image_datasets['train'], batch_size=batch_size, sampler=train_sampler,
		num_workers=0
		)
		dataloaders['valid'] = torch.utils.data.DataLoader(
		image_datasets['train'], batch_size=batch_size, sampler=valid_sampler,
		num_workers=0
		)
		dataloaders['test'] = torch.utils.data.DataLoader(
		image_datasets['test'], batch_size=batch_size, 
		num_workers=0, 
		)
        
		print("Data Loading......")
		#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
		#		  for x in arr}

		# number of classes should be 8 for emotion dataset
		class_num = len(image_datasets_all['test'][0].classes)
		# print (class_num)
		print(dataset_sizes)
		print("Model Loading...")

		if arg.net == 'AlexNet':
			model_url_alexnet['alexnet'] = model_url_alexnet['alexnet'].replace('https://', 'http://')
			model = models.alexnet(pretrained=arg.preTrained_f)
		elif arg.net == 'ResNet':
			model_url_resnet['resnet18'] = model_url_resnet['resnet18'].replace('https://', 'http://')
			model = models.resnet18(pretrained=arg.preTrained_f)
		elif arg.net == 'VGG':
			model_url_vgg['vgg16'] = model_url_vgg['vgg16'].replace('https://', 'http://')
			model = models.vgg16(pretrained=arg.preTrained_f)

		print("model Loaded")
		print("Optimizer Setting...")
		# for emotion dataset there are only 8 categories
		# so what we should do is to check the last layer of resnet
		# for example, the last layer of resNet is self.fc = nn.Linear(512, num_classes)
		# we can change the last fully connected layer as 
		#if not arg.train_all_para_f:
		#for name,param in model.named_parameters():
		 #   param.requires_grad=False

		if not arg.train_all_para_f:
			#Freeze layers if we only want to train the outer layers
			for name,param in model.named_parameters():
				param.requires_grad=False
			print("FREEZING LAYERS")
		
		#change the size of last layer
		if arg.dataset == 'Emotions':
			if arg.net == 'ResNet':
				model.fc= nn.Linear(512, out_features=class_num)
			else:
				model.classifier._modules['6']= nn.Linear(4096, out_features=class_num)  
		#optimizer must match the number of trainable parameters (should not include non trainable layers)
		if not arg.train_all_para_f:
			optimizer = optim.SGD(params=[model.fc.weight, model.fc.bias], lr=arg.lr, momentum=arg.m)
		else:
			optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=arg.m)
		print("Optimizer Set")

	# if we only want to train the last layer, we should
	# 1. set "requires_grad" in other layers as False
	# for para in list(model.parameters())[:-2]:
	#    para.requires_grad=False 

	# 2. construct an optimizer,optimizer that only cares about the last layer
	# optimizer = optim.SGD(params=[model.fc.weight, model.fc.bias], lr=1e-3)
	# In this way, only the parameter in the last layer is re-initialized

	# for gpu mode
		print("configuring gpu")
		if use_GPU:
			model.cuda()
	# for cpu mode
		else:
			model

		print("starting")

		model_path = './data/old_model_'+arg.net+'_'+str(arg.dataset)+'.pt'
		test_path = './data/old_model_'+arg.net+'_'+'test'+str(arg.dataset)+'.pt'

		#check if 
		if os.path.isfile(model_path):
			#model = torch.load('model/old_model.pt')
			model.load_state_dict(torch.load(model_path))


		# training
		print("Start Training")
		logger.info("Start Training")
		epochs = arg.epochs if arg.train_f else 0
		loss_per_epoch = np.zeros(epochs)
		acc_per_epoch = np.zeros(epochs)
		loss_per_epoch_valid = np.zeros(epochs)
		acc_per_epoch_valid = np.zeros(epochs)
        
		valid_counter = 0
		prev_loss = np.inf
		for epoch in range(epochs):
			# trainning
			overall_acc = 0
			# num_samples, (_, _) = enumerate(dataloaders['train'])
			overall_acc_valid = 0
                
			for batch_idx, (x, target) in enumerate(dataloaders['train']):

				optimizer.zero_grad()
				
				if use_GPU:
					x, target = Variable(x.cuda()), Variable(target.cuda())
				# for cpu mode
				else:
					x, target = Variable(x), Variable(target)
				
				# use cross entropy loss
				criterion = nn.CrossEntropyLoss()
				if use_GPU:
					outputs = model(x.cuda())
				else:
					outputs = model(x)
				#print("num clsses: " + str(class_num))
				#print(target)
				loss = criterion(outputs, target)
				_, pred_label = torch.max(outputs.data, 1)
				#print("This is pred label")
				#print(pred_label)
				
				correct = (pred_label == target.data).sum().cpu().data.numpy()
				overall_acc += correct
				accuracy = correct*1.0/batch_size

				loss.backward()              
				optimizer.step()             


				if batch_idx%100==0:
					print('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}%'.format(epoch,batch_idx, loss.item(), accuracy*100))
					logger.info('==>>> epoch:{}, batch index: {}, train loss:{}, accuracy:{}%'.format(epoch,batch_idx, loss.item(), accuracy*100))
				loss_per_epoch[epoch] += loss.item()
				acc_per_epoch[epoch] += accuracy
            
			for batch_idx, (x, target) in enumerate(dataloaders['valid']):

				if use_GPU:
					x, target = Variable(x.cuda()), Variable(target.cuda())
				# for cpu mode
				else:
					x, target = Variable(x), Variable(target)
				
				# use cross entropy loss
				criterion = nn.CrossEntropyLoss()
				if use_GPU:
					outputs = model(x.cuda())
				else:
					outputs = model(x)
				#print("num clsses: " + str(class_num))
				#print(target)
				loss = criterion(outputs, target)
				_, pred_label = torch.max(outputs.data, 1)
				#print("This is pred label")
				#print(pred_label)
				
				correct = (pred_label == target.data).sum().cpu().data.numpy()
				overall_acc_valid += correct
				accuracy = correct*1.0/batch_size
				#print(batch_idx)

				if batch_idx%100==0:
					print('==>>> epoch:{}, batch index: {}, valid loss:{}, accuracy:{}%'.format(epoch,batch_idx, loss.item(), accuracy*100))
					logger.info('==>>> epoch:{}, batch index: {}, valid loss:{}, accuracy:{}%'.format(epoch,batch_idx, loss.item(), accuracy*100))
				loss_per_epoch_valid[epoch] += loss.item()
				acc_per_epoch_valid[epoch] += accuracy
			if accuracy > 0.9:
			    print("should we break because of validation accuracy?")


			if loss_per_epoch_valid[epoch] > prev_loss:
				valid_counter += 1
				if valid_counter == 3:
					break
			else:
				valid_counter = 0
			prev_loss = loss_per_epoch_valid[epoch]

			# save the model per epochs
			torch.save(model.state_dict(), test_path)

		# saving Training loss and accuracy for results
			loss_per_epoch[epoch] /= (dataset_sizes['train']/float(batch_size))		
			acc_per_epoch[epoch] /= (dataset_sizes['train']/float(batch_size))

			print('epoch:{}, loss:{}'.format(epoch,loss_per_epoch[epoch]))
			print('epoch:{}, accuracy:{}'.format(epoch,acc_per_epoch[epoch]))
		np.save(str(arg.net)+'TrainLoss'+str(arg.dataset)+'.npy', loss_per_epoch)
		np.save(str(arg.net)+'TrainAccuracy'+str(arg.dataset)+'.npy', acc_per_epoch)

		# testing
		print("Start Testing")
		logger.info("Start Testing")
		if os.path.isfile(test_path):
			model.load_state_dict(torch.load(test_path))
		model.eval()
		correct, ave_loss = 0, 0
		correct_class = np.zeros(class_num)
		count_class = np.zeros(class_num)
		confusion_matrix = np.zeros((class_num,class_num))
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

			for i in range(len(pred_label)):
				true = target.data[i]
				predicted = pred_label[i]
				if pred_label[i] == target.data[i]:
					correct_class[pred_label[i]] += 1
				count_class[target.data[i]] += 1
				confusion_matrix[true,predicted]+=1
                
			correct += (pred_label == target.data).sum().cpu().data.numpy()
			ave_loss += loss.data[0]

		confusion_matrix  = confusion_matrix/confusion_matrix.sum(axis=1)[:,None]
            

		accuracy = correct*1.0/dataset_sizes['test']
		ave_loss /= dataset_sizes['test']
		print('==>>> test loss:{}, accuracy:{}'.format(ave_loss.item(), accuracy))

		logger.info('==>>> test loss:{}, accuracy:{}'.format(ave_loss.item(), accuracy))
		allResults = np.zeros(class_num+2)
		allResults[class_num] = accuracy

		allResults[class_num+1] = numberOfRetests

		for i in range(class_num):
			allResults[i] = correct_class[i]*1.0/count_class[i]
			logger.info('==>>> class:{:10} accuracy:{}'.format(image_datasets_all['test'][0].classes[i], allResults[i]))

		results.append(allResults)
		np.save(str(arg.net)+'TestResults'+str(arg.dataset)+'.npy',np.asarray(results))

		np.save(str(arg.net)+'ConfusionMatrix'+str(arg.dataset)+':testNUM:'+str(numberOfRetests)+'.npy',confusion_matrix)

	

if __name__ == "__main__":
	main()

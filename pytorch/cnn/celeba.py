#A CNN that can figure out if the subject in an image is smiling or not

import torch
import torchvision
from torchvision import transforms

#we have already downloded the dataset on disk 'cause we need to see what this is
#instead of blindly asking torchvision to download something.
image_path = '/home/som/prog/pyml/pytorch/Datasets/celeba'

#smile is attribute number 31. Binary value of 0 or 1.
#see https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html
get_smile = lambda attr: attr[31]

#the input images are of various sizes. We want to resize them to 64x64
transform = transforms.Compose([transforms.Resize([64,64]),
                                transforms.ToTensor()])

celeba_train_dataset = torchvision.datasets.CelebA(image_path,
                                                   split='train',
                                                   target_type='attr',
                                                   download=False,
                                                   transform=transform,
                                                   target_transform=get_smile)

celeba_valid_dataset = torchvision.datasets.CelebA(image_path,
                                                   split='valid',
                                                   target_type='attr',
                                                   download=False,
                                                   transform=transform,
                                                   target_transform=get_smile)

celeba_test_dataset = torchvision.datasets.CelebA(image_path,
                                                  split='test',
                                                  target_type='attr',
                                                  download=False,
                                                  transform=transform,
                                                  target_transform=get_smile)

#this is a pretty large dataset, and it takes a bit for pytorch to load it
# print(len(celeba_train_dataset))
# print(len(celeba_valid_dataset))
# print(len(celeba_test_dataset))

# (pytorch) som@som-linux-laptop:~/prog/pyml/pytorch/cnn$ python celeba.py 
# 162770
# 19867
# 19962

#create the dataloaders
from torch.utils.data import DataLoader

batch_size=32
torch.manual_seed(1)

train_dl = DataLoader(celeba_train_dataset,
                      batch_size,
                      shuffle=True)

valid_dl = DataLoader(celeba_valid_dataset,
                      batch_size,
                      shuffle=False)

test_dl = DataLoader(celeba_test_dataset,
                     batch_size,
                     shuffle=False)


#setup the CNN
#input images have size 64x64. 3 channels
#4 conv layers: 32, 64, 128, 256 feature maps. Kernel size = 3x3, padding = 1
#first three conv layers have max pooling with 2x2
#2 dropout layers for regularization

import torch.nn as nn

model = nn.Sequential()

model.add_module("conv1",
                 nn.Conv2d(in_channels=3,
                           out_channels=32,
                           kernel_size=3, #3x3
                           padding=1))

model.add_module("relu1", nn.ReLU())

model.add_module("pool1", nn.MaxPool2d(kernel_size=2))

model.add_module("dropout1", nn.Dropout(p=0.5))

model.add_module("conv2",
                 nn.Conv2d(in_channels=32,
                           out_channels=64,
                           kernel_size=3,
                           padding=1))

model.add_module("relu2", nn.ReLU())

model.add_module("pool2", nn.MaxPool2d(kernel_size=2))

model.add_module("dropout2", nn.Dropout(p=0.5))

model.add_module("conv3",
                 nn.Conv2d(in_channels=64,
                           out_channels=128,
                           kernel_size=3,
                           padding=1))

model.add_module("relu3", nn.ReLU())

model.add_module("pool3", nn.MaxPool2d(kernel_size=2))

model.add_module("conv4",
                 nn.Conv2d(in_channels=128,
                           out_channels=256,
                           kernel_size=3,
                           padding=1))

model.add_module("relu4", nn.ReLU())

#due to the 2x2 pooling, the input here is 8x8x256, i.e., 256 feature maps of size 8x8

#now for the fully connected layer
#that would ordinarily have 8x8x256=16384 neurons and that is a lot
#there is a method called global average pooling, that will compute the average of each feature map
#and therefore the number of outputs will be 256 instead
#global average pooling is essentially average pooling where the pooling size is equal to the size of the feature maps

model.add_module("globalavgpool",
                 nn.AvgPool2d(kernel_size=8))

model.add_module("flatten", nn.Flatten())

model.add_module("fc1", nn.Linear(256,1)) # 1 because the smiling attr is 0 or 1

model.add_module("sigmoid", nn.Sigmoid()) # and sigmoid is great because we just need a binary 0 or 1.

#model.cuda()

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from train import celeba_train

print('Starting Training')
celeba_train(model, 30, train_dl, valid_dl, loss_fn, optimizer)

#save the model parameters since this is a long training loop
torch.save(model, "/home/som/prog/pyml/pytorch/cnn/celeba_cnn.pt")


# (pytorch) som@som-linux-laptop:~/prog/pyml/pytorch/cnn$ time python celeba.py 
# Starting Training
# device = cuda:0
# Epoch 1 | Training Accuracy: 0.7262640595436096 | Validation Accuracy: 0.8376201391220093
# Epoch 2 | Training Accuracy: 0.8704921007156372 | Validation Accuracy: 0.8959581255912781
# Epoch 3 | Training Accuracy: 0.8877864480018616 | Validation Accuracy: 0.910756528377533
# Epoch 4 | Training Accuracy: 0.8954843878746033 | Validation Accuracy: 0.9118638634681702
# Epoch 5 | Training Accuracy: 0.8997296690940857 | Validation Accuracy: 0.9175013899803162
# Epoch 6 | Training Accuracy: 0.9032745361328125 | Validation Accuracy: 0.9164443612098694
# Epoch 7 | Training Accuracy: 0.9040486216545105 | Validation Accuracy: 0.914833664894104
# Epoch 8 | Training Accuracy: 0.9059654474258423 | Validation Accuracy: 0.9205718040466309
# Epoch 9 | Training Accuracy: 0.9068439602851868 | Validation Accuracy: 0.9181557297706604
# Epoch 10 | Training Accuracy: 0.9082140326499939 | Validation Accuracy: 0.9180047512054443
# Epoch 11 | Training Accuracy: 0.9086502194404602 | Validation Accuracy: 0.9171993732452393
# Epoch 12 | Training Accuracy: 0.9095901846885681 | Validation Accuracy: 0.9186590909957886
# Epoch 13 | Training Accuracy: 0.9109417796134949 | Validation Accuracy: 0.9197664260864258
# Epoch 14 | Training Accuracy: 0.9113472700119019 | Validation Accuracy: 0.921981155872345
# Epoch 15 | Training Accuracy: 0.9113104343414307 | Validation Accuracy: 0.9223838448524475
# Epoch 16 | Training Accuracy: 0.9127172827720642 | Validation Accuracy: 0.923088550567627
# Epoch 17 | Training Accuracy: 0.913294792175293 | Validation Accuracy: 0.9244979023933411
# Epoch 18 | Training Accuracy: 0.9132395386695862 | Validation Accuracy: 0.9230381846427917
# Epoch 19 | Training Accuracy: 0.9143822193145752 | Validation Accuracy: 0.92278653383255
# Epoch 20 | Training Accuracy: 0.9140812158584595 | Validation Accuracy: 0.9234408736228943
# Epoch 21 | Training Accuracy: 0.9153529405593872 | Validation Accuracy: 0.9206221103668213
# Epoch 22 | Training Accuracy: 0.9154696464538574 | Validation Accuracy: 0.9216288328170776
# Epoch 23 | Training Accuracy: 0.9160963296890259 | Validation Accuracy: 0.9207227826118469
# Epoch 24 | Training Accuracy: 0.9155433773994446 | Validation Accuracy: 0.9249005913734436
# Epoch 25 | Training Accuracy: 0.9157891273498535 | Validation Accuracy: 0.9223335385322571
# Epoch 26 | Training Accuracy: 0.9173434376716614 | Validation Accuracy: 0.9239945411682129
# Epoch 27 | Training Accuracy: 0.916864275932312 | Validation Accuracy: 0.9229878783226013
# Epoch 28 | Training Accuracy: 0.9179701209068298 | Validation Accuracy: 0.9207731485366821
# Epoch 29 | Training Accuracy: 0.9175277948379517 | Validation Accuracy: 0.9225851893424988
# Epoch 30 | Training Accuracy: 0.9173741936683655 | Validation Accuracy: 0.923692524433136

# real	69m50.201s
# user	59m54.112s
# sys	11m4.382s



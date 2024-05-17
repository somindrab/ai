#yet another mnist classification program
#but this one uses a Convolutional Neural Network (CNN)

#Load the data
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

image_path = './'
transform = transforms.Compose([transforms.ToTensor()])

mnist_dataset = torchvision.datasets.MNIST(root=image_path,
                                           train=True,
                                           transform=transform,
                                           download=True)

#split the training dataset into training and validation.
#torchvision has training and test, and so we need to do this
#split by ourselves

from torch.utils.data import Subset
mnist_validation_dataset = Subset(mnist_dataset,
                                  torch.arange(10000))

mnist_train_dataset = Subset(mnist_dataset,
                             torch.arange(10000,len(mnist_dataset)))


mnist_test_dataset = torchvision.datasets.MNIST(root=image_path,
                                                train=False,
                                                transform=transform,
                                                download=False)

#next, create dataloaders with a good batchsize
from torch.utils.data import DataLoader

batch_size=64
train_dl = DataLoader(mnist_train_dataset,
                      batch_size,
                      shuffle=True)

validation_dl = DataLoader(mnist_validation_dataset,
                           batch_size,
                           shuffle=False)


#now, for setting up the CNN
#batchsize = 64
#input: batchsize x 28 x 28 x 1
#conv1: batchsize x 28 x 28 x 32 (5 x 5 x 32 kernel)
#pooling1: batchsize x 14 x 14 x 32 (2x2 pooling with stride 2)
#conv2: 14 x 14 x 64 (5 x 5 x 64 kernel)
#pooling2: 7 x 7 x 64 (2 x 2 pooling with stride 2)
#fc1: 7x7x64 = 3136 x 1024 (1024 neurons, input is the flattened pooling2 output)
#fc2: 1024 x 10 (10 neurons, corresponding to the 10 possible classes)

model = nn.Sequential()
model.add_module("conv1",
                 nn.Conv2d(in_channels=1,
                           out_channels=32,
                           kernel_size=5, #5x5
                           padding=2 #this will keep the input and output sizes same
                           )
                 )

model.add_module("relu1", nn.ReLU())

model.add_module("pool1",
                 nn.MaxPool2d(kernel_size=2)) #2x2 pooling. default stride == pooling size

model.add_module("conv2",
                 nn.Conv2d(in_channels=32,
                           out_channels=64,
                           kernel_size=5,
                           padding=2)
                 )

model.add_module("relu2", nn.ReLU())

model.add_module("pool2", nn.MaxPool2d(kernel_size=2))

#flatten the output here to get ready for the fully connected dense layers
model.add_module("flatten", nn.Flatten())

model.add_module("fc1", nn.Linear(3136, 1024))
model.add_module("relufc1", nn.ReLU())

model.add_module("fc2", nn.Linear(1024, 10))

#no need to add an activation function; Softmax is used by default by
#Categorical Cross Entroy Loss (categorial as we have multiple class labels)
#and so we don't need to add one explicitly be default

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from mnist_train import train

train(model, 20, train_dl, validation_dl, loss_fn, optimizer)
                           
# (pytorch) som@som-linux-laptop:~/prog/pyml/pytorch/cnn$ python mnist.py 
# Epoch 1 | Training Accuracy: 0.9593200087547302 | Validation Accuracy: 0.984499990940094
# Epoch 2 | Training Accuracy: 0.9871199727058411 | Validation Accuracy: 0.9850000143051147
# Epoch 3 | Training Accuracy: 0.9921000003814697 | Validation Accuracy: 0.9891999959945679
# Epoch 4 | Training Accuracy: 0.9940000176429749 | Validation Accuracy: 0.9894000291824341
# Epoch 5 | Training Accuracy: 0.995140016078949 | Validation Accuracy: 0.9873999953269958
# Epoch 6 | Training Accuracy: 0.9959800243377686 | Validation Accuracy: 0.9907000064849854
# Epoch 7 | Training Accuracy: 0.9962800145149231 | Validation Accuracy: 0.9865000247955322
# Epoch 8 | Training Accuracy: 0.9974600076675415 | Validation Accuracy: 0.9901000261306763
# Epoch 9 | Training Accuracy: 0.9973999857902527 | Validation Accuracy: 0.9908000230789185
# Epoch 10 | Training Accuracy: 0.998199999332428 | Validation Accuracy: 0.9890999794006348
# Epoch 11 | Training Accuracy: 0.9976000189781189 | Validation Accuracy: 0.9897000193595886
# Epoch 12 | Training Accuracy: 0.998199999332428 | Validation Accuracy: 0.9904000163078308
# Epoch 13 | Training Accuracy: 0.9988600015640259 | Validation Accuracy: 0.9916999936103821
# Epoch 14 | Training Accuracy: 0.998520016670227 | Validation Accuracy: 0.9908000230789185
# Epoch 15 | Training Accuracy: 0.9980199933052063 | Validation Accuracy: 0.9884999990463257
# Epoch 16 | Training Accuracy: 0.9987000226974487 | Validation Accuracy: 0.9912999868392944
# Epoch 17 | Training Accuracy: 0.9987999796867371 | Validation Accuracy: 0.9911999702453613
# Epoch 18 | Training Accuracy: 0.998740017414093 | Validation Accuracy: 0.9904000163078308
# Epoch 19 | Training Accuracy: 0.9994999766349792 | Validation Accuracy: 0.9909999966621399
# Epoch 20 | Training Accuracy: 0.9983800053596497 | Validation Accuracy: 0.9887999892234802

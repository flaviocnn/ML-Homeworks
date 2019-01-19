import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_accuracy(net, dataloader):
  ########TESTING PHASE###########
  
    #check accuracy on whole test set
    correct = 0
    total = 0
    net.eval() #important for deactivating dropout and correctly use batchnorm accumulated statistics
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (
    accuracy))
    return accuracy


# function to define an old style fully connected network (multilayer perceptrons)
class old_nn(nn.Module):
    def __init__(self,D_in, H, D_out):
        super(old_nn, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H,H)
        self.fc3 = nn.Linear(H, D_out) #last FC for classification 

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
 

class Net(nn.Module):
    def __init__(self,D_in, H, D_out):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H,H)
        self.fc3 = nn.Linear(H, D_out)
        # A fully connected neural network layer is represented by the nn.Linear object,
        # with the first argument in the definition being the number of nodes in layer l 
        # and the next argument being the number of nodes in layer l+1. 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#function to define the convolutional network
class CNN(nn.Module):
    def __init__(self,D_out,filters):
        super(CNN, self).__init__()
        #conv2d first parameter is the number of kernels at input (you get it from the output value of the previous layer)
        #conv2d second parameter is the number of kernels you wanna have in your convolution, so it will be the n. of kernels at output.
        #conv2d third, fourth and fifth parameters are, as you can read, kernel_size, stride and zero padding :)
        self.conv1 = nn.Conv2d(3, filters, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(filters, filters * 2, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear( 2 * 4 * 4 * filters, 4096)
        self.fc2 = nn.Linear(4096, D_out) #last FC for classification 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.pool(self.conv_final(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #hint: dropout goes here!
        x = self.fc2(x)
        return x


class CNNExtA(nn.Module):
    def __init__(self,D_out,filters):
        super(CNNExtA, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(filters, filters * 2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters * 2)
        )
        
        self.fc1 = nn.Linear(filters * 2 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, D_out)

    def forward(self, x):
            x = self.layer1(x)

            x = self.layer2(x)

            x = self.layer3(x)

            x = self.layer4(x)

            x = x.view(x.shape[0], -1)

            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return x


class CNNExtB(nn.Module):
    def __init__(self,D_out,filters):
        super(CNNExtB, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(filters, filters * 2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters * 2)
        )
        
        self.fc1 = nn.Linear(filters * 2 * 3 * 3, 8192)
        self.fc2 = nn.Linear(8192, D_out)

    def forward(self, x):
            x = self.layer1(x)

            x = self.layer2(x)

            x = self.layer3(x)

            x = self.layer4(x)

            x = x.view(x.shape[0], -1)

            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return x



class CNNExtC(nn.Module):
    def __init__(self,D_out,filters):
        super(CNNExtC, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(filters)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(filters, filters * 2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(filters * 2)
        )

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(filters * 2 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, D_out)

    def forward(self, x):
            x = self.layer1(x)

            x = self.layer2(x)

            x = self.layer3(x)

            x = self.layer4(x)

            x = x.view(x.shape[0], -1)

            x = self.drop_out(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            return x




def train_net(net, trainloader, testloader, epochs, learning_rate, name):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    n_loss_print = len(trainloader)  #print every epoch, use smaller numbers if you wanna print loss more often!

    points = []
    since = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times

        net.train() #important for activating dropout and correctly train batchnorm
        running_loss = 0.0
        training_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % n_loss_print == (n_loss_print - 1):    # print every 2000 mini-batches
                training_loss = running_loss / n_loss_print
                print('\n[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, training_loss ))
                running_loss = 0.0

        accuracy = test_accuracy(net,testloader)
        points.append((training_loss,accuracy))

    print('Finished Training Net {}'.format(name))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    p = np.asarray(points)
    plt.title("Training Loss vs. Validation Accuracy")
    plt.xlabel("Training Loss")
    plt.ylabel("Validation Accuracy")
    plt.plot(p[:,0],p[:,1])
    plt.ylim((0,p.max() + 1))
    plt.grid()
    # grab a reference to the current axes
    ax = plt.gca()
    # set the xlimits to be the reverse of the current xlimits
    ax.set_xlim(ax.get_xlim()[::-1])
    # call `draw` to re-render the graph
    plt.draw()
    plt.pause(0.001)
    plt.show(block = False)


def train_model(net, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
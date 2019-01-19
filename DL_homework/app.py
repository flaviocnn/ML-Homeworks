import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import nets
import transformations as t
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# Deep Learning Homework - Flavio E. Cannavo'
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Set of parameters:

N, D_in, H, D_out, learning_rate, epochs = 256, 32*32*3, 4096, 100, 0.0001, 20
# ----------------------------------------------------------------------
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

# There are many rule-of-thumb methods for determining the correct number of neurons to use 
# in the hidden layers, such as the following:

# - The number of hidden neurons should be between the size of the input layer 
#       and the size of the output layer.
# - The number of hidden neurons should be 2/3 the size of the input layer, 
#       plus the size of the output layer.
# - The number of hidden neurons should be less than twice the size of the input layer.

def get_labels():
    
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    label_names = unpickle("./data/cifar-100-python/meta")
    classes = label_names[b'fine_label_names']
    labels = {}
    for i,label in enumerate(classes):
        labels[i] = label
    return labels


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(model_weights['conv1.weight']):
    #print(filt[0, :, :])
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')
    
    plt.show()


def plot_kernel_output(model,images):
    fig1 = plt.figure()
    plt.figure(figsize=(1,1))
    
    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1,2,0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def main():
    transform = t.basic()

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True, num_workers=4,drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=N,
                                         shuffle=False, num_workers=4,drop_last=True)
    
    dataiter = iter(trainloader)

    classes_dict = get_labels()
    classes = list(classes_dict.values())

    ########################################################################
    # Let us show some of the training images, for fun.
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print 4 labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # show images
    imshow(torchvision.utils.make_grid(images))


    test_nets = []
    test_nets.append(nets.old_nn(D_in, H, D_out))
    test_nets.append(nets.CNN(D_out,32))
    test_nets.append(nets.CNN(D_out,128))
    test_nets.append(nets.CNN(D_out,256))
    test_nets.append(nets.CNN(D_out,512))
    test_nets.append(nets.CNNExtA(D_out,128))
    test_nets.append(nets.CNNExtB(D_out,128))
    test_nets.append(nets.CNNExtC(D_out,128))

    for i,net in enumerate(test_nets,1):
     
        net.to(device) # enables CUDA
        nets.train_net(net,trainloader,testloader,epochs,learning_rate,i)



    #######################################################################
    # Set up new Network with data augmentation
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    transform = t.augmentation1()


    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True, num_workers=4,drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=N,
                                         shuffle=False, num_workers=4,drop_last=True)
        
    net = test_nets[2].to(device) # nets3
    nets.train_net(net,trainloader,testloader,epochs,learning_rate, "CNN Augmentation 1")

    # ----------------------------------------------------------------------

    transform = t.augmentation2()


    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=N,
                                          shuffle=True, num_workers=4,drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=N,
                                         shuffle=False, num_workers=4,drop_last=True)
        
    net = test_nets[2].to(device) # nets3
    nets.train_net(net,trainloader,testloader,epochs,learning_rate, "CNN Augmentation 2")
    


    ############################################################################
    # FINETUNING RESNET18

    from torchvision import models

    transform = t.augmentation3() # same as augmentation2 but different img size

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4,drop_last=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4,drop_last=True)
    

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, D_out)

    model_ft = model_ft.to(device)

    nets.train_net(model_ft,trainloader,testloader,10,learning_rate, "ResNet18")

    ########################################################################################
    plt.show()


if __name__ == '__main__':
     main()

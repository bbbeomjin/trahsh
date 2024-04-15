
import dataset
from dataset import MNIST
from model import LeNet5, CustomMLP
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer,tst_loader,max_epoch,batch_size):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
   
    avg_loss=0
    for e in range (max_epoch):
        model.train()
        total_correct = 0
        for i, (images,labels) in enumerate(trn_loader):
            images=images.to("cuda")
            labels=labels.to("cuda")
            optimizer.zero_grad()
            output = model(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            loss = criterion(output,labels)
            avg_loss += loss.sum()
            loss.backward()
            optimizer.step()
        acc = 100 * float(total_correct) / torch.tensor(60000)
        avg_loss = avg_loss / len(trn_loader)
        print('Epoch: [{0}]\t'
        'Train Accuracy {acc:.4f}\t'
        'Train Loss {loss:.4f} \t'.format(
        e, acc=acc,loss=avg_loss, ))


        t_loss,acc = test(model,tst_loader,device,criterion,e,batch_size)
    # write your codes here

    return loss, acc

def test(model, tst_loader, device, criterion,epoch=None,batch_size=None):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(tst_loader):
        images=images.to("cuda")
        labels=labels.to("cuda")

        output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(tst_loader)
  
    acc = 100 * float(total_correct) / torch.tensor(10000)

    if(epoch is not None):
        print('Epoch: [{0}]\t'
        'Test Accuracy {acc:.4f}\t'
        'Test Loss {loss:.4f} \t'.format(
        epoch, acc=acc,loss=avg_loss, ))
    else:
         print('Test Accuracy {acc:.4f} %\t'
        'Test Loss {loss:.4f} \t'.format(
        acc=acc,loss=avg_loss, ))

    # write your codes here

    return avg_loss, acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def main():
    """ Main function
        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss
    """
    #device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = "./data/train"
    test_dir = "./data/test"
    #dataset param
    batch_size = 512
    shuffle = True
    num_workers = 16
    
    train_set = MNIST(train_dir)
    test_set = MNIST(test_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #import model
    model = LeNet5()
    model.cuda()
    
    #check params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LeNet5 model parameter num : {total_params}")
    

    # train param
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
   
    epoch = 30
    # LeNet train start

    print("LeNet train start")
    model.train()
    train(model,train_loader,device,criterion,optimizer,test_loader,epoch,batch_size)

    ##test start
    acc ,loss = test(model,test_loader,device,criterion,batch_size=batch_size)
    print("LeNet train done")



    model_2 = CustomMLP()
    model_2.cuda()
    model_2.train()
    criterion_2 = nn.CrossEntropyLoss()
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01,momentum=0.9)
   
    #check params
    total_params = sum(p.numel() for p in model_2.parameters())

    print(f"LeNet5 model parameter num : {total_params}")
    print("Custom train start")
    train(model_2,train_loader,device,criterion_2,optimizer_2,test_loader,epoch,batch_size)
    acc ,loss = test(model_2,test_loader,device,criterion_2,batch_size=batch_size)
    print("Custom train done")

if __name__ == '__main__':
    main()


import dataset
from dataset import MNIST
from model import LeNet5
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchsummary import summary
# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer,tst_loader,max_epoch):
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
    for e in range (max_epoch):
        model.train()
        for i, (images,labels) in enumerate(trn_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss_val = loss.float()
            loss.backward()
            optimizer.step
            if(i%10==0):
                print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss:.4f} \t'.format(
                e, i, len(trn_loader), loss=loss_val))
        test(model,tst_loader,device,criterion,e)
    # write your codes here

    return trn_loss, acc

def test(model, tst_loader, device, criterion,epoch=None):
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
        output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)

    if(epoch is not None):
        print('Epoch: [{0}]\t'
        'Test Accuracy: {acc:.4f}'
        'Loss {loss.val:.4f} \t'.format(
        epoch, acc=acc,loss=avg_loss, ))
    else:
        print('Test Accuracy: {acc:.4f}'
        'Loss {loss.val:.4f} \t'.format(
        acc=acc, loss=avg_loss, ))

    # write your codes here

    return tst_loss, acc

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
    train_dir = "./data/train"
    test_dir = "./data/test"
    #dataset param
    batch_size = 512
    shuffle = False
    num_workers = 16

    train_set = MNIST(train_dir)
    test_set = MNIST(test_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #import model
    model = LeNet5()
    model.train()
    model.cuda()
    # train param
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 16
    ## train start
    print("train start")
    train(model,train_loader,device,criterion,optimizer,test_loader,epoch)

    ##test start
    test(model,test_loader,device,criterion)
    #save weight
    save_path = "weight"
    save_checkpoint({
                'state_dict': model.state_dict(),
            }, True, filename=os.path.join(save_path,'LeNet5.pt'))
    print("train done")
if __name__ == '__main__':
    main()

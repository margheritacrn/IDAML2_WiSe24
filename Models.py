from torch.optim import Adam, SGD
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import f1_score

def init_weights(l):
        """Weights initialization for linear layers"""
        if isinstance(l, nn.Linear):
            torch.nn.init.xavier_normal_(l.weight) 
            l.bias.data.fill_(0.01)
            print(l.weight.shape, l.bias.shape)


class CNNModel1(nn.Module):#input has size 3x128x128
    def __init__(self, loss_fn, lr=.001, epochs=50, input_channels=3, scheduler=False, fc_dropout_rate=0.5, fc_size=512, channels1=32, channels2=64, weight_decay=0, optim_type='ADAM'): 
        super().__init__() 
        self.numEpochs=epochs
        self.loss_fn=loss_fn
        self.lr=lr
        self.scheduler=scheduler
        self.input_channels=input_channels
        self.channels1=channels1
        self.channels2=channels2
        self.fc_dropout_rate=fc_dropout_rate
        self.fc_size=fc_size
        self.weight_decay=weight_decay
        self.optim_type=optim_type
        self.convolutions = nn.Sequential(nn.Conv2d(in_channels=self.input_channels, out_channels=self.channels1, kernel_size=(7,7), padding=3, bias=False),  #32x128x128
                                  nn.BatchNorm2d(self.channels1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2),  #32x64x64
                                  nn.Dropout(p=0.2),
                                  nn.Conv2d(in_channels=self.channels1, out_channels=self.channels2, kernel_size=(3,3), padding=1, stride=2) ,  #64x16x16
                                  nn.BatchNorm2d(num_features=self.channels2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=self.channels2, out_channels=self.channels2, kernel_size=(3,3),  padding=1, stride=2),  #64x16x16
                                  nn.BatchNorm2d(num_features=self.channels2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=self.channels2, out_channels=self.channels2, kernel_size=(3,3),padding='same'), #64x32x32
                                  nn.BatchNorm2d(num_features=self.channels2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=self.channels2, out_channels=self.channels2, kernel_size=(3,3), padding='same'),  #64x32x32
                                  nn.BatchNorm2d(num_features=self.channels2),
                                  nn.ReLU()
                                )  
        
        self.avg_pool=nn.AvgPool2d(kernel_size=2)
        self.dense = nn.Sequential(
                                  nn.Linear(self.channels2*64, self.fc_size),
                                  nn.ReLU(),
                                  nn.Dropout(p=self.fc_dropout_rate),
                                  nn.Linear(self.fc_size, 14))
        self.dense.apply(init_weights)
    def forward(self, x):
        x_conv=self.convolutions(x)
        #avg pooling
        x_conv=self.avg_pool(x_conv)
        #fully connected layers
        x_fc=self.dense(torch.flatten(x_conv, start_dim=1)) #exclude batch dim
        
        return  x_fc
    
    def get_num_params(self):
        return sum([param.numel() for param in self.parameters() if param.requires_grad])
    
    def optimizer(self):
        if(self.optim_type=='ADAM'):
            return Adam(self.parameters(),lr=self.lr, weight_decay=self.weight_decay)  #set weigth_decay for L2 regularization
        else:
            return SGD(self.parameters(),lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)  #set weigth_decay for L2 regularization
    
    def train_epoch(self, dataloader, device,optim, epoch, class_weights, tensorboard_writer=None):
        self.train()
        losses = [] 
        accs = []
        #Train:
        for batch in dataloader: 
            classes=np.array(torch.unique(batch[1]))
            if(len(classes)<14):
                continue
            loss_=self.loss_fn(weight=torch.Tensor(class_weights))  #Add weighting to the loss 
            optim.zero_grad() 
            input_features = batch[0].to(device)
            target = nn.functional.one_hot(batch[1], 14).to(device)  #One hot  encode for cross entropy loss
            logits = self(input_features)  #Model's output. To get probabilities: call softmax on logits. The chosen loss fun expects the logits.
            ce = loss_(logits, target.float()) # Compute the loss
            ce.backward() # Backpropagate and calculate gradients, this go in reverse into the computational graph--> gradient for each weight matrix
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1) #Clip gradients to avoid large gradients. Clipping divides by gradient's norm, thus the direction doesn't change.
            optim.step() # Update model parameters (weights): gradient descent update step. 
            losses.append(ce.detach().cpu().item()) #Transform loss to scalar.
            predicted_classes = logits.argmax(axis=1)  #Take as predicted class the one with highest probability.
            observed_classes = target.argmax(axis=1)
            sample_class_weights=[class_weights[i] for i in observed_classes]  #To compute f1 score.
            f1_train=f1_score(y_true=observed_classes, y_pred=predicted_classes, average='weighted', sample_weight=sample_class_weights)
            accs.append(f1_train)
            if(tensorboard_writer):
                tensorboard_writer.add_scalar('training loss',
                                np.mean(losses),
                                epoch)
        return [np.mean(losses), np.mean(accs)]
    
    def test_epoch(self, dataloader,device, epoch, class_weights, tensorboard_writer=None):
        self.eval() # set the model in evaluation mode--> this deactivates the dropout.
        losses = []
        accs = []
        #Evaluate:
        with torch.no_grad(): # At evaluation time we don't have to update params--> no grad needed.
            for batch in dataloader:
                if(len(torch.unique(batch[1]))<14):
                    continue
                loss_=self.loss_fn() 
                input_features = batch[0].to(device)
                target = nn.functional.one_hot(batch[1], 14).to(device)
                logits = self(input_features)
                ce = loss_(logits, target.float())
                losses.append(ce.detach().cpu().item())
                predicted_classes = logits.argmax(axis=1)
                observed_classes = target.argmax(axis=1)
                sample_class_weights=[class_weights[i] for i in observed_classes]
                f1_val=f1_score(y_true=observed_classes, y_pred=predicted_classes, average='weighted', sample_weight=sample_class_weights)
                accs.append(f1_val)
                tensorboard_writer.add_scalar('test loss',
                                    np.mean(losses),
                                    epoch)
        return [np.mean(losses), np.mean(accs)]
    
    def fit(self, train_dataloader, test_dataloader, device, class_weights_train, class_weights_test, tensorboard_writer=None):
        optim=self.optimizer()
        train_lss=[]
        test_lss=[]
        train_accs=[]
        test_accs=[]
        scheduler=ExponentialLR(optim,gamma=0.9)
        for epoch in range(self.numEpochs): #train and evaluate model
            train_loss = self.train_epoch(train_dataloader, device,optim, epoch,class_weights_train, tensorboard_writer)
            if(self.scheduler):
                scheduler.step(train_loss[0])
            test_loss = self.test_epoch(test_dataloader, device,epoch,  class_weights_test, tensorboard_writer)
            train_lss.append(train_loss[0])
            train_accs.append(train_loss[1])
            test_accs.append(test_loss[1])
            test_lss.append(test_loss[0])
            print(f"epoch: {epoch + 1}/{self.numEpochs}, train_f1_score: {train_loss[1]}, test_f1_score: {test_loss[1]}")
            #checkpoint
            torch.save(self.state_dict(), "net_weights_epoch_"+str(epoch)+".pth")
            
        return train_accs, train_lss, test_accs, test_lss
    
    def predict(self,input,device):
        self.eval() 
        input=input.to(device)
        logits = self(input).detach()
        pred=logits.argmax(axis=1).numpy()
        return pred
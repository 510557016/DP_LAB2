import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from matplotlib import pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
        # in_channels：输入的通道数目
        # out_channels： 输出的通道数目 
        # kernel_size：卷积核的大小，类型为int 或者元组，当卷积是方形的时候，只需要一个整数边长即可，卷积不是方形，要输入一个元组表示 高和宽。
        # stride： 卷积每次滑动的步长为多少，默认是 1 
        # padding： 设置在所有边界增加 值为 0 的边距的大小（也就是在feature map 外围增加几圈 0 ），例如当 padding =1 的时候，如果原来大小为 3 × 3 ，那么之后的大小为 5 × 5 。即在外围加了一圈 0 
        # groups：控制输入和输出之间的连接
        # bias： 是否将一个 学习到的 bias 增加输出中，默认是 True
        # padding_mode ： 字符串类型，接收的字符串只有 “zeros” 和 “circular”

        # class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        # y=((x-mean(x))/(var(x))^1/2+eps)*gamma+beta 
        # num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
        # eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5
        # momentum：一个用于运行过程中均值和方差的一个估计参数（我的理解是一个稳定系数，类似于SGD中的momentum的系数）
        # affine：当设为true时，会给定可以学习的系数矩阵gamma和beta
        
        # class torch.nn.ReLU(inplace=False)
        # inplace – can optionally do the operation in-place. Default: False
        
        # class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # kernel_size(int or tuple) - max pooling的窗口大小
        # stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
        # padding(int or tuple, optional) - 输入的每一条边补充0的层数
        # dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
        # return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
        # ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

        # ===========================
        # TODO 1: build your network
        # (weight-kernel+1)/stride+1 無條件進位
                      
        # Convolution 1 , input_shape=(3,256,256)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.relu1 = nn.ReLU() 

        # Max pool 1 ,  input_shape=(64,256,256)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolution 2 , input_shape=(64,128,128)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        self.relu2 = nn.ReLU()

        # Convolution 3 , input_shape=(128,128,128) 
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1) 
        self.relu3 = nn.ReLU()  
        
        # Convolution 4 , input_shape=(512,128,128)
        self.conv4 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1) 
        self.relu4 = nn.ReLU()
        self.drout1 = nn.Dropout2d(0.5)
        
        # Convolution 5 , input_shape=(512,128,128)
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=128, kernel_size=3, stride=1, padding=1) 
        self.relu5 = nn.ReLU()

        # Convolution 6 , input_shape=(128,128,128)
        self.conv6 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.relu6 = nn.ReLU() 

        # Convolution 7 , input_shape=(64,128,128)
        self.conv7 = nn.Conv2d(in_channels=64,out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.relu7 = nn.ReLU()  
        self.drout2 = nn.Dropout2d(0.5)
       
        # Fully connected 1 ,#input_shape=(32*128*128)
        self.fc1 = nn.Linear(in_features=32 * 128 * 128, out_features=120)       
        self.bnfc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)
          
        #CLASStorch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        #in_features指的是输入的二维张量的大小
        #out_features指的是输出的二维张量的大小
        #bias – If set to False, the layer will not learn an additive bias. Default: True
        
        #CLASStorch.nn.Dropout(p=0.5, inplace=False)
        #Dropout 防止过拟合 & 数据增    
        #self.fc = nn.Linear(8 * 50 * 50, 2)
         
    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        # please write down the output size of each layer
        # example:
        # out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)
       
       
        out = self.conv1(x) 
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        out = self.conv2(out) 
        out = self.relu2(out)
        
        out = self.conv3(out) 
        out = self.relu3(out)
                        
        out = self.conv4(out)        
        out = self.relu4(out)
        out = self.drout1(out) 
        
        out = self.conv5(out)        
        out = self.relu5(out)
                 
        out = self.conv6(out)        
        out = self.relu6(out)

        out = self.conv7(out)        
        out = self.relu7(out)
        out = self.drout2(out)
             
        out = torch.flatten(out, 1)       
        out = self.fc1(out) 
        out = self.bnfc1(out) 
        out = self.fc2(out)
        out = nn.functional.softmax(out, dim=1)          
        # ========================
        return out   

def train(model,device,n_epochs,train_loader,valid_loader,criterion,optimizer):
    train_acc_his=[]
    train_losses_his=[]

    valid_acc_his=[]
    valid_losses_his=[]
   
    for epoch in range(1, n_epochs+1):        
        # keep track of training
        train_loss = 0.0
        train_losses = []
        train_correct = 0
        train_total = 0

        valid_loss = 0.0
        valid_losses=[]
        val_correct = 0
        val_total = 0
                
        #torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        #size：定义输出张量形状的整数序列。可以是可变数量的参数，也可以是列表或元组之类的集合。
        #out：指定输出的tensor。
        #dtype：指定返回tensor中数据的类型，如果为None，使用默认值（一般为torch.float32可以使用 torch.set_default_tensor_type()更改。）
        #layout：返回tensor所需要的布局，默认为strided（密集型张量），还有torch.sparse_coo 稀疏性张量，用于存储稀疏矩阵时使用的布局。
        #device：指定返回tensor所处的设备，可以是cpu或者cuda，如果不指定则为默认（一般为cpu，可以使用torch.set_default_tensor_type()进行更改。）
        #requires_grad：指定返回的tensor是否需要梯度，默认为False
        
        train_pred = torch.zeros(10,1)
        train_target = torch.zeros(10,1)
        val_pred = torch.zeros(10,1)
        val_target = torch.zeros(10,1)
        count=0
        count2=0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        # ===============================
        # TODO 3: switch the model to training mode
        model.train()
        # ===============================
        #train_loader = total * 70 % = 2500 * 0.7 = 1750
        #BATCH_SIZE = 10 , total step = 1750 / 10 = 175
        step = 0
        for data, target in train_loader:
            step = step + 1           
            # move tensors to  device
            data, target = data.to(device), target.to(device)
            # =============================================
            # TODO 4: initialize optimizer to zero gradient
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # =================================================
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # =================================================
            # TODO 5: loss -> backpropagation -> update weights
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            optimizer.step()
            # =============================================
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            #print("pred=",pred)
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            #print("target.data.view_as(pred)=",target.data.view_as(pred))
            train_total += data.size(0)
            train_losses.append(loss.item()*data.size(0))            
            
            print("train_step=",step)
            #print("train_correct=",train_correct)
            #print("train_total=",train_total)
            #print("train_losses=",loss.item()*data.size(0))
            
            if count==0:
                train_pred=pred
                train_target=target.data.view_as(pred)
                count=count+1
            else:
                train_pred=torch.cat((train_pred,pred), 0)
                train_target=torch.cat((train_target,target.data.view_as(pred)), 0)
            
        train_pred=train_pred.cpu().view(-1).numpy().tolist()
        train_target=train_target.cpu().view(-1).numpy().tolist()

        ##########################    
        # validate the model #
        ##########################
        # ===============================
        # TODO 6: switch the model to validation mode
        model.eval()
        # ===============================
        with torch.no_grad():
          step=0
          for data, target in valid_loader:
              step = step + 1
              data, target = data.to(device), target.to(device)
              # forward pass: compute predicted outputs by passing inputs to the model
              output = model(data)
              # calculate the batch loss
              loss =criterion(output, target)
              #calculate accuracy
              pred = output.data.max(dim = 1, keepdim = True)[1]
              val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
              val_total += data.size(0)
              valid_losses.append(loss.item()*data.size(0))

              print("validate_step=",step)
              #print("validate_correct=",val_correct)
              #print("validate_total=",val_total)
              #print("validate_losses=",loss.item()*data.size(0))
            
              if count2==0:
                  val_pred=pred
                  val_target=target.data.view_as(pred)
                  count2=count2+1
              else:
                  val_pred=torch.cat((val_pred,pred), 0)
                  val_target=torch.cat((val_target,target.data.view_as(pred)), 0)
              
          val_pred=val_pred.cpu().view(-1).numpy().tolist()
          val_target=val_target.cpu().view(-1).numpy().tolist()

        # ================================
        # TODO 8: calculate accuracy, loss    
        # calculate average losses          
        train_loss=np.average(train_losses)
        valid_loss=np.average(valid_losses)
        
        # calculate average accuracy
        train_acc=train_correct/train_total
        valid_acc=val_correct/val_total

        train_acc_his.append(train_acc)
        train_losses_his.append(train_loss)
        valid_acc_his.append(valid_acc)
        valid_losses_his.append(valid_loss)

    return train_acc_his,train_losses_his,valid_acc_his,valid_losses_his,model

def main():
    # ==================
    # TODO 9: set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # =================

    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
        #學習率
    LEARNING_RATE = 0.001
        #每批丟入多少張圖片   
    BATCH_SIZE = 10
        #訓練資料路徑
    TRAIN_DATA_PATH = '/content/drive/MyDrive/train'
        #驗證資料路徑
    VALID_DATA_PATH = '/content/drive/MyDrive/train'
    TEST_DATA_PATH = '/content/drive/MyDrive/test_data'
    EPOCHS = 40
    MODEL_PATH = '/content/drive/MyDrive/train/model.pt'

    # train_transform 進行影像強化提高資料多樣性 
    # valid_transform 保持驗證公平性只採用調整大小
    
    # transforms.Compose
    # Resize                把给定的图片resize到given size
    # Normalize             用均值和标准差归一化张量图像
    # ToTensor              convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
    # CenterCrop            在图片的中间区域进行裁剪
    # RandomCrop            在一个随机的位置进行裁剪
    # FiceCrop              把图像裁剪为四个角和一个中心
    # RandomResizedCrop     将PIL图像裁剪成任意大小和纵横比
    # ToPILImage            convert a tensor to PIL image
    # RandomHorizontalFlip  以0.5的概率水平翻转给定的PIL图像
    # RandomVerticalFlip    以0.5的概率竖直翻转给定的PIL图像
    # Grayscale             将图像转换为灰度图像
    # RandomGrayscale       将图像以一定的概率转换为灰度图像
    # ColorJitter           随机改变图像的亮度对比度和饱和度

    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        #把灰階[0,255]變換成[0,1]
        transforms.ToTensor(),
        #transforms.Normalize(mean,std) 
        #image=(image-mean)/std 
        #0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1. 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    train_data = datasets.ImageFolder(TRAIN_DATA_PATH, transform=train_transform)
    #print(train_data.class_to_idx)
    valid_data = datasets.ImageFolder(VALID_DATA_PATH, transform=train_transform)
    test_data = datasets.ImageFolder(TEST_DATA_PATH, transform=train_transform)
    # =================

    #切分70%當作訓練集、30%當作驗證集
    #train_size = int(0.7 * len(train_data))
    #valid_size = len(train_data) - train_size
    #任意分配圖檔
    #train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    # Dataloader可以用Batch的方式訓練 shuffle = True 表示把資料打亂
    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE,shuffle=True)
    # ============================

    # build model, criterion and optimizer
    model_1 = Net().to(device)
    #定義損失函數及learn rate
    #CLASStorch.optim.Optimizer(params, defaults)
    #Adadelta   Implements Adadelta algorithm.
    #Adagrad    Implements Adagrad algorithm.
    #Adam       Implements Adam algorithm.
    #AdamW      Implements AdamW algorithm.
    #SparseAdam Implements lazy version of Adam algorithm suitable for sparse tensors.
    #Adamax     Implements Adamax algorithm (a variant of Adam based on infinity norm).
    #ASGD       Implements Averaged Stochastic Gradient Descent.
    #LBFGS      Implements L-BFGS algorithm, heavily inspired by minFunc.
    #NAdam      Implements NAdam algorithm.
    #RAdam      Implements RAdam algorithm.
    #RMSprop    Implements RMSprop algorithm.
    #Rprop      Implements the resilient backpropagation algorithm.
    #SGD        Implements stochastic gradient descent (optionally with momentum).
     
    # ================================
    # TODO 14: criterion and optimizer
    #optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)
    optimizer = torch.optim.SGD(model_1.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # ================================
    
    print("===== start train =====")
    train_acc_his,train_losses_his,valid_acc_his,valid_losses_his,model = train(model_1,device,EPOCHS,train_loader,valid_loader,criterion,optimizer)
    print("===== end train =====")
    print("===== test result =====")
    print("train_acc_his=",train_acc_his)
    print("train_losses_his=",train_losses_his)
    print("valid_acc_his=",valid_acc_his)
    print("valid_losses_his=",valid_losses_his)
    print("model=",model)

    #valid_acc_his,valid_losses_his = validation(model_1,device,EPOCHS,valid_loader,criterion)
    #print("valid_acc_his=",valid_acc_his)
    #print("valid_losses_his=",valid_losses_his)

    # ==================================
    # TODO 15: save the model parameters
    torch.save(model, MODEL_PATH)
    # ==================================

    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plot
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    plt.plot(train_acc_his, 'bo', label = 'training accuracy')
    plt.plot(valid_acc_his, 'r', label = 'validation accuracy')
    plt.title("Simple CNN Accuracy")
    plt.legend(loc='upper left')
    plt.legend()
    plt.subplot(222)
    plt.plot(train_losses_his, 'bo', label = 'training loss')
    plt.plot(valid_losses_his, 'r', label = 'validation loss')
    plt.title("Simple CNN Loss")
    plt.legend(loc='upper left')
    plt.legend()
    plt.show()
    # =========================================
    
if __name__ == '__main__':
    main()

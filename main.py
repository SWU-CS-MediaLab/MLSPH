import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from torch.utils.data import DataLoader
from config import opt
from data_handler import *
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import Txt_net
from models.resnet import resnet34
from utils import calc_map_k
import torch.nn.functional as Funtional1
from dataset.Mirflckr25K.dataset import DatasetMirflckr25KTrain,DatasetMirflckr25KValid
from valid import valid
from visdom import Visdom
import scipy.io as sio

def train():
    img_model = resnet34(opt.bit)
    txt_model = Txt_net(opt.y_dim,opt.bit)

    if opt.use_gpu:
        img_model = img_model.cuda()  
        txt_model = txt_model.cuda()  


    train_data = DatasetMirflckr25KTrain(opt.img_dir, opt.imgname_mat_dir,opt.tag_mat_dir,opt.label_mat_dir, batch_size=opt.batch_size, train_num=opt.training_size, query_num=opt.query_size)
    valid_data = DatasetMirflckr25KValid(opt.img_dir,opt.imgname_mat_dir,opt.tag_mat_dir,opt.label_mat_dir, query_num=opt.query_size)

   
    num_train = len(train_data)
    train_L = train_data.get_all_label()

    F_buffer = torch.randn(num_train, opt.bit)  
    G_buffer = torch.randn(num_train, opt.bit)  

    if opt.use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    
    B = torch.sign(F_buffer + G_buffer)

    batch_size = opt.batch_size  # 128

    lr = opt.lr 
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)

    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)  
    ones_ = torch.ones(num_train - batch_size, 1)  
    unupdated_size = num_train - batch_size  

    max_mapi2t = max_mapt2i = 0.
    lossResult=np.zeros([2,4,opt.max_epoch])
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=4,drop_last=True)
    for epoch in range(opt.max_epoch):  
        # train image net
        train_data.img_load()
        train_data.re_random_item()
        for data in tqdm(train_loader): 
            ind = data['index'].numpy()
            unupdated_ind = np.setdiff1d(range(num_train), ind)  

            sample_L = data['label']
            image = data['img']
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()  
                ones_ = ones_.cuda()  

           
            S = calc_neighbor(sample_L, train_L)  
            cur_f = img_model(image)  
            F_buffer[ind, :] = cur_f.data 
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            theta_x =calc_inner(cur_f,G)
            logloss_x=opt.alpha*torch.sum(torch.pow(S-theta_x,2))/(num_train * batch_size)
            theta_xx=calc_inner(cur_f,F)
            logloss_xx=opt.beta*torch.sum(torch.pow(S-theta_xx,2))/(num_train * batch_size)
            quantization_x = opt.gamma * torch.sum(torch.pow(B[ind, :] - cur_f, 2))/(batch_size * opt.bit)          
            loss_x = logloss_x + logloss_xx + quantization_x
            loss_x=10*loss_x

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

        # train txt net
        train_data.txt_load()
        train_data.re_random_item()
        for data in tqdm(train_loader): 
            ind = data['index'].numpy()
            unupdated_ind = np.setdiff1d(range(num_train), ind)  

            sample_L = data['label']
            text = data['txt']
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            
            S = calc_neighbor(sample_L, train_L) 
            cur_g= txt_model(text)  
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            # calculate loss
            theta_y = calc_inner(cur_g,F)
            logloss_y=opt.alpha*torch.sum(torch.pow(S-theta_y,2))/(num_train * batch_size)
            theta_yy=calc_inner(cur_g,G)
            logloss_yy=opt.beta*torch.sum(torch.pow(S-theta_yy,2))/(num_train * batch_size)
            quantization_y = opt.gamma *torch.sum(torch.pow(B[ind, :] - cur_g, 2))/(batch_size*opt.bit)
            loss_y = logloss_y + logloss_yy +  quantization_y
            loss_y=10*loss_y


            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

        print('...epoch: %3d, ImgLoss:%3.3f,TxtLoss:%3.3f,lr: %f' % (epoch + 1, loss_x, loss_y, lr))
        lossResult[0,:,epoch]=[logloss_x,logloss_xx,quantization_x,loss_x]
        lossResult[1,:,epoch]=[logloss_y,logloss_yy,quantization_y,loss_y]
        # update B
        B = torch.sign(F_buffer + G_buffer)

        if opt.valid:
            mapi2t, mapt2i = valid(opt, img_model, txt_model, valid_data)
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:  
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f;MAX_MAP(i->t): %3.4f, MAX_MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i,max_mapi2t,max_mapt2i))

        lr = learning_rate[epoch + 1]

        # set learning rate
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr

    print('...training procedure finish')
    if opt.valid:  #
        print('max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
    else:
        mapi2t, mapt2i = valid(opt, img_model, txt_model, valid_data)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i

    viz=Visdom(env='my_loss')
    viz.line(X=np.arange(opt.max_epoch)+1,
             Y=np.column_stack((lossResult[0,0,:],lossResult[0,1,:],lossResult[0,2,:],lossResult[0,3,:])),
             opts=dict(
                 showlegend=True,
                 legend=['logloss_x','logloss_xx','quantity_x','loss_x'],
                 title='image loss',
                 xlabel='epoch number',
                 ylabel='loss value',
             ))
    viz.line(X=np.arange(opt.max_epoch) + 1,
             Y=np.column_stack((lossResult[1, 0, :], lossResult[1, 1, :], lossResult[1, 2, :], lossResult[1, 3, :])),
             opts=dict(
                 showlegend=True,
                 legend=['logloss_y', 'logloss_yy', 'quantity_y', 'loss_y'],
                 title='text loss',
                 xlabel='epoch number',
                 ylabel='loss value',
             ))

def calc_neighbor(label1, label2):
    # calculate the similar matrix
    if opt.use_gpu:
        label1 = label1.float()
        label2 = label2.float()
        Sim = label1.matmul(label2.transpose(0, 1)).type(torch.cuda.FloatTensor)
    else:
        Sim = label1.matmul(label2.transpose(0, 1)).type(torch.FloatTensor)

    numLabel_label1 = torch.sum(label1, 1)
    numLabel_label2 = torch.sum(label2, 1)

    x = numLabel_label1.unsqueeze(1) + numLabel_label2.unsqueeze(0) - Sim
    Sim = 2 * Sim / x  # [0,2]

    # cosine similarity
    # label1 = myNormalization(label1)
    # label2 = myNormalization(label2)
    # Sim = (label1.unsqueeze(1) * label2.unsqueeze(0)).sum(dim=2)

    # print(torch.max(Sim))
    # print(torch.min(Sim))
    return Sim


def myNormalization(X):
    x1=torch.sqrt(torch.sum(torch.pow(X, 2),1)).unsqueeze(1)
    return X/x1

def calc_inner(X1,X2):
    X1=myNormalization(X1)
    X2=myNormalization(X2)
    X=torch.matmul(X1,X2.t())  # [-1,1]
   
    return X


if __name__ == '__main__':
    train()

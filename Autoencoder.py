import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader,Dataset
# from torchsummary import summary
import os
import torch
import cv2 as cv
import time
import numpy as np
from torchvision import  transforms
import matplotlib.pyplot as plt
from random import sample,seed
from easydict import EasyDict
config = EasyDict()
config.num_epochs = 1
image_name = 'peaper.tiff'   # 'peaper.tiff'   'babon.tiff'
path = 'D:/nafise/nafise/book_dr_moatar/papers for new article/random_cgr/new/'
############################################################################################
def CGR_():
    im = cv.imread(os.path.join(path , image_name),0)
    row= col = 512
    im = cv.resize(im, (row,col))
    midpoint =0.5, 0.5
    midpoint = list(midpoint)

    x = []
    y = []

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            binary = '{0:b}'.format(im[i][j])
            list_binary = list(binary)
            if len(list_binary) < 8:
                count = 8- len(list_binary)
                for a in range(count):
                    list_binary.insert(0,'0')
            for k in range(0,len(list_binary),2):
                n = ''.join((list_binary[k],list_binary[k+1]))
                if n == '00':
                    midpoint[0],midpoint[1] = (0+midpoint[0])/2, (0+ midpoint[1])/2
                    x.append(midpoint[0])
                    y.append(midpoint[1])

                elif n == '10':
                    midpoint[0],midpoint[1] = (1+midpoint[0])/2,(0+ midpoint[1])/2
                    x.append(midpoint[0])
                    y.append(midpoint[1])


                elif n == '01':
                    midpoint[0], midpoint[1]  = (0+midpoint[0])/2,(1+midpoint[1])/2
                    x.append(midpoint[0])
                    y.append(midpoint[1])


                elif n == '11':
                    midpoint[0], midpoint[1]  = (1+midpoint[0])/2, (1+midpoint[1])/2
                    x.append(midpoint[0])
                    y.append(midpoint[1])

    return x,y, row, col
##chaotic function #############################################################################################
def logistic(R, x0, N):
    x = x0
    x_list = [x0]
    for i in range(N - 1):
        x = R * x * (1. - x)
        x_list.append(x)
    return x_list

def xor_key():
    x, y, row, col= CGR_()
    x = [int(x[i] * 256) for i in range(row*row)]
    y = [int(y[i] * 256) for i in range(col*col)]
    CGR_sequence = np.hstack((x,y))
    seed(1)
    key1 = sample(list(CGR_sequence), row*col)
    key2 = logistic(4.0, 0.0000003, row*col)
    key2 = [int(key2[i] * 256) for i in range(len(key2))]

    im =  cv.imread(path+ image_name,0)
    cipher_image = np.zeros((row,col), np.uint8)
    cipher_image2 = np.zeros((row,col), np.uint8)
    c = 0
    for i in range(row):
        for j in range(col):
            binary_im = bin(im[i][j])
            key_binary = bin(key1[c])
            key_binary2 = bin(key2[c])
            cipher_image[i][j] = (int(key_binary[2:],2) ^ int(binary_im[2:],2))
            cipher_image2[i][j] = (int(key_binary2[2:],2) ^ int(bin(cipher_image[i][j])[2:],2))
            c+=1

    cv.imwrite(os.path.join(path, 'cipher_image_' + image_name), cipher_image2)
    return key1,key2

#######################################################################################################################
class TrainTest:
    def __init__(self, args, model: torch.nn.Module, train_dataset: Dataset):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)

        '''dataset and dataloader'''
        self.train_dataset = train_dataset

        tfms = transforms.Compose([
            transforms.ToTensor()])
        self.train_dataset = tfms(self.train_dataset)

    def train(self):
        data =  self.train_dataset
        data = data.detach()
        preds, cipher_image = self.model(data)
        cipher_image = cipher_image.squeeze(0).detach().numpy().astype(np.uint8)
        preds = preds.squeeze(0).cpu().detach().numpy()
        preds = ((preds + 1) / 2.0 * 255.0).astype(np.uint8)  # normalize

        return preds, cipher_image

########################################################################
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3),
            nn.ReLU(True),

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16,3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded



#revers#############################################################################
def reverse(key1, key2):
    seed(1)
    im =  cv.imread(os.path.join(path,'cipher_image2_'+image_name),0)
    print(im.shape)
    row = col = 512
    original_image1 = np.zeros((row,col), np.uint8)
    original_image2 = np.zeros((row,col), np.uint8)

    c = 0
    for i in range(row):
        for j in range(col):
            binary_im = bin(im[i][j])
            key_binary = bin(key1[c])
            key_binary2 = bin(key2[c])
            original_image2[i][j] = (int(key_binary2[2:], 2) ^ int(binary_im[2:], 2))
            original_image1[i][j] = (int(key_binary[2:], 2) ^ int(bin( original_image2[i][j])[2:], 2))

            c += 1


    cv.imwrite(os.path.join(path, 'decoded_,' + image_name), original_image1)

##############################################################################################
if __name__ == '__main__':
    start_time = time.time()
    key1,key2 = xor_key()
    model = autoencoder()
    # Save decoder weights
    torch.save(model.decoder.state_dict(), 'decoder_weights.pth')
    # print(model.parameters)
    # print(summary(model, (1,225,225)))
    cipher_image=  cv.imread(os.path.join(path,'cipher_image_'+ image_name),0)
    train_test = TrainTest(config, model, cipher_image)
    output, cipher_image2 = train_test.train()
    loss = cipher_image-output
    initial_image = (loss + output)
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    print(f"Execution Time: {execution_time_ms:.2f} milliseconds")


    cv.imwrite(os.path.join(path, 'cipher_image2_' + image_name), initial_image)  # save output image from decoder+loss
    cv.imwrite(os.path.join(path, 'encoded_' + image_name), cipher_image2)  # save output image fron encoder
    cv.imwrite(os.path.join(path, 'loss_' + image_name), loss)  # save loss image

    reverse(key1, key2)







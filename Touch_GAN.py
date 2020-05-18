# Using GAN for generating adversial keystroke dynamics
import numpy as np
import pandas as pd
import argparse
import os
import math
import sys
import csv
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from Touch_SVM import SVMDetector
from Touch_SVM import normalize_df
from Keystroke_GMM import GMMDetector
#from Keystroke_LSVM import LSVMDetector


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--vector_size", type=int, default=13, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.001, help="lower and upper clip value for disc. weights")
#parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

'''
To Do Generator
'''


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # Create Neural Network
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *block(input_len, 256, normalize=False),
            *block(256, 64, normalize=False),
            nn.Linear(64, 13),
            # nn.LeakyReLU(0.2, inplace=True)
            # nn.Softplus(beta=5, threshold=2)
            # nn.ReLU()
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, subject):
        new_vector = self.model(subject)
        return new_vector


'''
To Do Discriminator
'''


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # Create Neural Network for classifying
        self.model = nn.Sequential(
            # nn.BatchNorm1d(Feature_shape, 0.8),
            nn.Linear(Feature_shape, 62),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ReLU(),
            nn.Linear(62, 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(16, 2),
            # nn.Softmax()
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, var):
        # var = var.view(Feature_shape, -1)
        validity = self.model(var)
        # val = validity.transpose(0,1)
        # r = val[1]
        # r = r.view(-1,opt.batch_size)
        # print(len(validity))
        return validity


'''
To Do Blackbox
'''
# class Blackbox():


'''
RMSE
'''


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.batch_size = opt.batch_size

    def forward(self, yhat, y):
        yhat = yhat.transpose(0, 1)
        yhat = yhat.reshape(-1)
        yhat = yhat[:self.batch_size]
        return torch.sqrt(self.mse(yhat, y))


# Create OneHot Vector
def OneHot(target, batch_size):
    if target == 1:
        a = np.array([1] * batch_size)
    else:
        a = np.array([0] * batch_size)

    return torch.from_numpy(a).float()
    # return torch.from_numpy(a).long()


class TransGenerator:

    def ___init__(self, x):
        super().__init__()
        self.x = x

    def forward(self, x):
        self.row = math.ceil(x.shape[0])
        self.col = math.floor(x.shape[1] / 2)
        self.output = np.zeros([self.row, self.col * 3 + 1])

        for j in range(self.row):
            for i in range(self.col):
                self.output[j, i * 3] = x[j, i * 2]
                self.output[j, i * 3 + 1] = x[j, i * 2] + x[j, i * 2 + 1]
                self.output[j, i * 3 + 2] = x[j, i * 2 + 1]

            self.output[j, self.col * 3] = x[j, -1]

        return self.output




opt = parser.parse_args()
print(opt)
Feature_shape = int(opt.vector_size)

#target_subject = "s003"
path = "featMat.csv"
data = pd.read_csv(path)
id = data["user_id"].unique()
subjects = []
for idx,subject in enumerate(id):
    user_data = data.loc[data.user_id == subject, \
             ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
              'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered']]
    #print(user_data.shape,idx)
    if user_data.shape[0] >= 400:
        subjects.append(subject)

right = OneHot(1, opt.batch_size)
left = OneHot(0, opt.batch_size)


#cuda = True if torch.cuda.is_available() else False
cuda = False

# Define the length of random noise
input_len = int(13)
attacker_data = []
for target_subject in subjects:

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    if cuda:
        generator.cuda()
        discriminator.cuda()


    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)


    #optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5,0.999))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5,0.999))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    genuine_user_data = data.loc[data.user_id == target_subject, ["stroke duration", 'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
                                                                 'length of trajectory', 'mid-stroke pressure', 'mid-stroke area covered',
                                                                 '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
                                                                 '20\%-perc. dev. from end-to-end line', '50\%-perc. dev. from end-to-end line',
                                                                 '80\%-perc. dev. from end-to-end line']]

    genuine_user_data = normalize_df(genuine_user_data[:400])

    train_data = genuine_user_data[:200]
    mean_of_train = np.mean(np.single(train_data.values),axis=0)
    list_mean_of_train = list(mean_of_train)
    train_data = torch.from_numpy(np.single(train_data.values))

    train_sampler = RandomSampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler,
                           batch_size=opt.batch_size)

    batches_done = 0

    #tic = time.time()

    for epoch in range(opt.n_epochs):

        for i, vector in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(vector)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            #z = Variable(Tensor(np.random.normal(list_mean_of_train, 0.1, (opt.batch_size, input_len))))
            z = Variable(Tensor(np.random.normal(0, 2, (opt.batch_size, input_len))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss

            loss_D = criterion(discriminator(real_imgs), right) + criterion(discriminator(fake_imgs), left)
            #loss_D = -torch.mean(torch.log(discriminator(real_imgs))+torch.log(1-discriminator(fake_imgs)))
            #loss_D = torch.mean(torch.log(discriminator(real_imgs)))
            #print(discriminator(real_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = criterion(discriminator(gen_imgs),right)

                loss_G.backward()
                optimizer_G.step()

                # print(
                #    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                #    % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                # )

            #if batches_done % opt.sample_interval == 0:
                #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1




    #print_data = Variable(Tensor(np.random.normal(list_mean_of_train, 0.1, (2, input_len))))
    #print(generator(print_data))

    #t = Variable(Tensor(np.random.normal(list_mean_of_train, 0.1, (250, input_len))))
    t = Variable(Tensor(np.random.normal(0, 2, (250, input_len))))
    #Q = Variable(Tensor(np.random.normal(0, 1, (250, 31))))
    #Q = Q.numpy()
    fake_imgs_1 = generator(t).detach()
    Adversarial_attack_subject2 = fake_imgs_1.numpy()

    #k = TransGenerator().forward(Adversarial_attack_subject2)
    k = Adversarial_attack_subject2
    #k.reshape(1,-1)
    print(SVMDetector(target_subject,data,k).evaluate())
    #print(SVMDetector("s003",data,Q).evaluate())
    #filename = "attacker/" + "data" + target_subject + str(opt.n_epochs) + ".csv"
    #filepath = os.path.join("attacker",filename)
    #np.savetxt(filename, k, delimiter=",")
    #print(SVMDetector(target_subject,data,k).evaluate())
    attacker_data.append(k)
    #toc = time.time()
    #print(toc-tic)


print(SVMDetector(subjects,data,attacker_data).evaluate())



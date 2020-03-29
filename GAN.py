# Using GAN for generating adversial keystroke dynamics
import numpy as np
import pandas as pd
import argparse
import os
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from Keystroke_SVM import SVMDetector

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--vector_size", type=int, default=31, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.001, help="lower and upper clip value for disc. weights")
#parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)
Feature_shape = int(opt.vector_size)


path = "D:\\Master_Thesis\\User-Verification-based-on-Keystroke-Dynamics\\keystroke.csv"
data = pd.read_csv(path)
subjects = data["subject"].unique()

cuda = True if torch.cuda.is_available() else False

# Define the length of random noise
input_len = int(50)
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
            #layers.append(nn.ReLU())
            return layers


        self.model = nn.Sequential(
          *block(input_len,256,normalize=False),
          *block(256,64,normalize=False),
          nn.Linear(64,31),
          #nn.LeakyReLU(0.2, inplace=True)
          nn.Softplus(beta=5,threshold=2)
          #nn.ReLU()
        )

    def forward(self,subject):
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
            nn.BatchNorm1d(Feature_shape, 0.8),
            nn.Linear(Feature_shape, 62),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(),
            nn.Linear(62, 16),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(16, 2),
            #nn.Softmax()
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, var):
        #var = var.view(Feature_shape, -1)
        validity = self.model(var)
        #val = validity.transpose(0,1)
        #r = val[1]
        #r = r.view(-1,opt.batch_size)
        #print(len(validity))
        return validity

'''
To Do Blackbox
'''
#class Blackbox():



'''
LossFunction
'''
'''
class LossFunction(nn.module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self

    def foward(self, inputs, target):
        self.L = len(inputs)
        Target = OneHot(target, self.L)
        entropy = 1
'''

'''
RMSE
'''
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.batch_size = opt.batch_size

    def forward(self, yhat, y):
        yhat = yhat.transpose(0,1)
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
    #return torch.from_numpy(a).long()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
#criterion = CrossEntropyLoss()
#criterion = RMSELoss()
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


genuine_user_data = data.loc[data.subject == "s004", "H.period":"H.Return"]
#imposter_data = data.loc[data.subject != subject, :]
train_data = genuine_user_data[:200]
train_data = torch.from_numpy(np.single(train_data.values))

train_sampler = RandomSampler(train_data)
dataloader = DataLoader(train_data, sampler=train_sampler,
                       batch_size=opt.batch_size)

batches_done = 0

right = OneHot(1, opt.batch_size)
left = OneHot(0, opt.batch_size)


for epoch in range(opt.n_epochs):

    for i, vector in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(vector)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, input_len))))

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
            #loss_G = torch.mean(torch.log(1-discriminator(gen_imgs)))
            loss_G = criterion(discriminator(gen_imgs),right)

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        #if batches_done % opt.sample_interval == 0:
            #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1


'''
for epoch in range(opt.n_epochs):

    for i, vector in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(vector)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()


        loss_D = -torch.mean(torch.log(discriminator(real_imgs)))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        #for p in discriminator.parameters():
        #    p.data.clamp_(-opt.clip_value, opt.clip_value)



        print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] "
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item())
            )

        # if batches_done % opt.sample_interval == 0:
        # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1
'''
#val_data = genuine_user_data[201]

t = Variable(Tensor(np.random.normal(0, 1, (250, input_len))))
Q = Variable(Tensor(np.random.normal(0, 1, (250, 31))))
Q = Q.numpy()
fake_imgs_1 = generator(t)
Adversarial_attack_subject2 = fake_imgs.numpy()
k = pd.DataFrame()
k = Adversarial_attack_subject2
print(SVMDetector("s003",data,k).evaluate())
print(SVMDetector("s003",data,Q).evaluate())
#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np 
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from torch.distributions.normal import Normal
import os



class Encoder(nn.Module): 
    def __init__(self, latent_dim, input_size): 
        super(Encoder, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride=2, padding=1, bias=True)
        self.batchNorm1 = torch.nn.BatchNorm2d(16)
    
        self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride=2, padding=1, bias=True)
        self.batchNorm2 = torch.nn.BatchNorm2d(32)
        
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride=2, padding=1, bias=True)
        self.batchNorm3 = torch.nn.BatchNorm2d(64)
     
        
        self.flatten = nn.Flatten()
        
        self.mean = nn.Linear(64*(input_size // 8)*(input_size // 8), latent_dim)
        self.var = nn.Linear(64*(input_size // 8)*(input_size // 8), latent_dim)
        
        self.sampling = Sampling()
    
    def forward(self, x): 
        x = nn.ReLU()(self.batchNorm1(self.conv1(x)))
        x = nn.ReLU()(self.batchNorm2(self.conv2(x)))
        x = nn.ReLU()(self.batchNorm3(self.conv3(x)))
        #does this get updated
        x = self.flatten(x)
        
        mean = self.mean(x)
        log_var = self.var(x)
        z = self.sampling(mean, log_var)
        
        return mean, log_var, z
        

class Sampling(nn.Module): 
    def forward(self, mean, var): 
        batch, dim = mean.shape
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5*var)*epsilon
        

class Decoder(nn.Module): 
    def __init__(self, latent_dim, org_channels, org_height, org_width):
        super(Decoder, self).__init__()
        
        self.lin = nn.Linear(latent_dim, org_channels*org_height*org_width)
        
        self.reshape = lambda x: x.view(-1, org_channels, org_height, org_width)
         
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid(), # only to use with BCE_loss
        )
        
    
    def forward(self, x): 
        x = self.lin(x)
        x = self.reshape(x)
        x = self.decoder(x)
        return x
        
        


# In[7]:


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction


# In[8]:


def vae_gaussian_kl_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()

def reconstruction_loss(x_reconstructed, x):
    # bce_loss = nn.BCELoss()
    bce_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    return bce_logits_loss(x_reconstructed.float(), x.float())

def vae_loss(y_pred, y_true, beta):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    
    return recon_loss + beta*kld_loss












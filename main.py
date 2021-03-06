import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

epoch = 0
n_epochs = 200
dataset_name = "img_align_celeba"
batch_size = 16
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
n_cpu = 8
img_height = 128
img_width = 128
channels = 3
sample_interval = 400
checkpoint_interval = -1
residual_blocks = 6
selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]

c_dim = len(selected_attrs)
img_shape = (channels, img_height, img_width)

cuda = torch.cuda.is_available()

# Loss functions
criterion_cycle = torch.nn.L1Loss()


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10

# Initialize generator and discriminator
generator = GeneratorResNet(img_shape=img_shape, res_blocks=residual_blocks, c_dim=c_dim)
discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_cycle.cuda()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % epoch))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Configure dataloaders
train_transforms = [
    transforms.Resize(int(1.12 * img_height), Image.BICUBIC),
    transforms.RandomCrop(img_height),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(CelebADataset("../../data/%s" % dataset_name, transforms_=train_transforms, mode="train", attributes=selected_attrs),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

val_transforms = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_dataloader = DataLoader(
    CelebADataset(
        "../../data/%s" % dataset_name, transforms_=val_transforms, mode="val", attributes=selected_attrs
    ),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


label_changes = [
    ((0, 1), (1, 0), (2, 0)),  # Set to black hair
    ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
    ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
    ((3, -1),),  # Flip gender
    ((4, -1),),  # Age flip
]


def sample_images(batches_done):
    """Saves a generated sample of domain translations"""
    val_imgs, val_labels = next(iter(val_dataloader))
    val_imgs = Variable(val_imgs.type(Tensor))
    val_labels = Variable(val_labels.type(Tensor))
    img_samples = None
    for i in range(10):
        img, label = val_imgs[i], val_labels[i]
        # Repeat for number of label changes
        imgs = img.repeat(c_dim, 1, 1, 1)
        labels = label.repeat(c_dim, 1)
        # Make changes to labels
        for sample_i, changes in enumerate(label_changes):
            for col, val in changes:
                labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

        # Generate translations
        gen_imgs = generator(imgs, labels)
        # Concatenate images by width
        gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
        img_sample = torch.cat((img.data, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

    save_image(img_samples.view(1, *img_samples.shape), "images/%s.png" % batches_done, normalize=True)


# ----------
#  Training
# ----------

saved_samples = []
start_time = time.time()
for epoch in range(epoch, n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        # Model inputs
        imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(Tensor))

        # Sample labels as generator inputs
        sampled_c = Variable(Tensor(np.random.randint(0, 2, (imgs.size(0), c_dim))))
        # Generate fake batch of images
        fake_imgs = generator(imgs, sampled_c)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real images
        real_validity, pred_cls = discriminator(imgs)
        # Fake images
        fake_validity, _ = discriminator(fake_imgs.detach())
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data)
        # Adversarial loss
        loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        # Classification loss
        loss_D_cls = criterion_cls(pred_cls, labels)
        # Total loss
        loss_D = loss_D_adv + lambda_cls * loss_D_cls

        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Every n_critic times update generator
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Translate and reconstruct image
            gen_imgs = generator(imgs, sampled_c)
            recov_imgs = generator(gen_imgs, labels)
            # Discriminator evaluates translated image
            fake_validity, pred_cls = discriminator(gen_imgs)
            # Adversarial loss
            loss_G_adv = -torch.mean(fake_validity)
            # Classification loss
            loss_G_cls = criterion_cls(pred_cls, sampled_c)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recov_imgs, imgs)
            # Total loss
            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec

            loss_G.backward()
            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D_adv.item(),
                    loss_D_cls.item(),
                    loss_G.item(),
                    loss_G_adv.item(),
                    loss_G_cls.item(),
                    loss_G_rec.item(),
                    time_left,
                )
            )

            # If at sample interval sample and save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
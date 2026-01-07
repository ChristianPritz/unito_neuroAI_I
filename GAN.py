#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 22:24:58 2026

@author: christian
"""

import tensorflow as tf
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
import numpy as np
# ------------------
# Configuration
# ------------------
DATA_DIR = "/home/christian/Documents/teaching activities/NeuroAI I/code/datasets/faces"
IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 256
EPOCHS = 300


data_dir = pathlib.Path(DATA_DIR)

dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode=None,          # GANs don't need labels
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Normalize images to [-1, 1]
dataset = dataset.map(lambda x: (x / 127.5) - 1.0)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),

        layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, 4, strides=2, padding="same", use_bias=False),
        layers.Activation("tanh")   # Output in [-1, 1]
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(256, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


def sample_face(generator, latent_dim=LATENT_DIM, plot=True):
    """
    Generate a single face image and return the latent vector z.
    
    Returns:
        z        : (latent_dim,) numpy array
        image    : (H, W, 3) numpy array in [0, 1]
    """
    # Sample latent vector
    z = tf.random.normal([1, latent_dim])

    # Generate image
    img = generator(z, training=False)

    # Convert from [-1, 1] → [0, 1]
    img = (img + 1.0) / 2.0

    z_np = z.numpy().squeeze()
    img_np = img.numpy().squeeze()

    if plot:
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np)
        plt.axis("off")
        plt.title("Generated face")
        plt.show()

    return z_np, img_np


# Losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss

def generator_loss(fake_logits):
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)

# Optimizers
gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


generator = build_generator()
discriminator = build_discriminator()


@tf.function
def train_step(real_images):
    noise = tf.random.normal([tf.shape(real_images)[0], LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_logits)
        disc_loss = discriminator_loss(real_logits, fake_logits)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_plot(model, epoch, n=16):
    noise = tf.random.normal([n, LATENT_DIM])
    images = model(noise, training=False)
    images = (images + 1) / 2.0

    plt.figure(figsize=(4, 4))
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.show()


for epoch in range(EPOCHS):
    for real_batch in dataset:
        g_loss, d_loss = train_step(real_batch)

    print(f"Epoch {epoch+1}/{EPOCHS} | G: {g_loss:.3f} | D: {d_loss:.3f}")
    generate_and_plot(generator, epoch + 1)
    
    
z, face = sample_face(generator)

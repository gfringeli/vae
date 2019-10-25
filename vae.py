import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Lambda, Reshape, Flatten, LeakyReLU, Softmax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import cv2
import random
import click


'''
Loss Functions
'''

def kl_divergence_stdnorm(z_mean, z_log_sigma):
    
  kl = -0.5*K.mean(K.sum(1 + 2*z_log_sigma - K.square(z_mean) - K.exp(2*z_log_sigma), axis=1))
  return kl

def reconstruction_ll(x_true, x_prob_logit):
    
  x_true = K.batch_flatten(x_true)
  x_prob_logit = K.batch_flatten(x_prob_logit)
  
  ce = K.binary_crossentropy(x_true, x_prob_logit, from_logits=True)
  ce = K.sum(ce, axis=1)
  ce = K.mean(ce)
  
  return ce

def beta_vae_loss(x_true, z_mean, z_log_sigma, x_rec_logit, beta=4.0):
    
  reconstruction_l = reconstruction_ll(x_true, x_rec_logit)
  kl_divergence = kl_divergence_stdnorm(z_mean, z_log_sigma)
  
  beta_vae_l = reconstruction_l + beta*kl_divergence
  
  return beta_vae_l, reconstruction_l, kl_divergence

def discriminator_loss(p_z, p_z_perm):
    
  loss = 0.5*K.mean(K.binary_crossentropy(1.0, p_z[:,0])) + 0.5*K.mean(K.binary_crossentropy(0.0, p_z_perm[:,0]))
  return loss

def factor_vae_loss(x_true, z_mean, z_log_sigma, x_rec_logit, disc_logits, gamma=35):
    
  beta_vae_l, reconstruction_l, kl_divergence = beta_vae_loss(x_true, z_mean, z_log_sigma, x_rec_logit, beta=1.0)
  
  tc = K.mean(disc_logits[:,0] - disc_logits[:,1])
  
  factor_vae_l = beta_vae_l + gamma*tc
  
  return factor_vae_l, reconstruction_l, kl_divergence, tc


'''
Utility Functions
'''

def norm_sampling(mean, log_stdev):
    
  epsilon = K.random_normal(shape=K.shape(mean))
  
  return mean + K.exp(log_stdev) * epsilon

def permute_dims(z):
    
  shuffled_features = [K.expand_dims(tf.random.shuffle(z[:,i]), axis=-1) for i in range(K.int_shape(z)[1])]    
  shuffled = K.concatenate(shuffled_features, axis=-1)
  
  return shuffled


'''
Models
'''

class Encoder(Model):

  def __init__(self, latent_dims):
    super(Encoder, self).__init__(name='encoder')
    
    self.latent_dims = latent_dims
    
    self.reshape_input = Reshape((64,64,1), input_shape=(64,64))
    self.hidden_layers = [Conv2D(32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same') for _ in range(2)]
    self.hidden_layers += [Conv2D(64, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same') for _ in range(2)]
    self.hidden_layers += [Flatten(), Dense(128)]
            
    self.means = Dense(self.latent_dims)
    self.log_stdevs = Dense(self.latent_dims)
      
  def call(self, x):
      
    x = self.reshape_input(x)
    for h in self.hidden_layers:
      x = h(x)
        
    return self.means(x), self.log_stdevs(x)

class Decoder(Model):
    
  def __init__(self, latent_dims):
    super(Decoder, self).__init__(name='decoder')
    
    self.latent_dims = latent_dims
    
    self.hidden_layers = [
      Dense(128, activation='relu', input_shape=(self.latent_dims,)),
      Dense(4*4*64, activation='relu'),
      Reshape((4, 4, 64)),
      Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same'),
      Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same'),
      Conv2DTranspose(32, kernel_size=(4,4), strides=(2,2), activation='relu', padding='same'),
      Conv2DTranspose(1, kernel_size=(4,4), strides=(2,2), padding='same'),
      Reshape((64, 64))
    ]

  def call(self, x):
      
    for h in self.hidden_layers:
      x = h(x)
        
    return x

class Discriminator(Model):
  
  def __init__(self):
    super(Discriminator, self).__init__(name='discriminator')
    
    self.flatten_input = Flatten()
    self.dense_layers = [Dense(1000, activation=LeakyReLU()) for _ in range(6)]
    
    # logits
    self.logits = Dense(2)
    # 1. from q(z)
    # 2. from q_bar(z)
    
    self.probabilities = Softmax()
      
  def call(self, x):
      
    x = self.flatten_input(x)
    for d in self.dense_layers:
      x = d(x)
    
    logits = self.logits(x)
    
    return logits, self.probabilities(logits)

class BetaVAE():
    
  def __init__(self, latent_dims=10, beta=4.0):
    super(BetaVAE, self).__init__()
    
    self.beta = beta
    self.latent_dims = latent_dims
    
    self.encoder = Encoder(latent_dims)
    self.decoder = Decoder(latent_dims)
    
  def fit(self, X, batch_size=64):
              
    X_train, X_val = train_test_split(X, train_size=0.75, test_size=0.25, random_state=38, shuffle=True)
    
    X_train_ds = tf.data.Dataset.from_tensor_slices(X_train)
    X_train_ds = X_train_ds.shuffle(buffer_size=1024).batch(batch_size)
    
    beta_vae_optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    
    iterations_per_epoch = X_train.shape[0]/batch_size
    iterations = 3e5
    n_epochs = int(iterations/iterations_per_epoch)
    
    for epoch in range(0, n_epochs):
      print("Epoch %d/%d" % (epoch+1, n_epochs))
      
      bar = Progbar(X_train.shape[0], stateful_metrics=["val_loss"], verbose=1)

      last_step = int(np.ceil(X_train.shape[0]/batch_size)) - 1
      for step, x_batch in enumerate(X_train_ds):
        beta_vae_l, rec_l, kl_div = self._train_step(x_batch, beta_vae_optimizer)
        
        if step == last_step:
          val_beta_vae_l, val_rec_l, val_kl_div = self._evaluate(X_val)
          bar.add(x_batch.shape[0],
                  values=[
                    ("loss", beta_vae_l),
                    ("rec_loss", rec_l),
                    ("kl_div", kl_div),
                    ("val_loss", val_beta_vae_l),
                    ("val_rec_loss", val_rec_l),
                    ("val_kl_div", val_kl_div),
                  ])
            
        else:
          bar.add(x_batch.shape[0],
                  values=[
                    ("loss", beta_vae_l),
                    ("rec_loss", rec_l),
                    ("kl_div", kl_div),
                  ])
              
  @tf.function
  def _train_step(self, X, optimizer):
            
    with tf.GradientTape() as tape:

      z_mean, z_stdev = self.encoder(X)
      samples = norm_sampling(z_mean, z_stdev)
      z_rec = self.decoder(samples)
      
      beta_vae_l, rec_l, kl_div = beta_vae_loss(X, z_mean, z_stdev, z_rec, self.beta)

    # optimize VAE
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    grad = tape.gradient(beta_vae_l, variables)
    
    optimizer.apply_gradients(zip(grad, variables))
    
    return beta_vae_l, rec_l, kl_div
  
  @tf.function
  def _evaluate(self, X):
      
    z_mean, z_stdev = self.encoder(X)
    samples = norm_sampling(z_mean, z_stdev)
    z_rec = self.decoder(samples)
    
    beta_vae_l, rec_l, kl_div = beta_vae_loss(X, z_mean, z_stdev, z_rec, self.beta)
    
    return beta_vae_l, rec_l, kl_div

class FactorVAE():
  
  def __init__(self, latent_dims=10, gamma=35.0):
    super(FactorVAE, self).__init__(name='factor_vae')
    
    self.gamma = gamma
    self.latent_dims = latent_dims
    
    self.encoder = Encoder(latent_dims)
    self.decoder = Decoder(latent_dims)
    self.discriminator = Discriminator()
  
  def fit(self, X, batch_size=64):
              
    X_train, X_val = train_test_split(X, train_size=0.75, test_size=0.25, random_state=38, shuffle=True)
    
    X_train_ds = tf.data.Dataset.from_tensor_slices(X_train)
    X_train_ds = X_train_ds.shuffle(buffer_size=1024).batch(batch_size)
    
    vae_optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    discriminator_optimizer = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)
    
    iterations_per_epoch = X_train.shape[0]/128
    iterations = 3e5
    n_epochs = int(iterations/iterations_per_epoch)
    
    for epoch in range(0, n_epochs):
      print("Epoch %d/%d" % (epoch+1, n_epochs))
      
      bar = Progbar(X_train.shape[0], stateful_metrics=["val_loss"], verbose=1)

      last_step = int(np.ceil(X_train.shape[0]/batch_size)) - 1
      for step, x_batch in enumerate(X_train_ds):
        fvae_l, rec_l, disc_l = self._train_step(x_batch, vae_optimizer, discriminator_optimizer)
        
        if step == last_step:
          val_fvae_l, val_rec_l = self._evaluate(X_val)
          bar.add(x_batch.shape[0],
                  values=[
                    ("loss", fvae_l),
                    ("rec_loss", rec_l),
                    ("disc_loss", disc_l),
                    ("val_loss", val_fvae_l),
                    ("val_rec_loss", val_rec_l)])
          
        else:
          bar.add(x_batch.shape[0],
                  values=[
                    ("loss", fvae_l),
                    ("rec_loss", rec_l),
                    ("disc_loss", disc_l)
                  ])
            
  @tf.function
  def _train_step(self, X, vae_optimizer, discriminator_optimizer):
      
    batch_size = int(K.int_shape(X)[0]/2)
    
    X = tf.random.shuffle(X)
    
    X_1 = X[:batch_size]
    X_2 = X[batch_size:]
    
    # optimize VAE
    with tf.GradientTape() as vae_tape, tf.GradientTape() as disc_tape:

      # VAE
      z_mean, z_stdev = self.encoder(X_1)
      samples1 = norm_sampling(z_mean, z_stdev)
      z_rec = self.decoder(samples1)
      
      disc_logits, disc_probs = self.discriminator(samples1)
      
      fvae_l, rec_l, _, _ = factor_vae_loss(X_1, z_mean, z_stdev, z_rec, disc_logits, self.gamma)
      
      # discriminator
      z_mean, z_stdev = self.encoder(X_2)
      samples2 = norm_sampling(z_mean, z_stdev)
      samples_perm = K.stop_gradient(permute_dims(samples2))
  
      _, disc_probs_perm = self.discriminator(samples_perm)
      
      disc_l = discriminator_loss(disc_probs, disc_probs_perm)
    
    # optimize VAE
    vae_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
    vae_grad = vae_tape.gradient(fvae_l, vae_vars)
    
    vae_optimizer.apply_gradients(zip(vae_grad, vae_vars))
    
    # optimize discriminator
    discriminator_vars = self.discriminator.trainable_variables
    discriminator_grad = disc_tape.gradient(disc_l, discriminator_vars)
    
    discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator_vars))
    
    return fvae_l, rec_l, disc_l
  
  @tf.function
  def _evaluate(self, X):
      
    z_mean, z_stdev = self.encoder(X)
    samples = norm_sampling(z_mean, z_stdev)
    z_rec = self.decoder(samples)
    
    disc_logits, _ = self.discriminator(samples)

    fvae_l, rec_l, _, _ = factor_vae_loss(X, z_mean, z_stdev, z_rec, disc_logits, self.gamma)
    
    return fvae_l, rec_l


'''
Command Line Application
'''

@click.group()
def main():
  """
  Variational Auto-Encoder (including BetaVAE and FactorVAE)
  """
  pass

@main.command()
@click.option('--beta', '-b', default=1.0, help='β as defined in "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. Higgins et al. (2016)"')
@click.option('--gamma', '-g', default=0.0, help='γ as defined in "Disentangling by Factorising. Kim, Mnih. (2018)"')
@click.argument('output', type=click.Path(exists=True), nargs=1)
def train(beta, gamma, output):
  """
  Train a Variational Auto-Encoder or one of its variants β-VAE and FactorVAE. By default β=1 and γ=0, which corresponds to the standard VAE.
  Setting β to a different value will train a β-VAE. Setting γ to a different value will train a FactorVAE.
  Setting both β and γ to a different value is currently not supported.
  """

  data = np.load('./data/2d_sprites.npz', allow_pickle=True, encoding='latin1')
  imgs = data['imgs'].astype('float32')

  if beta==1.0 and gamma==0.0:
    click.echo('Train VAE')
    model = BetaVAE(latent_dims=10, beta=1.0)

  elif beta!=1.0 and gamma==0.0:
    click.echo('Train β-VAE with β=%f' % beta)
    model = BetaVAE(latent_dims=10, beta=beta)

  elif beta==1.0 and gamma!=0.0:
    click.echo('Train FactorVAE with γ=%f' % gamma)
    model = FactorVAE(latent_dims=10, gamma=gamma)

  else:
    click.echo('You are not allowed to set both β and γ to custom values.')
    return

  model.fit(imgs, batch_size=64)

  # store encoder and decoder weights
  if output[-1] != '/':
    output += '/'
  
  model.encoder.save_weights(output + 'encoder.h5')

  # decoder
  model.decoder.save_weights(output + 'decoder.h5')

if __name__ == '__main__':
  main()
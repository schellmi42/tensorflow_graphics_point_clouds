# noqa: E501
import tensorflow as tf
import MCCNN2.pc as pc
import MCCNN2.io as io
import numpy as np
import tensorflow_graphics
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)
tf.random.set_seed(42)

quick_test = False

data_dir = '../2019 - ModelNet_PointNet/'
num_classes = 40  # modelnet 10 or 40
points_per_file = 1024  # number of points loaded per model
samples_per_model = 1024  # number of input points per file

labels = []
with open(data_dir + f'modelnet{num_classes}_shape_names.txt') as inFile:
  for line in inFile:
    labels.append(line.replace('\n', ''))

train_set = []
train_labels = []
with open(data_dir + f'modelnet{num_classes}_train.txt') as inFile:
  for line in inFile:
    line = line.replace('\n', '')
    category = line[:-5]
    train_set. append(data_dir + category + '/' + line + '.txt')
    if category not in labels:
      raise ValueError('Unknown category ' + category)
    train_labels.append(labels.index(category))

test_set = []
test_labels = []
with open(data_dir + f'modelnet{num_classes}_test.txt') as inFile:
  for line in inFile:
    line = line.replace('\n', '')
    category = line[:-5]
    test_set. append(data_dir + category + '/' + line + '.txt')
    if category not in labels:
      raise ValueError('Unknown category ' + category)
    test_labels.append(labels.index(category))

num_classes = len(labels)

train_data_points = np.empty([len(train_set), points_per_file, 3])
train_data_features = np.empty([len(train_set), points_per_file, 3])
# ids = np.arange(0, 10000)
print(f'### loading modelnet{num_classes} train ###')
for i, filename in enumerate(train_set):
  points, features = \
      io.load_points_from_file_to_numpy(filename,
                                        max_num_points=points_per_file)
  # selection = np.random.choice(ids, 1024)
  points = points  # [selection]
  features = features  # [selection]
  train_data_points[i] = points
  train_data_features[i] = features
  if i % 100 == 0:
    print(f'{i}/{len(train_set)}')
  if quick_test and i > 100:
    break

test_data_points = np.empty([len(test_set), points_per_file, 3])
test_data_features = np.empty([len(test_set), points_per_file, 3])

print(f'### loading modelnet{num_classes} test ###')
for i, filename in enumerate(test_set):
  points, features = \
      io.load_points_from_file_to_numpy(filename,
                                        max_num_points=points_per_file)
  # selection = np.random.choice(ids, 1024)
  points = points  # [selection]
  features = features  # [selection]
  test_data_points[i] = points
  test_data_features[i] = features
  if i % 100 == 0:
    print(f'{i}/{len(test_set)}')
  if quick_test and i > 100:
    break


#-----------------------------------------------

def identity_layer(x, *args, **kwargs):
  return x


class conv_block():

  def __init__(self,
               num_features_in,
               num_features_out,
               hidden_size,
               layer_type,
               strided=False):

    self.res_layers = []
    self.skip_layers = []
    self.BN_layers = []
    self.activation_layers = []

    # -- residual layers --
    residual_feature_size = num_features_out // 2

    self.res_layers.append(pc.layers.Conv1x1(
        num_features_in=num_features_in,
        num_features_out=residual_feature_size))
    self.BN_layers.append(tf.keras.layers.BatchNormalization())
    self.activation_layers.append(tf.keras.layers.LeakyReLU())

    if layer_type == 'MCConv':
      self.res_layers.append(pc.layers.MCConv(
          num_features_in=residual_feature_size,
          num_features_out=residual_feature_size,
          num_dims=3,
          num_mlps=4,
          mlp_size=[8]))
    elif layer_type == 'PointConv':
      self.res_layers.append(pc.layers.PointConv(
          num_features_in=residual_feature_size,
          num_features_out=residual_feature_size,
          num_dims=3,
          size_hidden=8))
    elif layer_type == 'KPConv':
      self.res_layers.append(pc.layers.KPConv(
          num_features_in=residual_feature_size,
          num_features_out=residual_feature_size,
          num_dims=3,
          num_kernel_points=15))
    else:
      raise ValueError("Unknown layer type!")

    self.BN_layers.append(tf.keras.layers.BatchNormalization())
    self.activation_layers.append(tf.keras.layers.LeakyReLU())
    self.res_layers.append(pc.layers.Conv1x1(
        num_features_in=residual_feature_size,
        num_features_out=num_features_out))
    self.BN_layers.append(tf.keras.layers.BatchNormalization())

    # -- skip layers --
    if strided:
      self.skip_layers.append(pc.layers.MaxPooling())
    else:
      self.skip_layers.append(identity_layer)

    self.skip_layers.append(pc.layers.Conv1x1(
        num_features_in=num_features_in,
        num_features_out=num_features_out))
    self.BN_layers.append(tf.keras.layers.BatchNormalization())

    # --- union --
    self.activation_layers.append(tf.keras.layers.LeakyReLU())

  def __call__(self,
               features,
               point_cloud_in,
               point_cloud_out,
               conv_radius,
               pool_radius=None,
               training=False):
    # -- residual branch --
    # conv1x1, downsampling in feature dimension
    res = self.res_layers[0](features, point_cloud_in)
    # BN + lReLU
    res = self.BN_layers[0](res, training=training)
    res = self.activation_layers[0](res)
    # spatial convolution
    res = self.res_layers[1](res, point_cloud_in, point_cloud_out, conv_radius)
    # BN + lReLU
    res = self.BN_layers[1](res, training=training)
    res = self.activation_layers[1](res)
    # conv 1x1, upsampling in feature dimension
    res = self.res_layers[2](res, point_cloud_out)
    # BN
    res = self.BN_layers[2](res, training=training)
    # -- skip connection --
    # spatial maxpooling
    skip = self.skip_layers[0](features, point_cloud_in, point_cloud_out,
                               pool_radius)
    # conv1x1, upsampling in feature dimension
    skip = self.skip_layers[1](skip, point_cloud_out)
    # BN
    skip = self.BN_layers[3](skip, training=training)

    # --- Add + lReLU--
    return self.activation_layers[2](res + skip)


class mymodel(tf.keras.Model):

  def __init__(self,
               feature_sizes,
               hidden_size,
               pool_radii,
               conv_radii,
               layer_type='MCConv'):
    super(mymodel, self).__init__()
    self.num_levels = len(pool_radii)
    self.pool_radii = pool_radii.reshape(-1, 1)
    self.conv_radii = conv_radii
    self.strided_conv_blocks = []
    self.conv_blocks = []
    self.batch_layers = []
    self.dense_layers = []
    self.activations = []
    # -- encoder network
    for i in range(self.num_levels):
      self.strided_conv_blocks.append(conv_block(feature_sizes[i + 1],
                                                 feature_sizes[i + 1],
                                                 hidden_size,
                                                 layer_type,
                                                 strided=True))
      self.conv_blocks.append(conv_block(feature_sizes[i + 1],
                                         feature_sizes[i + 1],
                                         hidden_size,
                                         layer_type,
                                         strided=False))
    self.global_pooling = pc.layers.GlobalAveragePooling()
    # -- classification head ---
    self.batch_layers.append(tf.keras.layers.BatchNormalization())
    self.activations.append(tf.keras.layers.LeakyReLU())
    self.dense_layers.append(tf.keras.layers.Dense(feature_sizes[-2]))
    self.batch_layers.append(tf.keras.layers.BatchNormalization())
    self.activations.append(tf.keras.layers.LeakyReLU())
    self.dense_layers.append(tf.keras.layers.Dense(feature_sizes[-1]))

  def __call__(self,
               points,
               sizes,
               features,
               training):
    point_cloud = pc.PointCloud(points, sizes)
    pool_radii = self.pool_radii
    conv_radii = self.conv_radii
    point_hierarchy = pc.PointHierarchy(point_cloud, pool_radii, 'poisson')
    for i in range(self.num_levels):
      features = self.strided_conv_blocks[i](features,
                                             point_hierarchy[i],
                                             point_hierarchy[i + 1],
                                             self.conv_radii[i],
                                             self.pool_radii[i],
                                             training=training)
      features = self.conv_blocks[i](features,
                                     point_hierarchy[i + 1],
                                     point_hierarchy[i + 1],
                                     conv_radii[i],
                                     training=training)

    features = self.global_pooling(features, point_hierarchy[-1])
    features = self.batch_layers[-2](features, training)
    features = self.activations[-2](features)
    features = self.dense_layers[-2](features)
    features = self.batch_layers[-1](features, training)
    features = self.activations[-1](features)
    return self.dense_layers[-1](features)


#-----------------------------------------------
class modelnet_data_generator(tf.keras.utils.Sequence):

    def __init__(self,
                 points,
                 labels,
                 batch_size):
        self.points = points
        self.labels = np.array(labels, dtype=int)
        self.batch_size = batch_size
        self.epoch_size = len(self.points)

        self.ids = np.arange(0, points_per_file)
        # shuffle data before training
        self.on_epoch_end()

    def __len__(self):
      # number of batches per epoch
      return(int(np.floor(self.epoch_size / self.batch_size)))

    def __call__(self):
      out = self.__getitem__(self.index)
      self.index += 1
      return out

    def __getitem__(self, index):
      labels = \
          self.labels[index * self.batch_size:(index + 1) * self.batch_size]
      points = \
          self.points[index * self.batch_size:(index + 1) * self.batch_size]
      features = tf.ones([self.batch_size, samples_per_model, 1])
      sizes = tf.ones([self.batch_size], dtype=tf.int32) * samples_per_model

      # sample points
      sampled_points = np.empty([self.batch_size, samples_per_model, 3])
      for batch in range(self.batch_size):
        selection = np.random.choice(self.ids, samples_per_model)
        sampled_points[batch] = points[batch][selection]
      return sampled_points, features, sizes, labels

    def on_epoch_end(self):
      shuffle = np.random.permutation(np.arange(0, len(self.points)))
      self.points = self.points[shuffle]
      self.labels = self.labels[shuffle]
      self.index = 0

    # def generator_function(self):
    #   while True:
    #     self.on_epoch_end()
    #     for i in range(len(self)):
    #       yield self[i]

#-----------------------------------------------

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

batch_size = 16
num_epochs = 100
if quick_test:
  num_epochs = 2

hidden_size = 16
feature_sizes = [1, 128, 256, 512, 128, num_classes]
pool_radii = np.array([0.1, 0.2, 0.4])
conv_radii = pool_radii * 1.5


gen_train = modelnet_data_generator(train_data_points, train_labels,
                                    batch_size)
gen_test = modelnet_data_generator(test_data_points, test_labels, batch_size)

lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=len(gen_train),
    decay_rate=0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)


# --- Training Loop---
def training(model,
             epoch_print=1):
  train_loss_results = []
  train_accuracy_results = []
  test_loss_results = []
  test_accuracy_results = []

  for epoch in range(1):
    time_epoch_start = time.time()
    # --- Training ---
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for points, features, sizes, labels in gen_train:
      with tf.GradientTape() as tape:
        logits = model(points, sizes, features, training=True)
        pred = tf.nn.softmax(logits, axis=-1)
        loss = loss_function(y_true=labels, y_pred=pred)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      epoch_loss_avg.update_state(loss)
      epoch_accuracy.update_state(labels, pred)

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # --- Validation ---
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for points, features, sizes, labels in gen_test:
      logits = model(points, sizes, features, training=False)
      pred = tf.nn.softmax(logits, axis=-1)
      loss = loss_function(y_true=labels, y_pred=pred)
      epoch_loss_avg.update_state(loss)
      epoch_accuracy.update_state(labels, pred)
    test_loss_results.append(epoch_loss_avg.result())
    test_accuracy_results.append(epoch_accuracy.result())
    time_epoch_end = time.time()

    if epoch % epoch_print == 0:
      # End epoch
      print('Epoch {:03d} Time: {:.3f}s'.format(
          epoch,
          time_epoch_end - time_epoch_start))
      print('Training:  Loss: {:.3f}, Accuracy: {:.3%}'.format(
          train_loss_results[-1],
          train_accuracy_results[-1]))
      print('Validation: Loss: {:.3f}, Accuracy: {:.3%}'.format(
          test_loss_results[-1],
          test_accuracy_results[-1]))

model_MC = mymodel(feature_sizes, hidden_size, pool_radii, conv_radii,
                   layer_type='MCConv')
training(model_MC)
model_KP = mymodel(feature_sizes, hidden_size, pool_radii, conv_radii,
                   layer_type='KPConv')
training(model_KP)
model_PC = mymodel(feature_sizes, hidden_size, pool_radii, conv_radii,
                   layer_type='PointConv')
training(model_PC)

# noqa: E402
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0])
except ValueError:
  print('Invalid device or cannot modify virtual devices once initialized.')
  pass
except IndexError:
  print('No GPU found')
  pass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pylib.pc as pc
from pylib.pc import layers
import pylib.io as io
import numpy as np
import tensorflow_graphics
import time
import h5py

# for graph mode debugging
#tf.config.run_functions_eagerly(True)

# np.random.seed(42)
# tf.random.set_seed(42)

quick_test = False

# -- loading data ---

data_dir = './2019 - ModelNet_PointNet/'
hdf5_tmp_dir = "./tmp_modelnet"
num_classes = 40  # modelnet 10 or 40
points_per_file = 1024  # number of points loaded per model
samples_per_model = 1024  # number of input points per file
batch_size = 16

category_names = []
with open(data_dir + f'modelnet{num_classes}_shape_names.txt') as inFile:
  for line in inFile:
    category_names.append(line.replace('\n', ''))

train_set = []
train_labels = []
with open(data_dir + f'modelnet{num_classes}_train.txt') as inFile:
  for line in inFile:
    line = line.replace('\n', '')
    category = line[:-5]
    train_set.append(data_dir + category + '/' + line + '.txt')
    if category not in category_names:
      raise ValueError('Unknown category ' + category)
    train_labels.append(category_names.index(category))

test_set = []
test_labels = []
with open(data_dir + f'modelnet{num_classes}_test.txt') as inFile:
  for line in inFile:
    line = line.replace('\n', '')
    category = line[:-5]
    test_set.append(data_dir + category + '/' + line + '.txt')
    if category not in category_names:
      raise ValueError('Unknown category ' + category)
    test_labels.append(category_names.index(category))

num_classes = len(category_names)
print(f'### loading modelnet{num_classes} train ###')
if os.path.exists(hdf5_tmp_dir + "/train_data_small.hdf5"):
  h5File = h5py.File(hdf5_tmp_dir + "/train_data_small.hdf5", "r")
  train_data_points = h5File["train_data"][()]
  train_data_points = train_data_points[:, 0:points_per_file, :]
  h5File.close()
else:
  train_data_points = np.empty([len(train_set), points_per_file, 3])

  for i, filename in enumerate(train_set):
    points, _ = \
        io.load_points_from_file_to_numpy(filename,
                                          max_num_points=points_per_file)
    points = points
    train_data_points[i] = points
    if i % 500 == 0:
      print(f'{i}/{len(train_set)}')
    if quick_test and i > 100:
      break

  h5File = h5py.File(hdf5_tmp_dir + "/train_data_small.hdf5", "w")
  h5File.create_dataset("train_data", data=train_data_points)
  h5File.close()

print(f'### loading modelnet{num_classes} test ###')
if os.path.exists(hdf5_tmp_dir + "/test_data_small.hdf5"):
  h5File = h5py.File(hdf5_tmp_dir + "/test_data_small.hdf5", "r")
  test_data_points = h5File["test_data"][()]
  test_data_points = test_data_points[:, 0:points_per_file, :]
  h5File.close()
else:
  test_data_points = np.empty([len(test_set), points_per_file, 3])
  for i, filename in enumerate(test_set):
    points, _ = \
        io.load_points_from_file_to_numpy(filename,
                                          max_num_points=points_per_file)
    points = points
    test_data_points[i] = points
    if i % 500 == 0:
      print(f'{i}/{len(test_set)}')
    if quick_test and i > 100:
      break

  h5File = h5py.File(hdf5_tmp_dir + "/test_data_small.hdf5", "w")
  h5File.create_dataset("test_data", data=test_data_points)
  h5File.close()


#-----------------------------------------------
class mymodel(tf.keras.Model):
  ''' Model architecture with `L` convolutional layers followed by
  two dense layers.

  Args:
    features_sizes: A `list` of `ints`, the feature dimensions. Shape `[L+3]`.
    sample_radii: A `list` of `floats, the radii used for sampling
      of the point clouds. Shape `[L]`.
    conv_radii: A `list` of `floats`, the radii used by the convolution
      layers. Shape `[L]`.
    layer_type: A `string`, the type of convolution used,
      can be 'MCConv', 'KPConv', 'PointConv'.
    sampling_method: 'poisson disk' or 'cell average'.
  '''

  def __init__(self,
               feature_sizes,
               sample_radii,
               conv_radii,
               layer_type='MCConv',
               sampling_method='poisson disk',
               dropout_rate=0.5):

    super().__init__(name=None)

    self.num_layers = len(sample_radii)
    self.sample_radii = sample_radii.reshape(-1, 1)
    self.conv_radii = conv_radii
    self.sampling_method = sampling_method
    self.conv_layers = []
    self.batch_layers = []
    self.dense_layers = []
    self.activations = []
    self.dropouts = []
    # encoder
    for i in range(self.num_layers):
      # convolutional downsampling layers
      if layer_type == 'MCConv':
        self.conv_layers.append(layers.MCConv(
            num_features_in=feature_sizes[i],
            num_features_out=feature_sizes[i + 1],
            num_dims=3,
            num_mlps=4,
            mlp_size=[8]))
      elif layer_type == 'PointConv':
        self.conv_layers.append(layers.PointConv(
            num_features_in=feature_sizes[i],
            num_features_out=feature_sizes[i + 1],
            num_dims=3,
            size_hidden=32))
      elif layer_type == 'KPConv':
        self.conv_layers.append(layers.KPConv(
            num_features_in=feature_sizes[i],
            num_features_out=feature_sizes[i + 1],
            num_dims=3,
            num_kernel_points=15))
      else:
        raise ValueError("Unknown layer type!")
      if i < self.num_layers - 1:
        # batch normalization and activation function
        self.batch_layers.append(tf.keras.layers.BatchNormalization())
        self.activations.append(tf.keras.layers.LeakyReLU())
        self.conv_layers.append(layers.Conv1x1(
            num_features_in=feature_sizes[i + 1],
            num_features_out=feature_sizes[i + 1]))
        self.batch_layers.append(tf.keras.layers.BatchNormalization())
        self.activations.append(tf.keras.layers.LeakyReLU())
    # global pooling
    # self.global_pooling = layers.GlobalAveragePooling()
    self.batch_layers.append(tf.keras.layers.BatchNormalization())
    self.activations.append(tf.keras.layers.LeakyReLU())
    # MLP
    self.dropouts.append(tf.keras.layers.Dropout(dropout_rate))
    self.dense_layers.append(tf.keras.layers.Dense(feature_sizes[-2]))
    self.batch_layers.append(tf.keras.layers.BatchNormalization())
    self.activations.append(tf.keras.layers.LeakyReLU())
    self.dropouts.append(tf.keras.layers.Dropout(dropout_rate))
    self.dense_layers.append(tf.keras.layers.Dense(feature_sizes[-1]))

  @tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool)]
        )
  def __call__(self,
               points,
               features,
               sizes,
               training):
    ''' Evaluates network.

    Args:
      points: The point coordinates. Shape `[B, N, 3]`.
      features: Input features. Shape `[B, N, C]`.
      training: A `bool`, passed to the batch norm layers.

    Returns:
      The logits per class.
    '''
    sample_radii = self.sample_radii
    conv_radii = self.conv_radii
    sampling_method = self.sampling_method
    # input point cloud
    # Note: Here all point clouds have the same number of points, so no `sizes`
    #       or `batch_ids` are passed.
    point_cloud = pc.PointCloud(points, sizes=sizes, batch_size=batch_size)
    # spatial downsampling
    point_hierarchy = pc.PointHierarchy(point_cloud,
                                        sample_radii,
                                        sampling_method)
    # network evaluation
    for i in range(self.num_layers):
      if i < self.num_layers - 1:
        features = self.conv_layers[i * 2](
            features,
            point_hierarchy[i],
            point_hierarchy[i + 1],
            conv_radii[i])
        features = self.batch_layers[i * 2](features, training=training)
        features = self.activations[i * 2](features)
        features = self.conv_layers[i * 2 + 1](
            features,
            point_hierarchy[i + 1])
        features = self.batch_layers[i * 2 + 1](features, training=training)
        features = self.activations[i * 2 + 1](features)
      else:
        features = self.conv_layers[i * 2](
            features,
            point_hierarchy[i],
            point_hierarchy[i + 1],
            conv_radii[i],
            return_sorted=True)
    # classification head
    features = self.batch_layers[-2](features, training=training)
    features = self.activations[-2](features)
    features = self.dropouts[-2](features, training=training)
    features = self.dense_layers[-2](features)
    features = self.batch_layers[-1](features, training=training)
    features = self.activations[-1](features)
    features = self.dropouts[-1](features, training=training)
    return self.dense_layers[-1](features)


#-----------------------------------------------
class modelnet_data_generator(tf.keras.utils.Sequence):
  ''' Small generator of batched data.
  '''
  def __init__(self,
               points,
               labels,
               batch_size,
               augment):
      self.points = points
      self.labels = np.array(labels, dtype=int)
      self.batch_size = batch_size
      self.epoch_size = len(self.points)
      self.sizes = np.ones([batch_size]) * samples_per_model

      self.augment = augment
      # shuffle data before training
      self.on_epoch_end()

  def __len__(self):
    # number of batches per epoch
    return(int(np.floor(self.epoch_size / self.batch_size)))

  def __call__(self):
    ''' Loads batch and increases batch index.
    '''
    data = self.__getitem__(self.index)
    self.index += 1
    return data

  def __getitem__(self, index, samples_per_model=1024):
    ''' Loads data of current batch and samples random subset of the points.
    '''
    # constant input feature
    features = tf.ones([self.batch_size, samples_per_model, 1])

    # sample points
    self_indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
    sampled_points = np.empty([self.batch_size, samples_per_model, 3])
    out_labels = np.empty([self.batch_size])
    for batch in range(self.batch_size):

      sampled_points[batch] = self.points[self_indices[batch]][0:samples_per_model]
      out_labels[batch] = self.labels[self_indices[batch]]

      if self.augment:
        # Data augmentation - Anisotropic scale.
        cur_scaling = np.random.uniform(size=(1, 3)) * 0.2 + 0.9
        sampled_points[batch] = sampled_points[batch] * cur_scaling

    return sampled_points, features, out_labels

  def on_epoch_end(self):
    ''' Shuffles data and resets batch index.
    '''
    self.order = np.random.permutation(np.arange(0, len(self.points)))
    self.index = 0

#-----------------------------------------------

num_epochs = 400
if quick_test:
  num_epochs = 2

feature_sizes = [1, 128, 256, 512, 1024, 1024, num_classes]
sample_radii = np.array([0.1, 0.2, 0.4, np.sqrt(3)  * 2])
conv_radii = sample_radii

# initialize data generators
gen_train = modelnet_data_generator(
    train_data_points, train_labels, batch_size, augment=True)
gen_test = modelnet_data_generator(
    test_data_points, test_labels, batch_size, augment=False)

# loss function and optimizer
lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=20 * len(gen_train),  # every 20th epoch
    decay_rate=0.7,
    staircase=True)
#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_decay)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()


# --- Training Loop---
def training(model,
             optimizer,
             loss_function,
             num_epochs=400,
             epochs_print=1,
             epoch_save=1
             ):
  train_loss_results = []
  train_accuracy_results = []
  test_loss_results = []
  test_accuracy_results = []
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  for epoch in range(num_epochs):
    time_epoch_start = time.time()

    # --- Training ---
    iter_batch = 0
    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()
    for points, features, labels in gen_train:
      # evaluate model; forward pass
      with tf.GradientTape() as tape:
        logits = model(points, features, sizes=gen_train.sizes, training=True)
        pred = tf.nn.softmax(logits, axis=-1)
        loss = loss_function(y_true=labels, y_pred=pred)
      # backpropagation
      grads = tape.gradient(loss, model.trainable_variables)

      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      epoch_loss_avg.update_state(loss)
      epoch_accuracy.update_state(labels, pred)

      if iter_batch % 10 == 0:

        print(
            "\r {:03d} / {:03d} Loss: {:.3f}, Accuracy: {:.3%}      ".format(
                iter_batch, len(gen_train),
                epoch_loss_avg.result(),
                epoch_accuracy.result()),
            end="")

      iter_batch += 1

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # --- Validation ---
    epoch_loss_avg.reset_states()
    epoch_accuracy.reset_states()

    for points, features, labels in gen_test:
      # evaluate model; forward pass
      logits = model(points, features, gen_test.sizes, training=False)
      pred = tf.nn.softmax(logits, axis=-1)
      loss = loss_function(y_true=labels, y_pred=pred)

      epoch_loss_avg.update_state(loss)
      epoch_accuracy.update_state(labels, pred)

    test_loss_results.append(epoch_loss_avg.result())
    test_accuracy_results.append(epoch_accuracy.result())

    time_epoch_end = time.time()

    if epoch % epochs_print == 0:
      # End epoch
      print('Epoch {:03d} Time: {:.3f}s'.format(
          epoch,
          time_epoch_end - time_epoch_start))
      print('Training:   Loss: {:.3f}, Accuracy: {:.3%}'.format(
          train_loss_results[-1],
          train_accuracy_results[-1]))
      print('Validation: Loss: {:.3f}, Accuracy: {:.3%}'.format(
          test_loss_results[-1],
          test_accuracy_results[-1]))
    if epoch % epoch_save == 0:
      model.save_weights('saved_models/' + str(epoch))

# ----------------------------

dropout_rate = 0.5

model_MC = mymodel(feature_sizes, sample_radii, conv_radii,
                   layer_type='MCConv', dropout_rate=dropout_rate)                   

# load previously saved model weights
#model_MC.load_weights('saved_models/5')

training(model_MC,
         optimizer=optimizer,
         loss_function=loss_function,
         num_epochs=num_epochs)
"""
model_KP = mymodel(feature_sizes, pool_radii, conv_radii,
                   layer_type='KPConv', dropout_rate=dropout_rate)
training(model_KP, num_epochs)
model_PC = mymodel(feature_sizes, pool_radii, conv_radii,
                   layer_type='PointConv', dropout_rate=dropout_rate)
training(model_PC, num_epochs)
"""

import tensorflow as tf 
import MCCNN2.pc as pc
import MCCNN2.io as io
import numpy as np
import tensorflow_graphics
import os, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(42)
tf.random.set_seed(42)

quick_test=True

data_dir = '../2019 - ModelNet_PointNet/'
num_classes = 40 # modelnet 10 or 40

labels = []
with open(data_dir + 'modelnet%s_shape_names.txt'%num_classes) as inFile:
  for line in inFile:
    labels.append(line.replace('\n',''))

train_set = []
train_labels = []
with open(data_dir + 'modelnet%s_train.txt'%num_classes) as inFile:
  for line in inFile:
    line = line.replace('\n','')
    category = line[:-5]
    train_set. append(data_dir + category +'/' + line + '.txt')
    if not category in labels:
      raise ValueError('Unknown category %s'%category)
    train_labels.append(labels.index(category))

test_set = []
test_labels = []
with open(data_dir + 'modelnet%s_test.txt'%num_classes) as inFile:
  for line in inFile:
    line = line.replace('\n','')
    category = line[:-5]
    test_set. append(data_dir + category +'/' + line + '.txt')
    if not category in labels:
      raise ValueError('Unknown category %s'%category)
    test_labels.append(labels.index(category))

num_classes = len(labels)

train_data_points = np.empty([len(train_set),1024,3])
train_data_features = np.empty([len(train_set),1024,3])
ids = np.arange(0,10000)
print('### loading modelnet%s train###'%num_classes)
for i,filename in enumerate(train_set):
  points, features = io.load_points_from_file_to_numpy(filename)
  selection = np.random.choice(ids,1024)
  points = points[selection]
  features = features[selection]
  train_data_points[i]=points
  train_data_features[i] = features
  if i%100 == 0:
    print(f'{i}/{len(train_set)}')
  if quick_test and i>100:
    break

test_data_points = np.empty([len(test_set),1024,3])
test_data_features = np.empty([len(test_set),1024,3])

print('### loading modelnet%s test###'%num_classes)
for i,filename in enumerate(test_set):
  points, features = io.load_points_from_file_to_numpy(filename)
  selection = np.random.choice(ids,1024)
  points = points[selection]
  features = features[selection]
  test_data_points[i]=points
  test_data_features[i] = features
  if i%100 == 0:
    print(f'{i}/{len(test_set)}')
  if quick_test and i>100: 
    break
#-----------------------------------------------

class mymodel(tf.keras.Model):

  def __init__(self, feature_sizes, hidden_size, pool_radii, conv_radii):
    super(mymodel, self).__init__()
    self.num_layers = len(pool_radii)
    self.pool_radii = pool_radii.reshape(-1,1)
    self.conv_radii = conv_radii
    self.conv_layers = []
    self.batch_layers = []
    self.dense_layers = []
    self.activations = []
    for i in range(self.num_layers):
      self.conv_layers.append(pc.layers.MCConv(feature_sizes[i],feature_sizes[i+1],hidden_size, 3))
      self.batch_layers.append(tf.keras.layers.BatchNormalization())
      self.activations.append(tf.keras.layers.LeakyReLU())
    self.dense_layers.append(tf.keras.layers.Dense(feature_sizes[-2]))
    self.batch_layers.append(tf.keras.layers.BatchNormalization())
    self.activations.append(tf.keras.layers.LeakyReLU())
    self.dense_layers.append(tf.keras.layers.Dense(feature_sizes[-1]))

  def __call__(self, points, sizes,  features, training):
    point_cloud = pc.PointCloud(points, sizes)
    pool_radii = self.pool_radii
    conv_radii = self.conv_radii
    point_hierarchy = pc.PointHierarchy(point_cloud, pool_radii)
    for i in range(self.num_layers):
      if i == self.num_layers-1:
        return_sorted=True
      else:
        return_sorted=False
      features = self.conv_layers[i](features, point_hierarchy[i], point_hierarchy[i+1],conv_radii[i], return_sorted=return_sorted)
      features = self.batch_layers[i](features, training)
      features = self.activations[i](features)
    features = self.dense_layers[-2](features)
    features = self.batch_layers[-1](features, training)
    features = self.activations[i](features)
    return features

#-----------------------------------------------
class modelnet_data_generator(tf.keras.utils.Sequence):

    def __init__(self, points, label_list, batch_size):
        self.points = points
        self.label_list = np.array(label_list,dtype=int)
        self.batch_size = batch_size
        self.epoch_size = len(self.points)

        self.on_epoch_end()

    def __len__(self):
      # number of batches per epoch
      return(int(np.floor(self.epoch_size/self.batch_size)))

    def __call__(self):
      out = self.__getitem__(self.index)
      self.index += 1
      return out

    def __getitem__(self,index):
      labels = self.label_list[index*self.batch_size:(index+1)*self.batch_size]
      points = self.points[index*self.batch_size:(index+1)*self.batch_size]
      features = tf.ones([self.batch_size,1024,1])
      sizes = tf.ones([self.batch_size],dtype=tf.int32)*1024
      return points, features, sizes, labels

    def on_epoch_end(self):
      # chooses random subset of point pairs per epoch
      shuffle = np.random.permutation(np.arange(0,len(self.points)))
      self.points = self.points[shuffle]
      self.label_list = self.label_list[shuffle]
      self.index = 0  

    def generator_function(self):
      while True:
        self.on_epoch_end()
        for i in range(len(self)):
          yield self[i]

#-----------------------------------------------

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

batch_size = 16
num_epochs = 100

hidden_size = 16
feature_sizes = [1,128,256,512,128,num_classes]
pool_radii = np.array([0.1, 0.2, np.sqrt(6)+0.1])
conv_radii = pool_radii

lr_decay=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,
    decay_steps=100,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)

model = mymodel(feature_sizes,hidden_size,pool_radii, conv_radii)
gen_train = modelnet_data_generator(train_data_points, train_labels, batch_size)
gen_test = modelnet_data_generator(test_data_points, test_labels, batch_size)

train_loss_results = []
train_accuracy_results = []
for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  i=0
  time_epoch_start = time.time()
  for points, features, sizes, labels in gen_train:
    # time_batch_start = time.time()
    with tf.GradientTape() as tape:
      logits = model(points,sizes,features,True)
      pred = tf.nn.softmax(logits, axis=-1)
      loss = loss_function(y_true=labels, y_pred=pred)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # time_batch_end = time.time()
    epoch_loss_avg.update_state(loss)
    epoch_accuracy.update_state(labels, pred)

    # print("Epoch {:03d}: Batch: {:03d}, Loss: {:.3f}, Accuracy: {:.3%} Time: {:.3f}ms".format(epoch, i,
    #                                                              epoch_loss_avg.result(),
    #                                                              epoch_accuracy.result(),
    #                                                              1000*(time_batch_end-time_batch_start)))
    i+=1
  time_epoch_end = time.time()
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  if epoch % 1 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%} Time: {:.3f}s".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result(),
                                                                time_epoch_end-time_epoch_start))
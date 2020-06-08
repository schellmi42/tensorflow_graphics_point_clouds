import tensorflow as tf 
import pc
import numpy as numpy

num = 1000
dim = 2
batch_size = 10

points = np.random.rand(num,dim)
batch_ids =np.random.randint(0,batch_size,num)

point_cloud = pc.PointCloud(points, batch_ids, batch_size)


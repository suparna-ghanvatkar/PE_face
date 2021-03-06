import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('train_cnn/mean.binaryproto','rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))[0,:,:,:].mean(0)
plt.imshow(arr, cmap=cm.Greys_r)
plt.show()

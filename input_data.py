import tensorflow as tf
import numpy as np


class InputData(object):

	def __init__(self):
		dicData1 = {
			'b': 'bin',
			'l': 'lay',
			'p': 'place',
			's': 'set'
		}
		dicData2 = {
			'b': 'blue',
			'g': 'green',
			'r': 'red',
			'w': 'white'
		}
		dicData3 = {
			'a': 'at',
			'b': 'by',
			'i': 'in',
			'w': 'with'
		}
		dicData5 = {
			'0': 'zero',
			'1': 'one',
			'2': 'two',
			'3': 'three',
			'4': 'four',
			'5': 'five',
			'6': 'six',
			'7': 'seven',
			'8': 'eight',
			'9': 'nine'
		}
		dicData6 = {
			'a': 'again',
			'n': 'now',
			'p': 'please',
			's': 'soon'
		}
		self.dataMap = {
			0: dicData1,
			1: dicData2,
			2: dicData3,
			4: dicData5,
			5: dicData6
		}
		chars = 'abcdefghijklmnopqrstuvwxyz '
		self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
		self.index = 0
    	        self.images_list = None
    	        self.read_data_list()
        def read_data_list(self,filename='/home/lt/videodata/result/data.txt'):
		file_names=[]
		with open(filename,'r') as f:
			for line in f.readlines():
		    	    line=line.strip('\n')
		    	    file_names.append(line)
		self.images_list=file_names
		np.random.shuffle(self.images_list)
	def file2label(self,filename):
		label=getlabel(filename)
		return label
	def convert_label(self,label):
		size=len(label)
		result_label=[]
		for i in range(31):
			if i>=size:
				result_label.append(char_to_ix[label[i]])
			else:
				result_label.append(char_to_ix[' '])
		return result_label
	# def preprocess_image(self,image):
	# def preprocess_label(self,label):
	def get_label(self,filename):
		label_index=filename.split('/')[-1]
		labels=[]
		labels.append('sil')
		labels.append(' ')
		for i,s in enumerate(label_index):
			if i != 3:
				s=dataMap[i][s]
			labels.append(s)
			labels.append(' ')
		labels.append('sil')
    	        return labels
	def read_images_from_disk(self,file_list):
		batch_size=len(file_list)
		image_bytes=75*100*50*3
		images_tensor=np.empty((batch_size,75*100*50*3))
		labels=[]
		for i in range(batch_size):
			with open(file[i]+'/result.binary','rb') as f:
				bytess=f.read(image_bytes)
				img = np.fromstring(bytess,dtype=np.uint8)
				images_tensor[i]=img
				label=file2label(file[i])
				label=convert_label(label)
				labels.append(label)
		images_tensor.reshape([batch_size,75,50,100,3])
                x_ix = []
                x_val = []
                for batch_i, batch in enumerate(labels):
                    for time, val in enumerate(batch):
                        x_ix.append([batch_i, time])
                        x_val.append(val)
                x_shape = [len(labels), np.asarray(x_ix).max(0)[1]+1]
                x_ix = tf.constant(x_ix, tf.int64)
                x_val = tf.constant(x_val, tf.int64)
                x_shape = tf.constant(x_shape, tf.int64)
    	        return images_tensor,x_ix,x_val,x_shape
	def get_bacth_data(self,batch_size):
		file_list=self.images_list[self.index:(self.index+1)*batch_size]
    	        image_batch,x_ix,x_val,x_shape=read_images_from_disk(file_list)
		# image_batch=preprocess_image(image_batch)
		# label_batch=preprocess_image(label_batch)
		self.index+=1
		return image_batch,x_ix,x_val,x_shape

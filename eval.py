import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.optimizers import Adam, Adagrad


def _data_loader(args):
	'''
	Loads precomputed features
	'''
	feature1 = np.load(args.feature1_path)
	feature2 = np.load(args.feature2_path)
	labels = np.load(args.labels_path)
	feature2_eval = feature2[args.val_start_index:]
	labels_eval = labels[args.val_start_index:]
	return feature1, feature2_eval, labels_eval
	
	
def _eval(args, feature1, feature2, labels):
	'''
	evaluates the performance of trained model on validation set
	'''
	model = load_model(args.load_model)
	val_loss, val_acc = model.evaluate({'A1':feature1,'B1':feature2},{'dense3':labels})
	print('mae score is: %f'%val_loss)
	print('evaluation completed')


if __name__=="__main__":
	parser = argparse.ArgumentParser()				
	parser.add_argument('--feature1_path', type=str, default='features/val_features.npy', help="Path to precomputed image features")		
	parser.add_argument('--feature2_path', type=str, default='features/feature2.npy', help="Path to gender features")		
	parser.add_argument('--labels_path', type=str, default='features/labels.npy', help="Path to labels")				
	parser.add_argument('--load_model', type=str, default='saved_model/train_500.model', help="Path to labels")				
	parser.add_argument('--val_start_index', type=int, default=12107, help="The index for split of train and val")	
	args=parser.parse_args()
	print '--------------------------------'
	for key, value in vars(args).items():
		print key, ' : ', value
	print '--------------------------------'
	feature1, feature2, labels = _data_loader(args)
	_eval(args, feature1, feature2, labels)
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.optimizers import Adam, Adagrad

import time

def _data_loader(args):
	'''
	Loads precomputed features
	'''
	feature1 = np.load(args.feature1_path)
	feature2 = np.load(args.feature2_path)
	labels = np.load(args.labels_path)
	feature2_train = feature2[:args.val_start_index]
	labels_train = labels[:args.val_start_index]
	return feature1, feature2_train, labels_train

	
def _model(args, feature1, feature2, labels):	
	'''
	Creates, trains and saves the model
	'''
	# creating model
	A1 = Input(shape=(2048,),name='A1')
	model1 = Model(A1)

	B1 = Input(shape=(1,),name='B1')
	B2 = Dense(32, activation=None,name='B2')(B1)
	model2 = Model(B1,B2)

	concatenated = concatenate([A1, B2])

	dense1 = Dense(1000, activation='relu',name='dense1')(concatenated)
	dense2 = Dense(1000, activation='relu',name='dense2')(dense1)
	dense3 = Dense(1, activation=None,name='dense3')(dense2)

	model = Model([A1, B1], dense3)

	# hyperparameters for training
	lr = args.lr
	batch_size = args.batch_size
	epochs = args.num_epochs
	decay_ratio = args.decay_ratio
	decay_rate = lr/decay_ratio
	
	# Defining optimizer
	if args.optimizer == 'adam':
		optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
		
	elif args.optimizer == 'adagrad':
		optimizer= Adagrad(lr=lr,epsilon=1e-08, decay=decay_rate)
	
	# Compiling model
	model.compile(optimizer=optimizer,
				  loss='mean_absolute_error',
				  metrics=['accuracy'])
	
	# Training model
	start = time.time()			  
	model.fit({'A1':feature1,'B1':feature2},{'dense3':labels}, batch_size=batch_size, epochs=epochs, verbose=1)
	end = time.time()

	print('Time elapsed: %f'%(end-start))
	model.save(args.save_path + '_ep' + str(epochs) + '_lr' + str(lr) + '_bs' + str(batch_size))
	
	print('model training completed')

def _train(args):
	feature1, feature2, labels = _data_loader(args)
	_model(args, feature1, feature2, labels)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=2, help="Number of epochs")		
    parser.add_argument('--embedding_dim', type=int, default=1000, help="Dimension of fully connected layer")		
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")		
    parser.add_argument('--feature1_path', type=str, default='features/train_features.npy', help="Path to precomputed image features")		
    parser.add_argument('--feature2_path', type=str, default='features/feature2.npy', help="Path to gender features")		
    parser.add_argument('--labels_path', type=str, default='features/labels.npy', help="Path to labels")		
    parser.add_argument('--save_path', type=str, default='saved_model/', help="Path to save the model")		
    parser.add_argument('--decay_ratio', type=int, default=100, help="The ration at which the learing rate is dereased")		
    parser.add_argument('--val_start_index', type=int, default=12107, help="The index for split of train and val")		
    parser.add_argument('--lr', type=float, default='0.001', help="learning rate")			
    args=parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    _train(args)

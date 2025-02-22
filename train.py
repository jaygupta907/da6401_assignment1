from dataset import Batch_Dataset
from config import get_args
from model import Sequential
from layer import Perceptron_Layer
import numpy as np

args = get_args()

data = Batch_Dataset('mnist')
data.preprocess()
train_batches = data.create_train_batches(batch_size=32, shuffle=True)
test_batches = data.create_test_batches(batch_size=32, shuffle=True)


model = Sequential()
for i in range(args.num_layers-1):
    model.add(Perceptron_Layer(784,args.hidden_size),activation='relu')
model.add(Perceptron_Layer(256,10),activation='softmax')

model.train(train_batches,test_batches,args)

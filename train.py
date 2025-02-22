from dataset import Batch_Dataset
from config import get_args
from model import Sequential
from layer import Perceptron_Layer
import numpy as np
import wandb

args = get_args()

wandb.init(project=args.wandb_project, entity=args.wandb_entity,name="run_1")

data = Batch_Dataset(args.dataset)
data.preprocess()
train_batches = data.create_train_batches(batch_size=args.batch_size, shuffle=True)
test_batches  = data.create_test_batches(batch_size=args.batch_size, shuffle=True)

model = Sequential(args=args)
model.add(Perceptron_Layer(784,args.hidden_size,weight_init=args.weight_init),activation='relu')
for i in range(args.num_layers):
    model.add(Perceptron_Layer(args.hidden_size,args.hidden_size,weight_init=args.weight_init),activation='relu')
model.add(Perceptron_Layer(args.hidden_size,10,weight_init=args.weight_init),activation='softmax')

model.train(train_batches,test_batches,args)

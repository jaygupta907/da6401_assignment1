from dataset import Batch_Dataset
from config import get_args
from model import Sequential
from layer import Perceptron_Layer
import numpy as np
import wandb
from optimizers import SGD,Momentum,Nesterov,Adagrad,RMSProp,Adam,Nadam

args = get_args()

wandb.init(project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,name=f"{args.optimizer}")

data = Batch_Dataset(args.dataset)

data.preprocess()
train_batches = data.create_train_batches(batch_size=args.batch_size, shuffle=True)
test_batches  = data.create_test_batches(batch_size=args.batch_size, shuffle=True)

input_dim = train_batches[0][0].shape[1]
num_classes = train_batches[0][1].shape[1]

model = Sequential(args=args)
model.add(Perceptron_Layer(input_dim,
                           args.hidden_size,
                           args,
                           weight_init=args.weight_init),
                           activation='relu')
for i in range(args.num_layers):
    model.add(Perceptron_Layer(args.hidden_size,
                               args.hidden_size,
                               args,
                               weight_init=args.weight_init),
                               activation='relu')
model.add(Perceptron_Layer(args.hidden_size,
                           num_classes,
                           args,
                           weight_init=args.weight_init),
                           activation='softmax')



model.train(train_batches,test_batches,args)

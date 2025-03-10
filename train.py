from dataset import Batch_Dataset
from config import get_args
from model import Sequential
from layer import Perceptron_Layer
import wandb
import numpy as np
args = get_args()

wandb.init(project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=f"opt_{args.optimizer}_lr_{args.learning_rate}_batch_{args.batch_size}_layer_{args.num_layers}_hidden_{args.hidden_size}_act_{args.activation}_decay_{args.weight_decay}_init_{args.weight_init}_epoch_{args.epochs}")


# wandb.init(project=args.wandb_project,
#             entity=args.wandb_entity,
#             config=args,
#             name=f"Best_Hyperparameters_mse")

# wandb.init(project=args.wandb_project,
#             entity=args.wandb_entity,
#             config=args,
#             name=f"Best_Hyperparameters_cross_entropy")


# wandb.init(project=args.wandb_project,
#             entity=args.wandb_entity,
#             config=args,
#             name=f"fashion_mnist_recommendations")


data = Batch_Dataset(args.dataset)
train_batches =  data.create_train_batches(batch_size=args.batch_size, shuffle=True)
test_batches  =  data.create_test_batches(batch_size=args.batch_size, shuffle=True)
val_batches   =  data.create_val_batches(batch_size=args.batch_size, shuffle=True)
classes =  data.classes
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
model.train(train_batches,test_batches,val_batches)
model.evaluate(test_batches,'test',classes)

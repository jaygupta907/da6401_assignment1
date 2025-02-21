from dataset import Batch_Dataset

data = Batch_Dataset('mnist')
data.preprocess()
train_batches = data.create_train_batches(batch_size=32, shuffle=True)
for x_batch, y_batch in train_batches:
    print(x_batch.shape, y_batch.shape)
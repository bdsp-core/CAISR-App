import numpy as np
from tensorflow import keras
import ray

@ray.remote
class DataLoader:
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = None
        self.current_index = 0
        self.x_train = None
        self.y_train = None
        self.load_data()

    def load_data(self):
        # Load your data here
        # Replace this with your actual data loading code
        #data = np.load(self.data_path)
        self.x_train = np.zeros((20,20))
        self.x_train = np.zeros((20))+1
        self.num_samples = len(self.x_train)
        np.random.shuffle(self.x_train)
        np.random.shuffle(self.y_train)

    def get_batch(self):
        if self.current_index >= self.num_samples:
            np.random.shuffle(self.x_train)
            np.random.shuffle(self.y_train)
            self.current_index = 0

        x_batch = self.x_train[self.current_index:self.current_index + self.batch_size]
        y_batch = self.y_train[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        return x_batch, y_batch

# Create a Ray actor for the DataLoader
data_path = 'path/to/your/data.npy'
batch_size = 32

data_loader = DataLoader.remote(data_path, batch_size)

# Example usage in Keras training loop
for epoch in range(10):
    # Get a batch of data from the DataLoader
    x_batch, y_batch = ray.get(data_loader.get_batch.remote())

    # Train your model using the batch of data
    #model.train_on_batch(x_batch, y_batch)

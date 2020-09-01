class SimCLR_TrainConfig(object):
    def __init__(self):
        self.feature_dim = 128       # Feature dim for latent vector
        self.temperature = 0.5       # Temperature used in softmax
        self.k = 200                 # Top k most similar images used to predict the label
        self.batch_size = 512        # Number of images in each mini-batch
        self.epochs = 500            # Number of sweeps over the dataset to train
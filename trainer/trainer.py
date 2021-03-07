import torch
import torch.nn
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_data, val_data):
        pass

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def display_results(self):
        raise NotImplementedError

    def update_log(self):
        raise NotImplementedError

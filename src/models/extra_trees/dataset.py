import logging
import numpy as np
from models.generic_dataset import GenericDataset


class Dataset(GenericDataset):
    ZPOLARITY = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
        6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

    ZSIGN = {-1: 'negative',  0.: 'neutral', 1: 'positive'}

    def __init__(self, config):
        super().__init__()
        train, test = self.data
        self.X_train, self.X_test, self.y_train = self.preprocessing(train, test)

    def preprocessing(self, train, test):
        

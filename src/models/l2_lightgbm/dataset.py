import numpy as np
import pandas as pd
import os
from models.generic_dataset import GenericDataset


class Dataset(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_names = []
        self.X_train = self.prepare_oofs()
        print('OOFs prepared.')
        self.y_train = self.prepare_labels()
        print('Labels prepared.')
        self.X_test = self.prepare_inferences()
        print('Inference prepared.')

    def prepare_labels(self):
        train, _ = self.data
        return train[self.config.LIST_CLASSES].values

    def prepare_oofs(self):
        oofs = []
        for oof_file in sorted(os.listdir(self.config.L1_OOF_DIR)):
            full_path = os.path.join(self.config.L1_OOF_DIR, oof_file)
            model_name = oof_file.split('__')[1]
            oofs.append((model_name, pd.read_csv(full_path)))
        data_oofs = np.hstack([np.array(oof_model[self.config.LIST_CLASSES])
                               for _, oof_model in oofs])
        self.model_names = [model_name for model_name, _ in oofs]
        return data_oofs

    def prepare_inferences(self):
        inferences = []
        for inf_file in sorted(os.listdir(self.config.L1_INFERENCE_DIR)):
            full_path = os.path.join(self.config.L1_INFERENCE_DIR, inf_file)
            model_name = inf_file.split('__')[1]
            inferences.append((model_name, pd.read_csv(full_path)))
        if sorted([model_name for model_name, _ in inferences]) == sorted(self.model_names):
            data_infs = np.hstack([np.array(inf_model[self.config.LIST_CLASSES])
                                         for _, inf_model in inferences])
            return data_infs
        raise RuntimeError('Models in OOF and INF do not match, terminating.')

    def engineer_feature(series, func, normalize=True):
        feature = series.apply(func)

        if normalize:
            feature = pd.Series(z_normalize(feature.values.reshape(-1,1)).reshape(-1,))
        feature.name = func.__name__
        return feature


    def engineer_features(series, funclist, normalize=True):
        features = pd.DataFrame()
        for func in funclist:
            feature = engineer_feature(series, func, normalize)
            features[feature.name] = feature
        return features

    def z_normalize(data):
        scaler.fit(data)
        return scaler.transform(data)

    def asterix_freq(x):
        return x.count('!')/len(x)

    def uppercase_freq(x):
        return len(re.findall(r'[A-Z]',x))/len(x)

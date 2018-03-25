import os
from models.generic_config import GenericConfig


class L2CatboostConfig(GenericConfig):
    DATA_DIR = GenericConfig.DATA_DIR
    L1_OOF_DIR = os.path.join(GenericConfig.DATA_DIR, 'processed/oof')
    L1_INFERENCE_DIR = os.path.join(GenericConfig.DATA_DIR, 'processed/inference')

    def display_info(self):
        config_info = "\nConfiguration:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                config_info += ("\n{:30} {}".format(a, getattr(self, a)))
        config_info += "\nOOF:\n"
        config_info += str(os.listdir(self.L1_OOF_DIR))
        return config_info

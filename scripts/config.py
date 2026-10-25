import torch

class Config:
    def __init__(self):
        self.SEED = 42
        self.MODEL_SAVE_PATH = 'best_calorie_model.pth'

        # Ингредиенты на английском -> используем английский BERT
        self.TEXT_MODEL = 'distilbert-base-uncased'
        self.IMAGE_MODEL = 'efficientnet_b0'

        self.IMG_SIZE = 224
        self.MAX_LENGTH = 64
        self.HIDDEN_DIM = 256

        self.BATCH_SIZE = 16
        self.EPOCHS = 20
        self.WEIGHT_DECAY = 1e-2

        # Доля валидации, которую выделяем из train для early stopping
        self.VAL_SPLIT = 0.15

        self.TEXT_LR = 1e-5
        self.IMAGE_LR = 1e-4
        self.CLASSIFIER_LR = 1e-3

        self.TARGET_MEAN = None
        self.TARGET_STD = None
        self.MASS_MEAN = None
        self.MASS_STD = None

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

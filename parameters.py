import torch
from torch import nn
from torchvision import transforms
from utilities import Utilities

RESULTS_PATH    = './results.csv'
DATA_DIR        = './input/'
TRAIN_ZIP_DIR   = DATA_DIR + 'train.zip'
TEST_ZIP_DIR    = DATA_DIR + 'test.zip'
SAMPLE_SUBMIS   = DATA_DIR + 'sample_submission.csv'
ANNOTATIONS_DIR = DATA_DIR + 'train.csv'
TRAIN_DIR       = './train'
TEST_DIR        = './test'
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SIZE      = (1, 3, 32, 32)
N_LABELS        = 2
N_EPOCHS        = 10
BATCH_SIZE      = 64
VALIDATION_PERC = 0.2
AUC_PERC        = 0.1
LOSS_FN         = nn.NLLLoss()
OPTIMIZER       = torch.optim.Adam
LEARNING_RATE   = 0.001
MOMENTUM        = 0.9
MEAN            = (0.5, 0.5, 0.5)
STD             = (0.5, 0.5, 0.5)
LABELS_MAP      = {0: 'No Cactus', 1: 'Cactus'}
TRANSFORM       = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])
U               = Utilities(DEVICE)

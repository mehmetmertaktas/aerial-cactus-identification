from torch import load
from parameters import DEVICE, INPUT_SIZE, RESULTS_PATH
from utils import save_results
from load_data import test_dl

model = load('./model.pth', map_location=DEVICE)

print('\nSaving results...')

save_results(RESULTS_PATH, test_dl, model)

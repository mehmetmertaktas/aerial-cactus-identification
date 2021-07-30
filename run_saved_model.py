from torchinfo import summary
from parameters import DEVICE, INPUT_SIZE, RESULTS_PATH, U
from load_data import *

model = torch.load('./model.pth', map_location=DEVICE)
print('\n==================================== MODEL SUMMARY =======================================')
summary(model, input_size=INPUT_SIZE)

print('\nSaving results...')

U.save_results(RESULTS_PATH, test_data, model)

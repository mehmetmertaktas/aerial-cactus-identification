import torch
from torchinfo import summary
from model import Net
from load_data import *
from parameters import *

U.display_data(data, classes=LABELS_MAP)

model = Net().to(DEVICE)
model.apply(U.init_weights)

print('\n==================================== MODEL SUMMARY =======================================')
summary(model, input_size=INPUT_SIZE)

print('\nTraining... (with validation)\n')

optimizer = OPTIMIZER(model.parameters(),
                             lr=LEARNING_RATE)
history = U.train(model,
                  train_dl,
                  valid_dl,
                  optimizer=optimizer,
                  lr=LEARNING_RATE,
                  epochs=N_EPOCHS,
                  loss_fn=LOSS_FN)

U.plot_history(history, validation=True)

U.AUC(model, auc_dl)

model_ = Net().to(DEVICE)
model_.apply(U.init_weights)

print('\nTraining... (without validation)\n')

optimizer = OPTIMIZER(model_.parameters(),
                             lr=LEARNING_RATE)
history = U.train(model_,
                  all_dl,
                  optimizer=optimizer,
                  lr=LEARNING_RATE,
                  epochs=N_EPOCHS,
                  loss_fn=LOSS_FN)

U.plot_history(history)

torch.save(model_, './model.pth')

print('\nSaving results...')

U.save_results(RESULTS_PATH, test_data, model_)

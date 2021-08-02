import torch
from model import Net
from load_data import (
    data,
    train_dl,
    valid_dl,
    auc_dl,
    all_dl,
    test_dl
)
from utils import (
    display_data,
    init_weights,
    train,
    plot_history,
    plot_auc_curve,
    save_results
)
from parameters import (
    LABELS_MAP,
    DEVICE,
    INPUT_SIZE,
    OPTIMIZER,
    LEARNING_RATE,
    N_EPOCHS,
    LOSS_FN,
    RESULTS_PATH
)

display_data(data, classes=LABELS_MAP)

model = Net().to(DEVICE)
model.apply(init_weights)

print('\nTraining... (with validation)\n')

optimizer = OPTIMIZER(model.parameters(),
                      lr=LEARNING_RATE)
history = train(model,
                train_dl,
                valid_dl,
                optimizer=optimizer,
                lr=LEARNING_RATE,
                epochs=N_EPOCHS,
                loss_fn=LOSS_FN)

plot_history(history, validation=True)

plot_auc_curve(model, auc_dl)

model_ = Net().to(DEVICE)
model_.apply(init_weights)

print('\nTraining... (without validation)\n')

optimizer = OPTIMIZER(model_.parameters(),
                      lr=LEARNING_RATE)
history = train(model_,
                all_dl,
                optimizer=optimizer,
                lr=LEARNING_RATE,
                epochs=N_EPOCHS,
                loss_fn=LOSS_FN)

plot_history(history)

torch.save(model_, './model.pth')

print('\nSaving results...')

save_results(RESULTS_PATH, test_dl, model_)

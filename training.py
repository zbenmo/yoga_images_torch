# training.py

import torch
import torch.nn as nn


def train(train_loader, model, optimizer, devic):
    model.train()

    for data in train_loader:
        inputs = data['image']
        targets = data['targets']

        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grads()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()


def evaluate(valid_loader, model, device):
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():

        for data in valid_loader:
            inputs = data['image']
            targets = data['targets']

            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(inputs)

            final_targets.extend(targets)
            final_outputs.extend(outputs)

    return final_outputs, final_targets

import torch
from torch import nn
from torchmetrics import Accuracy

import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm   # for progress bar

from timeit import default_timer as timer

def print_train_time(start: float, end: float):
    total_time = end - start
    print(f"Total Time: {total_time:.3f}s")
    return total_time

print(torch.__version__)
print(torchvision.__version__)

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(), target_transform=None)

print(len(train_data))
print(len(test_data))
class_names = train_data.classes
print(class_names)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(len(train_dataloader), len(test_dataloader))

class CVModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)



def train_step(model, optimizer, loss_fn, accuracy, train_dataloader):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy(y_pred, y) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    
def test_step(model, loss_fn, accuracy, test_dataloader):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = model(X_test)
            loss = loss_fn(test_pred, y_test)
            test_loss +=  loss
            test_acc += accuracy(test_pred, y_test)

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

torch.manual_seed(42)

epochs = 5

model_2 = CVModelV1(28*28, 10, len(class_names))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
accuracy = Accuracy(task='multiclass', num_classes=len(class_names))

train_time_start = timer()

is_func = False

if is_func:

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-----")
        train_step(model_2, optimizer, loss_fn, accuracy, train_dataloader)
        test_step(model_2, loss_fn, accuracy, test_dataloader)

else:

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-----")
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(train_dataloader):
            model_2.train()
            y_pred = model_2(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy(y_pred, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)  # find average train_loss per batch for that particular epochs
        train_acc /= len(train_dataloader)

        test_loss, test_acc = 0, 0
        model_2.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                test_pred = model_2(X_test)
                loss = loss_fn(test_pred, y_test)
                test_loss += loss
                test_acc += accuracy(test_pred, y_test)

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"\n Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f} | Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}")


    train_time_end = timer()
    total_train_time = print_train_time(train_time_start, train_time_end)
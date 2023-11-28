import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pkl
import datetime

PROJECT_NAME = 'pytorch-resume-run'
CHECKPOINT_PATH = './checkpoint.pth'
N_EPOCHS = 100
TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M")

# Dummy data
X = torch.randn(64, 8, requires_grad=True)
Y = torch.empty(64, 1).random_(2)
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
metric = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epoch = 0
run = wandb.init(
    project=PROJECT_NAME,
    entity='dt-collision-avoidance',
    id=TIME,
    resume='must'
    )
print(run.resumed)
print(run.path)
if wandb.run.resumed:
    wandb.restore(CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f'Start: {epoch}')

model.train()
while epoch < N_EPOCHS:
    time.sleep(0.1)
    print(f'Epoch: {epoch}', end='\r')
    optimizer.zero_grad()
    output = model(X)
    loss = metric(output, Y)
    wandb.log({'loss': loss.item()}, step=epoch)
    loss.backward()
    optimizer.step()

    torch.save({ # Save our checkpoint loc
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, CHECKPOINT_PATH, pickle_protocol=pkl.HIGHEST_PROTOCOL)
    wandb.save(CHECKPOINT_PATH) # saves checkpoint to wandb
    if epoch==50:
        break
    epoch += 1
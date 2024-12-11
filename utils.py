import torch
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

def save_checkpoint(model, optimizer, filename="model.pth"):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    state = torch.load(filename)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])

import numpy as np
import torch
import os




def save_checkpoint(model,acc=-1,checkpoint_dirname='checkpoints',version=1):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'trainAcc': acc
    }
    filename = os.path.join(checkpoint_dirname,'{}_v{}.pth'.format(arch,version))
    torch.save(state, filename)


def resume_checkpoint(model,checkpoint_dirname='checkpoints',version=1):
    arch = type(model).__name__
    filename = os.path.join(checkpoint_dirname,'{}_v{}.pth'.format(arch,version))
        
    message = "There's not checkpoint"
    assert os.path.exists(filename),message

    print("Loading checkpoint: {} ...".format(filename))
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    return model
    
def show_checkpoint(model_name,checkpoint_dirname='checkpoints',version=1):
    
    filename = os.path.join(checkpoint_dirname,'{}_v{}.pth'.format(model_name,version))    
    message = "There's not checkpoint"
    assert os.path.exists(filename),message
    checkpoint = torch.load(filename)
    print('arch:',checkpoint['arch'])
    print('trainAcc:',checkpoint['trainAcc'])
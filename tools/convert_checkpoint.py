import torch
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1:
        loc_func = lambda storage, loc: storage
        states = torch.load(sys.argv[1], map_location=loc_func)
    else:
        print('usage python tools/convert_checkpoint.py [filename]')
        exit(0)
    print('epoch %d' % states['epoch'])
    os.makedirs('checkpoints', exist_ok=True)
    states = states['model_state_dict']
    model_state = {}
    for key in states.keys():
        newKey = key
        if 'module.' in newKey:
            newKey = newKey[7:]
        model_state[newKey] = states[key]
    torch.save(model_state, 'checkpoints/attention_loc.pth.tar')

import torch
import torchvision
from .networks import VisualNet, AudioNet, weights_init

class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=''):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net
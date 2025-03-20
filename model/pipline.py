import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _Loss
import os

class Pipeline(torch.nn.Module):
        def __init__(self, visual_prompt, s_model, label_mapping, args):
            super(Pipeline, self).__init__()
            self.visual_prompt = visual_prompt
            self.s_model = s_model
            self.label_mapping = label_mapping
        
        def train(self, mode=True):
            self.training = mode
            self.visual_prompt.train(mode)
            self.s_model.eval()
            self.label_mapping.train(mode)
        
        # def eval(self):
        #     self.training = False
        #     self.visual_prompt.eval()
        #     self.s_model.eval()
        #     self.label_mapping.eval()
            
        def forward(self, x):
            out = self.visual_prompt(x)
            out = self.s_model(out)
            out = self.label_mapping(out)
            return out
"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm

import numpy as np

def identity(x):
    return x

def read_dim(s):
    a, b, c, d, e = s.split('.')
    return [int(a), int(b), int(c), int(d), int(e)]

eps = 1e-11

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            set_activation=torch.sum,
            set_output_size=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.set_activation = set_activation
        self.set_output_size = set_output_size
        self.b_init_value = b_init_value
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)

class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hn', torch.zeros(1, 1, self.hidden_dim))
        self.register_buffer('cn', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (1, task, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            if self.layer_norm and i < len(self.fcs) - 1:
                out = self.layer_norms[i](out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hn, self.cn))
        self.hn = hn
        self.cn = cn
        # take the last hidden state to predict z
        # out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hn = self.hn.new_full((1, num_tasks, self.hidden_dim), 0)
        self.cn = self.cn.new_full((1, num_tasks, self.hidden_dim), 0)
        
        
def sample_gumbel(shape,eps=1e-20,use_cuda=True):
    U=torch.rand(shape)
    if use_cuda:
        U=U.cuda()
        
    return -torch.log(-torch.log(U+eps)+eps)
    
            
def gumbel_softmax_sample(logits,temperature):
    y=logits+sample_gumbel(logits.size())
    
    return F.softmax(y/temperature,dim=-1)
    
    
def gumbel_softmax(logits,temperature,hard=False):
    y=gumbel_softmax_sample(logits,temperature)
    
    if not hard:
        return y

    shape=y.size()
    _,ind=y.max(dim=-1) 
    y_hard=torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard=y_hard.view(*shape) 
    
    y_hard=(y_hard-y).detach()+y
    
    return y_hard 


class Softmax_Net(nn.Module):
    def __init__(self,
                 dim_xz,
                 experts_in_gates,
                 dim_logit_h,
                 num_logit_layers,
                 num_experts):
        super().__init__()
        self.dim_xz=dim_xz
        self.experts_in_gates=experts_in_gates
        self.dim_logit_h=dim_logit_h
        self.num_logit_layers=num_logit_layers
        self.num_experts=num_experts
        
        self.logit_modules=[]
        if self.experts_in_gates:
            self.logit_modules.append(nn.Linear(self.dim_xz, self.dim_logit_h))
            for i in range(self.num_logit_layers):
                self.logit_modules.append(nn.ReLU())
                self.logit_modules.append(nn.Linear(self.dim_logit_h, self.dim_logit_h))
            self.logit_modules.append(nn.ReLU())
            self.logit_modules.append(nn.Linear(self.dim_logit_h, 1))
        else:
            self.logit_modules.append(nn.Linear(self.dim_xz, self.dim_logit_h))
            for i in range(self.num_logit_layers):
                self.logit_modules.append(nn.ReLU())
                self.logit_modules.append(nn.Linear(self.dim_logit_h, self.dim_logit_h))
            self.logit_modules.append(nn.ReLU())
            self.logit_modules.append(nn.Linear(self.dim_logit_h, self.num_experts))            
        self.logit_net=nn.Sequential(*self.logit_modules)
        
    def forward(self,x_z,temperature,hard=False,gumbel_max=False):
        if self.experts_in_gates:
            logit_output=self.logit_net(x_z)
        else:
            x_z=torch.mean(x_z,dim=-2)
            logit_output=self.logit_net(x_z)
        
        if not self.experts_in_gates:
            logit_output=logit_output.unsqueeze(-1)
        
        if gumbel_max:
            logit_output=logit_output+sample_gumbel(logit_output.size())
        
        softmax_y=F.softmax(logit_output/temperature,dim=-2)
        
        if not hard:
            return softmax_y 
        else:
            softmax_y=softmax_y.squeeze(-1) 
            shape=softmax_y.size()
            _,ind=softmax_y.max(dim=-1)
            y_hard=torch.zeros_like(softmax_y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard=y_hard.view(*shape) 
            
            y_hard=(y_hard-softmax_y).detach()+softmax_y
            
            softmax_y,y_hard=softmax_y.unsqueeze(-1),y_hard.unsqueeze(-1)
            
            return softmax_y, y_hard            
        
        
        

    



"""
Launcher for experiments with MoE-NP. Here the model corresponds to MoE_PEARL class since this is within the PEARL framework.
"""

import os
import pathlib
import numpy as np
import click
import json
import torch
import torch.nn as nn

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, Softmax_Net
from rlkit.torch.sac.sac import MoE_PEARLSoftActorCritic
from rlkit.torch.sac.agent import MoE_PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs_moe_pearl.default_moe import default_moe_config


def experiment(variant):

    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim \
        if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    
    num_experts = variant['num_experts']
    dim_logit_h = variant['dim_logit_h']
    num_logit_layers = variant['num_logit_layers']
    
    temperature = variant['temperature']
    hard = variant['hard']
    gumbel_max = variant['gumbel_max']

    expert_modules = nn.ModuleList([encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    ) for i in range(num_experts)]) 
    
    experts_in_gates = variant['experts_in_gates']
    
    if variant['experts_in_gates']:
        softmax_gates = Softmax_Net(
            dim_xz=obs_dim+latent_dim, 
            experts_in_gates=experts_in_gates,
            dim_logit_h=dim_logit_h, num_logit_layers=num_logit_layers, 
            num_experts=num_experts,
        )
    else:
        softmax_gates = Softmax_Net(
            dim_xz=obs_dim, 
            experts_in_gates=experts_in_gates,
            dim_logit_h=dim_logit_h, num_logit_layers=num_logit_layers, 
            num_experts=num_experts,
        )
        
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = MoE_PEARLAgent(
        latent_dim=latent_dim, num_experts=num_experts, 
        expert_modules=expert_modules, experts_in_gates=experts_in_gates, softmax_gates=softmax_gates, 
        temperature=temperature, hard=hard, gumbel_max=gumbel_max, policy=policy,
        **variant['algo_params']
    )
    algorithm =MoE_PEARLSoftActorCritic(
        env=env, 
        train_tasks=list(tasks[:variant['n_train_tasks']]), 
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]), 
        latent_dim=latent_dim, 
        num_experts=num_experts, 
        nets=[agent, qf1, qf2, vf], 
        **variant['algo_params']
    )

    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))

        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, \
                                      exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    algorithm.train() 

def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def main(config='./configs_moe_pearl/humanoid-multi-goal-0.2.json', gpu=0):
    variant = default_moe_config 

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)




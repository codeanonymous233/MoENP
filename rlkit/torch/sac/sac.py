from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import math

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

import os
import pdb


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            # L[int(i+c*period)] = v
            yield v
            v += step
            i += 1
        yield 1.
    return L 



class MoE_PEARLSoftActorCritic(MetaRLAlgorithm):
    
    def __init__(
        self,
        env,
        train_tasks,
        eval_tasks,
        latent_dim,
        num_experts,
        nets,
        
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        context_lr=1e-3,
        gate_lr=1e-3,
        kl_lambda0=1.0,
        kl_lambda1=1.0,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.0,
        optimizer_class=optim.Adam,
        recurrent=False,
        use_information_bottleneck=True,
        use_next_obs_in_context=False,
        sparse_rewards=False,
        whether_moe_critic_loss=False,
    
        soft_target_tau=1e-2,
        plotter=None,
        render_eval_paths=False,
        **kwargs):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            whether_moe=True,
            **kwargs
        )
        
        self.soft_target_tau=soft_target_tau
        self.policy_mean_reg_weight=policy_mean_reg_weight
        self.policy_std_reg_weight=policy_std_reg_weight
        self.policy_pre_activation_weight=policy_pre_activation_weight
        self.plotter=plotter
        self.render_eval_paths=render_eval_paths
    
        self.recurrent=recurrent
        self.latent_dim=latent_dim
        self.qf_criterion=nn.MSELoss()
        self.vf_criterion=nn.MSELoss()
        self.vib_criterion=nn.MSELoss()
        self.l2_reg_criterion=nn.MSELoss()
        self.kl_lambda0=kl_lambda0
        self.kl_lambda1=kl_lambda1
        
        self.use_information_bottleneck = use_information_bottleneck
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = sparse_rewards
        self.whether_moe_critic_loss = whether_moe_critic_loss
        
        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()
        
        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.expert_optimizers = [optimizer_class(
            self.agent.expert_modules[i].parameters(),
            lr=context_lr,
        ) for i in range(self.agent.num_experts)]
        self.softmax_gates_optimizer = optimizer_class(
            self.agent.softmax_gates.parameters(),
            lr=gate_lr,
        )
        
    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks+[self.agent]+[self.qf1,self.qf2,self.vf,self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)        
    
    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]
    
    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked    
    
    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]

        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]

        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context
    
    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)

            self.agent.detach_z()
            self.agent.detach_alpha()
            
    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q
    
    def _moe_min_q(self, obs_unsq, actions_unsq, z_experts_unsq, alpha):
        q1 = self.qf1(obs_unsq, actions_unsq, z_experts_unsq.detach()) 
        q2 = self.qf2(obs_unsq, actions_unsq, z_experts_unsq.detach()) 
        '''
        q1,q2 = torch.sum(torch.mul(q1, alpha.detach()),dim=-2), torch.sum(torch.mul(q2, alpha.detach()),dim=-2) 
        
        q1,q2 = torch.mul(q1, alpha.detach()), torch.mul(q2, alpha.detach()) 
        '''
        
        min_q = torch.min(q1, q2)
        return min_q        
        
    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):
        with torch.autograd.set_detect_anomaly(True):

            num_tasks = len(indices)
    
            # data is (task, batch, feat)
            obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
    
            # run inference in networks
            policy_outputs, alpha, z_experts_unsq, task_z = self.agent(obs, context)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            

            # flattens out the task dimension
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
    
            # Q and V networks
            q1_pred = self.qf1(obs, actions, task_z)
            q2_pred = self.qf2(obs, actions, task_z)
            v_pred = self.vf(obs, task_z.detach())
            
            # get targets for use in V and Q updates
            with torch.no_grad():
                target_v_values = self.target_vf(next_obs, task_z)

            if self.use_information_bottleneck:
                for i in range(self.agent.num_experts):
                    self.expert_optimizers[i].zero_grad()                 
                cont_kl_div = self.agent.compute_cont_kl_div()
                for i in range(self.agent.num_experts):
                    cont_kl_loss = self.kl_lambda0 * cont_kl_div[i]
                    cont_kl_loss.backward(retain_graph=True) 
            
            
            if self.use_information_bottleneck:
                cat_dim = self.agent.num_experts
                self.softmax_gates_optimizer.zero_grad()
                cat_kl_div = self.agent.compute_cat_kl_div(alpha,cat_dim)
                cat_kl_loss = self.kl_lambda1 * cat_kl_div
                cat_kl_loss.backward(retain_graph=True)


            # qf and encoder update (note encoder does not get grads from policy or vf)
            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
            
            rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
            rewards_flat = rewards_flat * self.reward_scale
            terms_flat = terms.view(self.batch_size * num_tasks, -1)
            q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
            qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)                
            
            
            qf_loss.backward() 
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()
            for i in range(self.agent.num_experts):
                self.expert_optimizers[i].step() 
            self.softmax_gates_optimizer.step()
            
            min_q_new_actions = self._min_q(obs, new_actions, task_z)
            
            # vf update
            v_target = min_q_new_actions - log_pi
            vf_loss = self.vf_criterion(v_pred, v_target.detach())
                
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            self._update_target_network()
    
            # policy update
            log_policy_target = min_q_new_actions
            
            policy_loss = (
                    log_pi - log_policy_target
                        ).mean()
    
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            pre_tanh_value = policy_outputs[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss
    
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means_experts[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars_experts[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Continuous Divergence'] = ptu.get_numpy(sum(cont_kl_div))
                self.eval_statistics['KL Categorical Divergence'] = ptu.get_numpy(cat_kl_div)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            expert_modules=[self.agent.expert_modules[i].state_dict()
                            for i in range(self.agent.num_experts)],
            softmax_gates=self.agent.softmax_gates.state_dict()
        )
        return snapshot    
        

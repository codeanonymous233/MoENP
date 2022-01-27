import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2

    

class MoE_PEARLAgent(nn.Module):
    
    def __init__(
        self,
        latent_dim,
        num_experts,
        expert_modules,
        experts_in_gates,
        softmax_gates,
        temperature,
        hard,
        gumbel_max,
        policy,
        **kwargs):
        super().__init__()
        self.latent_dim=latent_dim
        
        self.num_experts=num_experts
        self.expert_modules=expert_modules
        self.experts_in_gates=experts_in_gates 
        self.softmax_gates=softmax_gates 
        self.temperature=temperature
        self.hard=hard
        self.gumbel_max=gumbel_max
        self.policy=policy
        
        self.recurrent=kwargs['recurrent']
        self.use_ib=kwargs['use_information_bottleneck']
        self.sparse_rewards=kwargs['sparse_rewards']
        self.use_next_obs_in_context=kwargs['use_next_obs_in_context'] 
        
        self.register_buffer('z_experts',torch.zeros(1, num_experts, latent_dim))
        self.register_buffer('z_means_experts',torch.zeros(1, num_experts, latent_dim))
        self.register_buffer('z_vars_experts',torch.zeros(1, num_experts, latent_dim))
        
        self.register_buffer('alpha',torch.zeros(1, num_experts, 1))
        
        self.clear_z()
   
    
    def sample_single_expert_z(self,z_means,z_vars): 
        posteriors=[torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))] # len(posteriors)=1
        z=[d.rsample() for d in posteriors]
        z_expert= torch.stack(z)
        
        return z_expert
        
    def sample_z(self):
        z_experts_list=[(self.sample_single_expert_z(self.z_means_experts[:,i,:],self.z_vars_experts[:,i,:])).unsqueeze(1)  
                        for i in range(self.num_experts)]
        z_experts=torch.cat(z_experts_list,dim=1)
        self.z_experts=z_experts
        
    
    def clear_z(self,num_tasks=1):
        assert self.recurrent==False
        
        z_means_experts_list=[(ptu.zeros(num_tasks,self.latent_dim)).unsqueeze(1) 
                              for i in range(self.num_experts)]
        z_means_experts=torch.cat(z_means_experts_list,dim=1)
        
        if self.use_ib:
            z_vars_experts_list=[(ptu.ones(num_tasks,self.latent_dim)).unsqueeze(1) 
                                 for i in range(self.num_experts)]
            z_vars_experts=torch.cat(z_vars_experts_list,dim=1)
        else:
            z_vars_experts_list=[(ptu.zeros(num_tasks,self.latent_dim)).unsqueeze(1) 
                             for i in range(self.num_experts)]
            z_vars_experts=torch.cat(z_vars_experts_list,dim=1)
            
        self.z_means_experts=z_means_experts
        self.z_vars_experts=z_vars_experts
        self.sample_z()
        self.context = None
    
        
    def detach_z(self):
        self.z_experts=self.z_experts.detach()
    
    def detach_alpha(self):
        self.alpha=self.alpha.detach()
    
    def update_context(self, inputs):
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
       
    
    def compute_gaussian_div(self,latent_dim,z_means,z_vars):
        prior = torch.distributions.Normal(ptu.zeros(latent_dim), 
                                           ptu.ones(latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) 
                          for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) 
                       for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))  
        
        return kl_div_sum
    
    
    def compute_cont_kl_div(self):
        kl_gaussian_experts=[self.compute_gaussian_div(self.latent_dim, self.z_means_experts[i],
                                  self.z_vars_experts[i]) 
                             for i in range(self.num_experts)]
        
        return kl_gaussian_experts
    
    
    def compute_cat_kl_div(self,alpha,cat_dim):
        log_ratio = torch.log(alpha * cat_dim + 1e-20) 
        kl_div_sum = torch.sum(alpha * log_ratio, dim=-1).mean()  
        return kl_div_sum
    
    
    def amort_param_trans(self,params):
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params]) 
        
        return z_means,z_vars
    
    
    def infer_posterior(self,context):
        params_experts=[self.expert_modules[i](context) 
                        for i in range(self.num_experts)]
        params_experts=[params_experts[i].view(context.size(0), -1, self.expert_modules[i].output_size)
                        for i in range(self.num_experts)] 
        
        if self.use_ib:
            z_means_experts_list=[self.amort_param_trans(params_experts[i])[0] 
                                  for i in range(self.num_experts)] 
            self.z_means_experts=torch.cat([z_means.unsqueeze(1) for z_means in z_means_experts_list],dim=1)  
            
            z_vars_experts_list=[self.amort_param_trans(params_experts[i])[1] 
                                 for i in range(self.num_experts)]
            self.z_vars_experts=torch.cat([z_vars.unsqueeze(1) for z_vars in z_vars_experts_list],dim=1) 
        else:
            z_means_experts_list=[torch.mean(params_experts[i],dim=1) 
                                  for i in range(self.num_experts)]
            self.z_means_experts=torch.cat([z_means.unsqueeze(1) for z_means in z_means_experts_list],dim=1)
            
        self.sample_z() 
        
    
    def get_action(self, obs, deterministic=False):
        z_experts_unsq=self.z_experts 
        obs_unsq=torch.cat([(ptu.from_numpy(obs[None])).unsqueeze(1) for i in range(self.num_experts)],dim=1) 
        
        if self.experts_in_gates:
            x_z=torch.cat((obs_unsq,z_experts_unsq),dim=-1)
            alpha,y_hard=self.softmax_gates(x_z,self.temperature,self.hard,self.gumbel_max) 
        else:
            alpha,y_hard=self.softmax_gates(obs_unsq,self.temperature,self.hard,self.gumbel_max) 
        
        self.alpha=alpha
        obs_expert=torch.mul(y_hard,z_experts_unsq) 
        z=torch.sum(obs_expert,dim=-2)             
        
        obs = ptu.from_numpy(obs[None])
        in_=torch.cat([obs, z], dim=-1)
        
        return self.policy.get_action(in_,deterministic=deterministic)    
        
    
    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)    
        
    
    def forward(self,obs,context):
        t,b,_=obs.size() 
        self.infer_posterior(context)
        self.sample_z()

        task_z_experts=self.z_experts
        z_experts_unsq=task_z_experts.unsqueeze(1).expand(-1,obs.size()[1],-1,-1) 
        obs_unsq=obs.unsqueeze(2).expand(-1,-1,task_z_experts.size()[1],-1) 
        
        if self.experts_in_gates:
            x_z=torch.cat((obs_unsq,z_experts_unsq),dim=-1) 
            alpha,y_hard=self.softmax_gates(x_z,self.temperature,self.hard,self.gumbel_max) 
        else:
            alpha,y_hard=self.softmax_gates(obs_unsq,self.temperature,self.hard,self.gumbel_max) 
            
        obs_expert=torch.mul(y_hard,z_experts_unsq) 
        expert_z=torch.sum(obs_expert,dim=2)
        
        obs=obs.view(t * b, -1) 
        task_z=expert_z.view(t * b, -1) 
        in_=torch.cat([obs,task_z.detach()],dim=-1) 
        
        policy_outputs=self.policy(in_,reparameterize=True, return_log_prob=True) 
        
        return policy_outputs,alpha,z_experts_unsq,task_z
        

    def log_diagnostics(self, eval_statistics):
        pass
    
    
    @property
    def networks(self):
        return [*(self.expert_modules),self.softmax_gates,
                self.policy]    

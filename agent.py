from models import Actor,Critic


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

def tie_projecion_layers(actor,critic):
    assert (len(actor.proj) == len(critic.proj)), "The two models must share the same architecture"
    N = len(actor.proj)
    for i in range(N):
        assert (type(actor.proj[i])==type(critic.proj[i])), "The two models must share the same architecture"
        actor.proj[i].weight = critic.proj[i].weight
        actor.proj[i].bias = critic.proj[i].bias

class SAC(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        encoder,
        encoder_output,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        log_interval=100,
        detach_encoder=True,
        curl_latent_dim=128,
        encoder_tau = .05,
        projection_dim = 128,
        **kwargs
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_tau = encoder_tau
        self.projection_dim = projection_dim
        
        self.actor = Actor(
            encoder,self.detach_encoder, action_shape, hidden_dim,
            actor_log_std_min, actor_log_std_max,encoder_output,projection_dim
        ).to(device)

        self.critic = Critic(
            encoder,self.detach_encoder,encoder_output, action_shape, hidden_dim,projection_dim
        ).to(device)

        self.critic_target = Critic(
            encoder,self.detach_encoder,encoder_output, action_shape, hidden_dim,projection_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        tie_projecion_layers(self.actor,self.critic)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.proj.parameters())+list(self.critic.Q1.parameters())+list(self.critic.Q2.parameters()), 
            lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.critic_target.train()
        self.critic.train()
        self.actor.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _,f = self.actor(obs)
            return mu.cpu().data.numpy().flatten(),f.cpu().data.numpy()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            _,_, pi, _,f  = self.actor(obs)
            return pi.cpu().data.numpy().flatten(),f.cpu().data.numpy()

    def critic_step(self,obs,action,reward,next_obs,not_done,L,step):
        with torch.no_grad():
            _,_,pi,log_pi,_=self.actor(next_obs,from_obs=False)
            target_Q1, target_Q2 = self.critic_target(next_obs, pi,from_obs=False)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
        
        current_Q1, current_Q2 = self.critic(
            obs, action,from_obs=False)

        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def actor_alpha_step(self,obs,L,step):
        _,_,pi,log_pi,_ = self.actor(obs,from_obs=False)
        actor_Q1, actor_Q2 = self.critic(obs, pi,from_obs=False)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self,replay_buffer,L,step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.critic_step(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.actor_alpha_step(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            if not self.detach_encoder:
                utils.soft_update_params(
                    self.critic.encoder, self.critic_target.encoder,
                    self.encoder_tau
                )
                utils.soft_update_params(
                        self.critic.proj, self.critic_target.proj,
                        self.encoder_tau
                )
            else:
                utils.soft_update_params(
                        self.critic.proj, self.critic_target.proj,
                        self.encoder_tau
                    )
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
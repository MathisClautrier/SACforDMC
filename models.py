import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_FREQ = 10000

def encode(encoder,inputs,feature_dim):
    assert (inputs.shape[1] % 3 == 0), "Inputs must be stacked RGB frames"
    assert (len(inputs.shape) == 4), "Inputs must be Bx3*nb_framesxWxH"
    if inputs.max() >1:
        inputs = inputs/255.
    batch,nb_frames,W,H = inputs.shape
    nb_frames = nb_frames//3
    inputs = inputs.view(batch*nb_frames,3,W,H)
    z = encoder(inputs).view(batch,nb_frames,feature_dim)
    z_t = z[:,1:,:]
    z_t_1 = z[:,:-1,:].detach()
    delta = z_t - z_t_1
    f = torch.cat([z,delta], axis = 1)
    f = torch.flatten(f,1,2)
    return f

class Actor(nn.Module):
    def __init__(self,encoder,freeze_encoder,action_shape,hidden_dim,log_std_min,log_std_max,
    feature_dim,projection_dim):
        super().__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            self.encoder.eval()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.proj = nn.Sequential(
            nn.Linear(5*feature_dim,projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.pi = nn.Sequential(
            nn.Linear(projection_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*action_shape[0])
        )
        self.feature_dim = feature_dim
        self.init_weights()
        self.action_shape = action_shape

    def init_weights(self):
        for i in range(5):
            if i %2 ==0:
                nn.init.orthogonal_(self.pi[i].weight.data)
                self.pi[i].bias.data.fill_(0.0)

    def transform(self,inputs):
        inputs = torch.FloatTensor(inputs).to('cuda')
        inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            f = encode(self.encoder,inputs,self.feature_dim)
        return f

    def forward(self,inputs,from_obs = True):
        if from_obs:
            if self.freeze_encoder:
                with torch.no_grad():
                    f = encode(self.encoder,inputs,self.feature_dim)
            else:
                f = self.encoder(inputs)
        else:
            f = inputs
        fproj = self.proj(f)
        mu_std = self.pi(fproj)
        mu,log_std = mu_std.chunk(2,dim=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(log_std) + 1)

        std = torch.exp(log_std)
        epsilon = torch.randn_like(mu)
        pi = epsilon*std + mu

        log_pi = (-0.5 * epsilon.pow(2) - log_std).sum(-1, keepdim=True) - 0.5 * np.log(2 * np.pi) * epsilon.size(-1)

        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)

        return mu,log_std,pi,log_pi,f

class QNetwork(nn.Module):
    def __init__(self,hidden_dim,projection_dim,action_shape):
        super().__init__()


        self.Qfunction = nn.Sequential(
            nn.Linear(projection_dim+action_shape,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

        self.init_weights()

    def init_weights(self):
        for i in range(5):
            if i %2 ==0:
                nn.init.orthogonal_(self.Qfunction[i].weight.data)
                self.Qfunction[i].bias.data.fill_(0.0)

    def forward(self,action,observation):
        assert (action.shape[0] == observation.shape[0]),"Unconsistent batch size"

        inputs = torch.cat([observation, action], dim=1)

        return self.Qfunction(inputs)

class Critic(nn.Module):
    def __init__(self,encoder,freeze_encoder,feature_dim,action_shape,hidden_dim,projection_dim):
        super().__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        self.proj = nn.Sequential(
            nn.Linear(5*feature_dim,projection_dim),
            nn.LayerNorm(projection_dim),
        )

        self.Q1 = QNetwork(hidden_dim,projection_dim,action_shape[0])
        self.Q2 = QNetwork(hidden_dim,projection_dim,action_shape[0])
        self.feature_dim = feature_dim

    def forward(self,inputs,actions, from_obs=True):
        if from_obs:
            if self.freeze_encoder:
                with torch.no_grad():
                    f = encode(self.encoder,inputs,self.feature_dim)
            else:
                f = self.encoder(inputs)
        else:
            f = inputs
        fproj = self.proj(f)
        q1 = self.Q1(actions,fproj)
        q2 = self.Q2(actions,fproj)
        return q1,q2

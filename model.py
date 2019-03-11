import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# DDPG Model
## Actor model
class DDPGActor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=(64,64)):
        
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units + (action_size,)
        model_list = []
        for i in range(len(fc_units)-1):
            model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))
            if i != len(fc_units) - 2:
                model_list.append(nn.LayerNorm(fc_units[i+1]))    # Layer Normalization to improve parameter Noise
        self.model = nn.ModuleList(model_list)


    def forward(self, state):                     # actor model, input is state
        x = state
        for i in range(len(self.fc_units)):
            x = F.relu(self.model[2*i](x))
            x = self.model[2*i+1](x)
        out = F.tanh(self.model[-1](x))  # Here we use tanh because continuous actions is between -1, 1
        return out                             # out put is action    

## Critic Model
class DDPGCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=(64,64)):
        super(DDPGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units + (1,)
        model_list = []
        for i in range(len(fc_units)-1):
            if i == 1:
                model_list.append(nn.Linear(fc_units[i]+action_size, fc_units[i+1]))
            else:     
                model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))        
        self.model = nn.ModuleList(model_list)    
    
    def forward(self, xs):
        x, a = xs                                    # critic model, input are state and action
        x = F.leaky_relu(self.model[0](x))
        x = F.leaky_relu(self.model[1](torch.cat([x,a],1)))
        for i in range(2, len(self.fc_units)):
            x = F.leaky_relu(self.model[i](x))
        out = self.model[-1](x)
        return out                                   # output is state value




## Approximator model
class ApproxActor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=(64,64)):
        
        super(ApproxActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units + (action_size,)
        model_list = []
        for i in range(len(fc_units)-1):
            model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))
        self.model = nn.ModuleList(model_list)
        
        self.std = nn.Parameter(torch.zeros(action_size))


    def forward(self, state, action=None):                     # actor model, input is state
        x = state
        for i in range(len(self.fc_units)):
            x = F.relu(self.model[i](x))
        mean = F.tanh(self.model[-1](x))  # Here we use tanh because continuous actions is between -1, 1
        if action is None:
            return mean
        
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        
        return log_prob, entropy                             # output is log_prob and entropy    

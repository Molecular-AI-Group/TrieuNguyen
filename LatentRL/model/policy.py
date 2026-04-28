import torch 
import torch.nn as nn 
from torch.distributions import Normal

class Policy(nn.Module):
    def __init__(self, dim=128, N=8, sigma=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dim))
        self.sigma = float(sigma)
        self.N = N

    def forward(self):
        dist = Normal(self.theta, self.sigma)
        a = dist.sample((self.N,))
        log_prob = dist.log_prob(a).sum(-1)
        entropy = dist.entropy().sum(-1)
        return a, log_prob, entropy


class DeeperPolicy(nn.Module):
    def __init__(self, dim=128, N=8, sigma=0.5, hidden_dims=[256, 256], learnable_sigma=False, min_sigma=0.01, max_sigma=1.0):
        super().__init__()
        self.N = N
        self.learnable_sigma = learnable_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        # Build MLP for mean prediction
        layers = []
        input_dim = dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, dim)
        
        if learnable_sigma:
            # Log-sigma head: outputs per-dimension log standard deviation
            self.log_sigma_head = nn.Linear(input_dim, dim)
            # Initialize so that initial sigma ≈ the provided sigma value
            nn.init.constant_(self.log_sigma_head.bias, torch.tensor(sigma).log().item())
            nn.init.zeros_(self.log_sigma_head.weight)
        else:
            self.sigma = float(sigma)
        
        self.base_input = nn.Parameter(torch.randn(dim))

    def forward(self):
        # Generate mean from learned input
        h = self.net(self.base_input)
        mean = self.mean_head(h)
        
        if self.learnable_sigma:
            # Clamp log_sigma for numerical stability, then exponentiate
            log_sigma = self.log_sigma_head(h)
            sigma = torch.exp(log_sigma).clamp(self.min_sigma, self.max_sigma)
        else:
            sigma = self.sigma
        
        dist = Normal(mean, sigma)
        a = dist.sample((self.N,))
        log_prob = dist.log_prob(a).sum(-1)
        entropy = dist.entropy().sum(-1)
        return a, log_prob, entropy
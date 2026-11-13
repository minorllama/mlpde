"""A periodic PINN trained on the Burgers residual by automatic differentiation."""
import math

import torch
from torch import nn


class BurgersPINN(nn.Module):
    def __init__(self, final_time, width=64):
        super().__init__()
        self.final_time = final_time
        self.network = nn.Sequential(
            nn.Linear(7, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, x, t):
        s = t / self.final_time
        harmonics = [f(k * x) for k in (1, 2, 3) for f in (torch.sin, torch.cos)]
        correction = self.network(torch.cat((2 * s - 1, *harmonics), dim=1))
        return -torch.sin(x) + s * correction


def derivative(value, coordinate):
    return torch.autograd.grad(value.sum(), coordinate, create_graph=True)[0]


def collocation(count, final_time, device):
    return (
        2 * math.pi * torch.rand(count, 1, device=device),
        final_time * torch.rand(count, 1, device=device),
    )


def physics_loss(model, x, t, nu):
    x = x.detach().requires_grad_()
    t = t.detach().requires_grad_()
    u = model(x, t)
    ux = derivative(u, x)
    ut = derivative(u, t)
    uxx = derivative(ux, x)
    return (ut + u * ux - nu * uxx).square().mean()


def train_pinn(nu, final_time, adam_steps, lbfgs_steps, device):
    model = BurgersPINN(final_time).to(device)
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(adam_steps):
        x, t = collocation(768, final_time, device)
        adam.zero_grad()
        loss = physics_loss(model, x, t, nu)
        loss.backward()
        adam.step()
        if (step + 1) % 500 == 0:
            print(f"PINN Adam {step + 1}: residual MSE {loss.item():.3g}", flush=True)

    if lbfgs_steps:
        x, t = collocation(4096, final_time, device)
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            loss = physics_loss(model, x, t, nu)
            loss.backward()
            return loss

        lbfgs.step(closure)
    return model.eval()

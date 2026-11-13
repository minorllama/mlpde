"""Conditional DDPM and a deterministic residual baseline with the same U-Net."""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def high_pass(signal, cutoff):
    modes = torch.fft.rfft(signal, dim=-1)
    mask = torch.arange(modes.shape[-1], device=signal.device) > cutoff
    return torch.fft.irfft(modes * mask, n=signal.shape[-1], dim=-1)


def periodic_conv(in_channels, out_channels):
    return nn.Conv1d(in_channels, out_channels, 3, padding=1, padding_mode="circular")


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_width):
        super().__init__()
        self.first = periodic_conv(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.time = nn.Linear(embedding_width, out_channels)
        self.second = periodic_conv(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, signal, embedding):
        hidden = self.norm1(self.first(signal)) + self.time(embedding)[..., None]
        hidden = self.norm2(self.second(F.silu(hidden)))
        return F.silu(hidden + self.skip(signal))


class ResidualUNet(nn.Module):
    def __init__(self, steps, width=24, embedding_width=64):
        super().__init__()
        self.steps = steps
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, 5, 16)))
        self.time = nn.Sequential(
            nn.Linear(32, embedding_width), nn.SiLU(),
            nn.Linear(embedding_width, embedding_width),
        )
        self.encoder = Block(2, width, embedding_width)
        self.middle = Block(width, 2 * width, embedding_width)
        self.decoder = Block(3 * width, width, embedding_width)
        self.output = periodic_conv(width, 1)

    def forward(self, noisy, low, step):
        phase = step[:, None] * self.frequencies / self.steps
        embedding = self.time(torch.cat((phase.sin(), phase.cos()), dim=1))
        fine = self.encoder(torch.stack((noisy, low), dim=1), embedding)
        coarse = self.middle(F.avg_pool1d(fine, 2), embedding)
        coarse = F.interpolate(coarse, size=fine.shape[-1], mode="nearest")
        return self.output(self.decoder(torch.cat((fine, coarse), dim=1), embedding))[:, 0]


class Schedule:
    def __init__(self, steps, device):
        self.steps = steps
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float64)
        curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        self.beta = (1 - curve[1:] / curve[:-1]).clamp(max=0.999).float()
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(dim=0)
        previous = F.pad(self.alpha_bar[:-1], (1, 0), value=1)
        self.posterior_variance = self.beta * (1 - previous) / (1 - self.alpha_bar)

    def corrupt(self, clean, noise, step):
        weight = self.alpha_bar[step, None]
        signal = weight.sqrt()
        sigma = (1 - weight).sqrt()
        noisy = signal * clean + sigma * noise
        velocity = signal * noise - sigma * clean
        return noisy, velocity

    @torch.no_grad()
    def sample(self, model, low, cutoff):
        """One independent residual per row of low, in normalized units."""
        state = high_pass(torch.randn_like(low), cutoff)
        for i in reversed(range(self.steps)):
            step = torch.full((low.shape[0],), i, device=low.device, dtype=torch.long)
            velocity = high_pass(model(state, low, step), cutoff)
            signal = self.alpha_bar[i].sqrt()
            sigma = (1 - self.alpha_bar[i]).sqrt()
            clean = signal * state - sigma * velocity
            previous = self.alpha_bar[i - 1] if i > 0 else 1.0
            clean_weight = previous**0.5 * self.beta[i] / (1 - self.alpha_bar[i])
            state_weight = self.alpha[i].sqrt() * (1 - previous) / (1 - self.alpha_bar[i])
            state = clean_weight * clean + state_weight * state
            if i > 0:
                innovation = high_pass(torch.randn_like(state), cutoff)
                state = state + self.posterior_variance[i].sqrt() * innovation
        return high_pass(state, cutoff)


def train_reconstructors(low, residual, whitened, cutoff, schedule, updates, batch_size=32):
    diffusion = ResidualUNet(schedule.steps).to(low.device)
    baseline = ResidualUNet(schedule.steps).to(low.device)
    ema = deepcopy(diffusion).eval().requires_grad_(False)
    diffusion_optimizer = torch.optim.AdamW(diffusion.parameters(), lr=7e-4)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=7e-4)
    for update in range(updates):
        index = torch.randint(len(low), (batch_size,), device=low.device)
        condition = low[index]
        target = residual[index]
        step = torch.randint(schedule.steps, (batch_size,), device=low.device)
        noise = high_pass(torch.randn_like(target), cutoff)
        noisy, velocity = schedule.corrupt(whitened[index], noise, step)

        diffusion_optimizer.zero_grad()
        prediction = high_pass(diffusion(noisy, condition, step), cutoff)
        diffusion_loss = F.mse_loss(prediction, velocity)
        diffusion_loss.backward()
        diffusion_optimizer.step()
        with torch.no_grad():
            decay = min(0.99, (update + 1) / (update + 10))
            for averaged, current in zip(ema.parameters(), diffusion.parameters()):
                averaged.lerp_(current, 1 - decay)

        baseline_optimizer.zero_grad()
        prediction = baseline(torch.zeros_like(target), condition, torch.zeros_like(step))
        baseline_loss = F.mse_loss(high_pass(prediction, cutoff), target)
        baseline_loss.backward()
        baseline_optimizer.step()

        if (update + 1) % 250 == 0:
            print(f"Reconstruction step {update + 1}: "
                  f"velocity MSE {diffusion_loss.item():.3g}, "
                  f"baseline MSE {baseline_loss.item():.3g}", flush=True)

    return ema, baseline.eval()

"""Cole–Hopf and a dealiased Fourier method for periodic viscous Burgers."""
import numpy as np

NU = 0.025
FINAL_TIME = 1.15
SNAPSHOT_TIME = 0.75
CUTOFF = 8
GRID = 256


def grid(n):
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def initial_data(coefficients, x):
    """Rows are sine and cosine amplitudes, starting at mode one."""
    modes = np.arange(1, coefficients.shape[-1] + 1)
    angles = modes[:, None] * x
    a, b = coefficients[..., 0, :], coefficients[..., 1, :]
    u0 = a @ np.sin(angles) + b @ np.cos(angles)
    primitive = (-a / modes) @ np.cos(angles) + (b / modes) @ np.sin(angles)
    return u0, primitive


def cole_hopf(coefficients, t, n=GRID, nu=NU, oversample=4):
    """Evaluate (G_t * (u0 phi0)) / (G_t * phi0) by periodic quadrature."""
    x = grid(n)
    if t == 0:
        return initial_data(coefficients, x)[0]
    if t < 0 or nu <= 0:
        raise ValueError("Require t >= 0 and nu > 0")

    # Resolve the heat kernel even when its width is small relative to the grid.
    oversample = max(oversample, int(np.ceil(2 * np.pi / (n * np.sqrt(nu * t)))))
    y = grid(n * oversample)
    u0, primitive = initial_data(coefficients, y)
    log_phi = -primitive / (2 * nu)
    phi = np.exp(log_phi - log_phi.max(axis=-1, keepdims=True))
    distance = (x[:, None] - y + np.pi) % (2 * np.pi) - np.pi
    # Include enough periodic images that the omitted Gaussian tails are tiny.
    images = max(1, int(np.ceil(np.sqrt(4 * nu * t * 40) / (2 * np.pi))))
    heat = np.zeros_like(distance)
    for image in range(-images, images + 1):
        heat += np.exp(-(distance + 2 * np.pi * image) ** 2 / (4 * nu * t))
    return ((u0 * phi) @ heat.T) / (phi @ heat.T)


def spectral(u0, times, nu=NU, max_dt=0.003):
    """Fourier Galerkin truncation with the strict 2/3 rule and RK4."""
    n = u0.size
    k = np.fft.rfftfreq(n, d=1 / n)
    keep = 3 * k < n
    viscous = -nu * k**2
    state = np.fft.rfft(u0) * keep

    def rhs(v):
        u = np.fft.irfft(v, n=n)
        flux = np.fft.rfft(0.5 * u**2)
        return viscous * v - 1j * k * keep * flux

    # Conservative explicit step bound for diffusion and advection.
    kmax = k[keep][-1]
    dt_limit = min(max_dt, 1.5 / (nu * kmax**2),
                   0.4 / (kmax * max(np.max(np.abs(u0)), 1e-12)))
    result = []
    previous = 0.0
    for t in times:
        if t < previous:
            raise ValueError("Output times must be sorted and nonnegative")
        steps = max(1, int(np.ceil((t - previous) / dt_limit)))
        dt = (t - previous) / steps
        for _ in range(steps):
            a = rhs(state)
            b = rhs(state + 0.5 * dt * a)
            c = rhs(state + 0.5 * dt * b)
            d = rhs(state + dt * c)
            state += (dt / 6) * (a + 2 * b + 2 * c + d)
        result.append(np.fft.irfft(state, n=n))
        previous = t
    return np.asarray(result)


def random_initial_data(rng, count):
    """Large scales and independent smaller scales create a conditional ensemble."""
    coefficients = np.zeros((count, 2, 12))
    phase = rng.uniform(0, 2 * np.pi, size=count)
    amplitude = rng.uniform(0.7, 1.3, size=count)
    coefficients[:, 0, 0] = amplitude * np.cos(phase)
    coefficients[:, 1, 0] = amplitude * np.sin(phase)
    coefficients[:, :, 1:8] = rng.normal(size=(count, 2, 7)) * (
        0.48 / np.arange(2, 9) ** 1.1
    )
    coefficients[:, :, 8:] = rng.normal(0, 0.16, size=(count, 2, 4))
    return coefficients

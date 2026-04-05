import numpy as np
import matplotlib.pyplot as plt

SPACE_SIZE = 100
TIME_STEPS = 1000
DX = 0.1
DT = 0.01 # Lower than limit_k
DIFFUSION_COEFFICIENT= 0.01 # Standard for slow invasion of tumor
PROLIFERATION_RATE = 0.3 # For visible growth, but not so aggressive
X = SPACE_SIZE // 2 # Double slash to prevent error of float index number
Y = SPACE_SIZE // 2


def calculate_diffusion(u, dx, d_coeff):
    # Formula for laplacian: (u_top + u_bottom + u_left + u_right - 4 * u) / dx^2
    u_top = u[1:-1, :-2]
    u_bottom = u[1:-1, 2:]
    u_left = u[:-2, 1:-1]
    u_right = u[2:, 1:-1]
    u = u[1:-1, 1:-1]
    laplacian = (u_top + u_bottom + u_left + u_right - 4 * u) / dx ** 2
    return laplacian * d_coeff

def calculate_reaction(u, rho):
    return u * rho * (1 - u)

# Stability check
limit_K = DX ** 2 / (4 * DIFFUSION_COEFFICIENT)
if DT >= limit_K:
    print("DT is too large, the simulation will be unstable")

u_old = np.zeros((SPACE_SIZE, SPACE_SIZE))
u_new = np.zeros((SPACE_SIZE, SPACE_SIZE)) #XXX DLACZEGO 2 NAWIASY?

# Initial condition (patient zero)
u_old[X, Y] = 0.1

# Simulation
for t in range(TIME_STEPS):
    diff = calculate_diffusion(u_old, DX, DIFFUSION_COEFFICIENT)
    react = calculate_reaction(u_old[1: -1, 1: -1], PROLIFERATION_RATE)

    u_new[1:-1, 1:-1] = u_old[1:-1, 1:-1] + DT * (diff + react)

    # Boundary conditions (Neumanna -> bouncing off the edge)
    u_new[0, :] = u_new[1, :] # Top boundary
    u_new[-1, :] = u_new[-2, :] # Bottom boundary
    u_new[:, 0] = u_new[:, 1] # Left boundary
    u_new[:, -1] = u_new[:, -2] # Right boundary

    # Updating u
    u_old[:] = u_new[:]

plt.figure(figsize=(8, 6))
plt.imshow(u_old, cmap='bone', origin='lower')
plt.colorbar(label='Concentration of cancer cells')
plt.title(f"Simulation after {TIME_STEPS} steps")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()




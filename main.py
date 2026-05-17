import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib


# --- constants ---
DX = 0.1
MAX_ITERATIONS = 20
SIMULATION_STEPS = 200
# final threshold to binarize the fuzzy simulation for sharp visualization
FINAL_VIS_THRESHOLD = 0.15


def calculate_diffusion(u: np.ndarray, dx: float, diff_map: np.ndarray) -> np.ndarray:
    """calculates laplacian using 5-point stencil and applies heterogeneous diffusion map."""
    # correct numpy indexing: axis 0 (rows/y), axis 1 (cols/x)
    u_up = u[:-2, 1:-1]
    u_down = u[2:, 1:-1]
    u_left = u[1:-1, :-2]
    u_right = u[1:-1, 2:]
    u_mid = u[1:-1, 1:-1]

    laplacian = (u_up + u_down + u_left + u_right - 4 * u_mid) / (dx ** 2)

    # apply brain tissue heterogenity
    return laplacian * diff_map[1:-1, 1:-1]


def calculate_reaction(u: np.ndarray, rho: float) -> np.ndarray:
    """calculates logistic growth."""
    return u * rho * (1 - u)


def calculate_dice(sim_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """calculates dice similarity coefficient between simulation and ground truth."""
    # low threshold for optimizer to 'see' early growth, high for gt
    sim_binary = (sim_mask > 0.05).astype(int)
    target_binary = (target_mask > 0.5).astype(int)

    intersection = np.logical_and(sim_binary, target_binary).sum()
    total_area = sim_binary.sum() + target_binary.sum()

    if total_area == 0:
        return 0.0
    return (2.0 * intersection) / total_area


def run_simulation(diff_map: np.ndarray, rho: float, dt: float, dx: float,
                   steps: int, start_x: int, start_y: int) -> np.ndarray:
    """runs the simulation without visualization (fast mode)."""
    # ensure correct data dimensions from nibabel (X, Y)
    size_x, size_y = diff_map.shape
    u_old = np.zeros((size_x, size_y))
    u_new = np.zeros((size_x, size_y))

    # patient zero at correct orientation
    u_old[start_x, start_y] = 1.0

    # finite difference time loop
    for _ in range(steps):
        u_core = u_old[1:-1, 1:-1]

        diff = calculate_diffusion(u_old, dx, diff_map)
        react = calculate_reaction(u_core, rho)

        u_new[1:-1, 1:-1] = u_core + dt * (diff + react)

        # neumann boundary conditions (reflecting edges)
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]

        u_old[:] = u_new[:]

    return u_old


def evaluate_cost(diff_map: np.ndarray, rho: float, dt: float, dx: float,
                  steps: int, start_x: int, start_y: int, target_mask: np.ndarray) -> tuple:
    """runs simulation and returns dice score along with the result matrix."""
    sim_result = run_simulation(diff_map, rho, dt, dx, steps, start_x, start_y)
    dice = calculate_dice(sim_result, target_mask)
    return dice, sim_result


def custom_hill_climbing(target_mask: np.ndarray, base_diff_map: np.ndarray,
                         dt: float, dx: float, steps: int, start_x: int, start_y: int) -> tuple:
    """custom adaptive hill climbing algorithm to find optimal d and rho."""
    current_d_mult = 1.0
    current_rho = 0.05
    step_d = 0.5
    step_rho = 0.02

    best_dice, best_sim = evaluate_cost(
        base_diff_map * current_d_mult, current_rho, dt, dx, steps, start_x, start_y, target_mask
    )
    print(f"start: d_mult={current_d_mult:.2f}, rho={current_rho:.3f} | dice: {best_dice:.4f}")

    for _ in range(MAX_ITERATIONS):
        # define 4 search directions (up, down, left, right)
        neighbors = [
            (current_d_mult + step_d, current_rho),
            (current_d_mult - step_d, current_rho),
            (current_d_mult, current_rho + step_rho),
            (current_d_mult, current_rho - step_rho)
        ]

        found_better = False

        for test_d, test_rho in neighbors:
            # security: prevent overflow and non-physical parameters
            if test_d <= 0.1 or test_rho <= 0.001 or test_d > 2.0:
                continue

            test_diff_map = base_diff_map * test_d
            score, sim_result = evaluate_cost(
                test_diff_map, test_rho, dt, dx, steps, start_x, start_y, target_mask
            )

            if score > best_dice:
                best_dice = score
                best_sim = sim_result
                current_d_mult = test_d
                current_rho = test_rho
                found_better = True
                print(f" -> step taken! d_mult={current_d_mult:.2f}, rho={current_rho:.3f} | dice: {best_dice:.4f}")

        if not found_better:
            step_d *= 0.5
            step_rho *= 0.5
            print(f" -> local peak. reducing step: step_d={step_d:.3f}, step_rho={step_rho:.4f}")

        if step_d < 0.05 and step_rho < 0.005:
            print("required precision reached. stopping optimization.")
            break

    return current_d_mult, current_rho, best_dice, best_sim


def create_crisp_tumor_mask(fuzzy_simulation: np.ndarray, threshold: float) -> np.ndarray:
    """binarizes the fuzzy simulation output to create a masked, crisp shape."""
    # creatingmasked array to show only tumor, making background transparent
    crisp_mask = (fuzzy_simulation > threshold).astype(float)
    return np.ma.masked_where(crisp_mask == 0, crisp_mask)


def main():
    """main program execution."""
    print("loading patient data (tumor mask)...")
    mask_path = '/Users/marysianowakowska/Downloads/TCIA_testdataset/testset_TCIA/UCSD-PTGBM-0004_01/UCSD-PTGBM-0004_01_BraTS_tumor_seg.nii.gz'
    img = nib.load(mask_path)
    data = img.get_fdata()

    sums = np.sum(data, axis=(0, 1))
    best_slice_idx = int(np.argmax(sums))
    print(f"analyzing slice: {best_slice_idx}")

    target_mask = data[:, :, best_slice_idx]
    target_mask = (target_mask > 0).astype(float)

    # calculating tumor centroid to locate 'patient zero'
    coords = np.argwhere(target_mask == 1)
    if len(coords) > 0:
        start_x, start_y = coords.mean(axis=0).astype(int)
    else:
        start_x, start_y = target_mask.shape[0] // 2, target_mask.shape[1] // 2

    # kontrolny print
    print(f"\n!!! START POINT CALCULATED AT: X={start_x}, Y={start_y} !!!\n")

    print("loading flair anatomy...")
    flair_path = '/Users/marysianowakowska/Downloads/TCIA_testdataset/testset_TCIA/UCSD-PTGBM-0004_01/UCSD-PTGBM-0004_01_FLAIR.nii.gz'
    flair_img = nib.load(flair_path)
    flair_data = flair_img.get_fdata()

    flair_slice = flair_data[:, :, best_slice_idx]

    flair_max = np.max(flair_slice)
    if flair_max > 0:
        flair_normalized = flair_slice / flair_max
    else:
        flair_normalized = flair_slice

    base_d = 0.005
    diff_map = base_d * (0.01 + 0.99 * flair_normalized)

    # skull barrier
    diff_map[flair_normalized < 0.1] = 0.0

    # safe time step calculation (CFL condition)
    max_d = np.max(diff_map)
    limit_k = DX ** 2 / (4 * max_d)
    dt = limit_k * 0.5
    print(f"safe dt calculated: {dt:.5f}")

    print("\nstarting custom intelligent optimization...")
    best_d_mult, best_rho, best_dice, best_sim = custom_hill_climbing(
        target_mask=target_mask,
        base_diff_map=diff_map,
        dt=dt,
        dx=DX,
        steps=SIMULATION_STEPS,
        start_x=start_x,
        start_y=start_y
    )

    print("\n--- optimization finished ---")
    print(f"optimal diffusion multiplier: {best_d_mult:.3f}")
    print(f"optimal proliferation (rho): {best_rho:.4f}")
    print(f"final dice score: {best_dice:.4f}")
    print("-----------------------------\n")

    # --- 5. VISUALIZATION PHASE ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- LEFT PLOT: Ground Truth on Flair Anatomy ---
    # Plot base flair brain first (using gray/bone cmap)
    axes[0].imshow(flair_normalized.T, cmap='bone', origin='lower')

    # Create masked ground truth to allow the brain background to show through
    # we make binary mask of 1s white, and 0s transparent
    gt_mask_display = np.ma.masked_where(target_mask == 0, target_mask)
    axes[0].imshow(gt_mask_display.T, cmap='bone', vmin=0, vmax=1, origin='lower', alpha=0.9)

    axes[0].set_title("ground truth (real tumor) on flair")
    # Mark the centroid
    axes[0].plot(start_x, start_y, 'ro', markersize=2)

    # --- RIGHT PLOT: Optimized Crisp Simulation on Flair Anatomy ---
    # Plot base flair brain as base layer
    axes[1].imshow(flair_normalized.T, cmap='bone', origin='lower')

    # Convert fuzzy simulated distribution to a crisp, binarized white mask
    # We use mcolors.ListedColormap to plot in crisp white
    crisp_sim_mask = create_crisp_tumor_mask(best_sim, FINAL_VIS_THRESHOLD)

    white_cmap = mcolors.ListedColormap(['white']) # only crisp white, no heat colors

    # Plot optimized binarized simulation in crisp white on top
    axes[1].imshow(crisp_sim_mask.T, cmap=white_cmap, origin='lower', alpha=0.8)
    axes[1].set_title("optimized crisp simulation on flair")

    plt.show()


if __name__ == "__main__":
    main()
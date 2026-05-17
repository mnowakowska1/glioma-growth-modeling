import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib

# --- constants ---
DX = 0.1
MAX_ITERATIONS = 20
SIMULATION_STEPS = 400
FINAL_VIS_THRESHOLD = 0.05
BASE_PATH = '/Users/marysianowakowska/Downloads/TCIA_testdataset/testset_TCIA/UCSD-PTGBM-0004_01/'

def load_normalized_slice(filename: str, slice_idx: int) -> np.ndarray:
    """loads a nifti file and normalizes its specific slice robustly to [0, 1]."""
    filepath = os.path.join(BASE_PATH, filename)
    img = nib.load(filepath)
    data = img.get_fdata()
    slice_data = data[:, :, slice_idx]
    p99 = np.percentile(slice_data, 99.5)
    if p99 > 0:
        normalized = slice_data / p99
        return np.clip(normalized, 0.0, 1.0)
    return slice_data

def calculate_diffusion_anisotropic(u: np.ndarray, dx: float, diff_map: np.ndarray,
                                    flair_slice: np.ndarray) -> np.ndarray:
    """calculates anisotropic diffusion guided by image structure (pseudo-dti)."""
    # 1. gradienty anatomii (bariery i drogi w mózgu)
    grad_y, grad_x = np.gradient(flair_slice, dx)
    # 2. gradienty stężenia nowotworu
    u_grad_y, u_grad_x = np.gradient(u, dx)
    # 3. waga krawędzi - hamuje dyfuzję w poprzek silnych struktur (np. naczyń, barier)
    edge_weight = 1.0 / (1.0 + (grad_x**2 + grad_y**2) * 50.0)

    flux_x = diff_map * edge_weight * u_grad_x
    flux_y = diff_map * edge_weight * u_grad_y

    # 4. dywergencja strumienia (anizotropowy laplasjan)
    _, d_flux_x = np.gradient(flux_x, dx)
    d_flux_y, _ = np.gradient(flux_y, dx)
    return d_flux_x + d_flux_y

def calculate_reaction(u: np.ndarray, rho_map: np.ndarray) -> np.ndarray:
    """calculates logistic growth using a heterogeneous proliferation map (asl)."""
    return u * rho_map * (1 - u)

def calculate_dice(sim_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """calculates dice similarity coefficient between simulation and target."""
    sim_binary = (sim_mask > 0.01).astype(int)
    target_binary = (target_mask > 0.5).astype(int)
    intersection = np.logical_and(sim_binary, target_binary).sum()
    total_area = sim_binary.sum() + target_binary.sum()
    return (2.0 * intersection) / total_area if total_area > 0 else 0.0

def run_simulation(base_diff_map: np.ndarray, base_rho_map: np.ndarray, u_initial: np.ndarray,
                   d_mult: float, rho_mult: float, dx: float, steps: int,
                   flair_slice: np.ndarray) -> np.ndarray:
    """runs the simulation using structure-guided anisotropic diffusion."""
    current_diff_map = base_diff_map * d_mult
    current_rho_map = base_rho_map * rho_mult
    max_d = np.max(current_diff_map)
    dt = (dx ** 2 / (4 * max_d)) * 0.4 if max_d > 0 else 0.01
    u_old = np.copy(u_initial)
    u_new = np.copy(u_initial)

    for _ in range(steps):
        u_core = u_old[1:-1, 1:-1]
        rho_core = current_rho_map[1:-1, 1:-1]
        # wywołanie dyfuzji sterowanej anatomią
        diff = calculate_diffusion_anisotropic(u_old, dx, current_diff_map, flair_slice)
        react = calculate_reaction(u_core, rho_core)
        u_new[1:-1, 1:-1] = u_core + dt * (diff[1:-1, 1:-1] + react)

        # warunki brzegowe (Neumann)
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]
        u_old[:] = u_new[:]
    return u_old

def evaluate_cost(base_diff_map: np.ndarray, base_rho_map: np.ndarray, u_initial: np.ndarray,
                  d_mult: float, rho_mult: float, dx: float, steps: int,
                  target_mask: np.ndarray, flair_slice: np.ndarray) -> tuple:
    """runs simulation and evaluates its accuracy."""
    sim_result = run_simulation(base_diff_map, base_rho_map, u_initial, d_mult, rho_mult, dx, steps, flair_slice)
    dice = calculate_dice(sim_result, target_mask)
    return dice, sim_result

def custom_hill_climbing(target_mask: np.ndarray, base_diff_map: np.ndarray, base_rho_map: np.ndarray,
                         u_initial: np.ndarray, dx: float, steps: int, start_d: float, start_rho: float,
                         flair_slice: np.ndarray) -> tuple: # <-- dodany flair_slice
    """adaptive hill climbing algorithm exploring 2d parameter space."""
    current_d_mult, current_rho_mult = start_d, start_rho
    step_d, step_rho = 0.25, 0.05

    best_dice, best_sim = evaluate_cost(
        base_diff_map, base_rho_map, u_initial, current_d_mult, current_rho_mult, dx, steps, target_mask, flair_slice
    )
    print(f"hill climbing start: d={current_d_mult:.2f}, rho={current_rho_mult:.2f} | dice: {best_dice:.4f}")

    for _ in range(MAX_ITERATIONS):
        neighbors = [(current_d_mult + step_d, current_rho_mult), (current_d_mult - step_d, current_rho_mult),
                     (current_d_mult, current_rho_mult + step_rho), (current_d_mult, current_rho_mult - step_rho)]
        found_better = False
        for test_d, test_rho in neighbors:
            if test_d <= 0.05 or test_rho <= 0.01 or test_d > 3.0: continue
            score, sim_result = evaluate_cost(
                base_diff_map, base_rho_map, u_initial, test_d, test_rho, dx, steps, target_mask, flair_slice
            )
            if score > best_dice:
                best_dice, best_sim = score, sim_result
                current_d_mult, current_rho_mult = test_d, test_rho
                found_better = True
                print(f" -> step: d={current_d_mult:.2f}, rho={current_rho_mult:.2f} | dice: {best_dice:.4f}")
        if not found_better:
            step_d *= 0.5; step_rho *= 0.5
            if step_d < 0.02 and step_rho < 0.01: break
    return current_d_mult, current_rho_mult, best_dice, best_sim

def create_crisp_tumor_mask(fuzzy_simulation: np.ndarray, threshold: float) -> np.ndarray:
    """binarizes the continuous simulation output for crisp visualization overlay."""
    crisp_mask = (fuzzy_simulation > threshold).astype(float)
    return np.ma.masked_where(crisp_mask == 0, crisp_mask)

def main():
    print("loading multimodal patient data...")
    gt_filepath = os.path.join(BASE_PATH, 'UCSD-PTGBM-0004_01_BraTS_tumor_seg.nii.gz')
    img_gt = nib.load(gt_filepath); data_gt = img_gt.get_fdata()
    best_slice_idx = int(np.argmax(np.sum(data_gt, axis=(0, 1))))
    print(f"analyzing optimal z-slice: {best_slice_idx}")

    target_mask = (data_gt[:, :, best_slice_idx] > 0).astype(float)
    flair_norm = load_normalized_slice('UCSD-PTGBM-0004_01_FLAIR.nii.gz', best_slice_idx)

    core_filepath = os.path.join(BASE_PATH, 'UCSD-PTGBM-0004_01_enhancing_cellular_tumor_seg.nii.gz')
    u_initial = (nib.load(core_filepath).get_fdata()[:, :, best_slice_idx] > 0).astype(float)

    adc_norm = load_normalized_slice('UCSD-PTGBM-0004_01_ADC_vendor.nii.gz', best_slice_idx)
    base_diff_map = 0.01 * adc_norm
    base_diff_map[flair_norm < 0.1] = 0.0
    asl_norm = load_normalized_slice('UCSD-PTGBM-0004_01_ASL_CBF.nii.gz', best_slice_idx)
    base_rho_map = 0.1 * asl_norm

    print("\n--- stage 1: coarse grid search ---")
    best_coarse_dice, best_start_d, best_start_rho = -1.0, 1.0, 1.0
    for test_d in [0.25, 0.5, 1.0]:
        for test_rho in [0.5, 1.5, 3.0]:
            score, _ = evaluate_cost(
                base_diff_map, base_rho_map, u_initial, test_d, test_rho, DX, SIMULATION_STEPS, target_mask, flair_norm
            )
            print(f"coarse: d={test_d}, rho={test_rho} -> dice: {score:.4f}")
            if score > best_coarse_dice:
                best_coarse_dice, best_start_d, best_start_rho = score, test_d, test_rho

    print("\n--- stage 2: fine hill climbing ---")
    best_d_mult, best_rho_mult, best_dice, best_sim = custom_hill_climbing(
        target_mask, base_diff_map, base_rho_map, u_initial, DX, SIMULATION_STEPS, best_start_d, best_start_rho, flair_norm
    )

    print(f"\nFINAL DICE: {best_dice:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    display_flair = flair_norm * 0.5

    axes[0].imshow(display_flair.T, cmap='gray', origin='lower')
    axes[0].imshow(np.ma.masked_where(target_mask == 0, target_mask).T, cmap='autumn', origin='lower', alpha=0.8)
    axes[0].set_title("ground truth")

    axes[1].imshow(display_flair.T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(u_initial == 0, u_initial).T, cmap='autumn', origin='lower', alpha=0.8)
    axes[1].set_title("initial core")

    axes[2].imshow(display_flair.T, cmap='gray', origin='lower')
    axes[2].imshow(create_crisp_tumor_mask(best_sim, FINAL_VIS_THRESHOLD).T, cmap='autumn', origin='lower', alpha=0.8)
    axes[2].set_title(f"result (dice: {best_dice:.2f})")

    for ax in axes: ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
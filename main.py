import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# # set backend
# matplotlib.use('TkAgg')

# --- configuration ---
patient_id = "0003_01"
base_dir = "/Users/marysianowakowska/Downloads/TCIA_testdataset/testset_TCIA/"

# --- simulation constants ---
dx = 0.1
max_iterations = 25
simulation_steps = 600
final_vis_threshold = 0.05

# paths setup
patient_folder = f"UCSD-PTGBM-{patient_id}"
base_path = os.path.join(base_dir, patient_folder)


def load_normalized_slice(filename, slice_idx):
    """loads nifti and normalizes slice to [0, 1]."""
    filepath = os.path.join(base_path, filename)
    img = nib.load(filepath)
    data = img.get_fdata()
    slice_data = data[:, :, slice_idx]
    p99 = np.percentile(slice_data, 99.5)
    return np.clip(slice_data / p99, 0.0, 1.0) if p99 > 0 else slice_data


def calculate_diffusion_anisotropic(u, dx, diff_map, flair):
    """
    manual implementation of anisotropic diffusion: div(d * grad(u)).
    we use fluxes at pixel edges to maintain correct dimensions.
    """
    k_const = 0.1  # stiffness of anatomical boundaries

    # x-flux (on vertical edges)
    grad_u_x = (u[1:-1, 1:] - u[1:-1, :-1]) / dx
    d_x = (diff_map[1:-1, 1:] + diff_map[1:-1, :-1]) / 2.0
    grad_f_x = (flair[1:-1, 1:] - flair[1:-1, :-1]) / dx
    weight_x = 1.0 / (1.0 + (grad_f_x / k_const)**2)
    flux_x = d_x * weight_x * grad_u_x

    # y-flux (on horizontal edges)
    grad_u_y = (u[1:, 1:-1] - u[:-1, 1:-1]) / dx
    d_y = (diff_map[1:, 1:-1] + diff_map[:-1, 1:-1]) / 2.0
    grad_f_y = (flair[1:, 1:-1] - flair[:-1, 1:-1]) / dx
    weight_y = 1.0 / (1.0 + (grad_f_y / k_const)**2)
    flux_y = d_y * weight_y * grad_u_y

    # divergence (net flow in central pixels)
    div_x = (flux_x[:, 1:] - flux_x[:, :-1]) / dx
    div_y = (flux_y[1:, :] - flux_y[:-1, :]) / dx

    laplacian = np.zeros_like(u)
    laplacian[1:-1, 1:-1] = div_x + div_y
    return laplacian


def calculate_dice(sim_mask, target_mask):
    """calculates dice coefficient score."""
    sim_binary = (sim_mask > 0.01).astype(int)
    target_binary = (target_mask > 0.5).astype(int)
    intersection = np.logical_and(sim_binary, target_binary).sum()
    total_area = sim_binary.sum() + target_binary.sum()
    return (2.0 * intersection) / total_area if total_area > 0 else 0.0


def run_simulation(base_diff_map, base_rho_map, u_initial,
                   d_mult, rho_mult, dx, steps, flair_slice, brain_mask):
    """explicit euler solver for fisher-kolmogorov equation."""
    current_d_map = base_diff_map * d_mult
    current_rho_map = base_rho_map * rho_mult

    # cfl stability check
    max_d = np.max(current_d_map)
    dt = (dx ** 2 / (4 * max_d)) * 0.4 if max_d > 0 else 0.01

    u_old = np.copy(u_initial)
    u_new = np.copy(u_initial)

    for _ in range(steps):
        u_core = u_old[1:-1, 1:-1]
        rho_core = current_rho_map[1:-1, 1:-1]

        diff = calculate_diffusion_anisotropic(u_old, dx, current_d_map, flair_slice)
        react = u_core * rho_core * (1 - u_core)

        u_new[1:-1, 1:-1] = u_core + dt * (diff[1:-1, 1:-1] + react)

        # neumann boundary conditions
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]

        u_new *= brain_mask
        u_old[:] = u_new[:]

    return u_old


def evaluate_cost(base_diff_map, base_rho_map, u_initial,
                  d_mult, rho_mult, dx, steps, target_mask,
                  flair_slice, brain_mask):
    """helper function to run and score a simulation."""
    sim_result = run_simulation(base_diff_map, base_rho_map, u_initial,
                                d_mult, rho_mult, dx, steps,
                                flair_slice, brain_mask)
    dice = calculate_dice(sim_result, target_mask)
    return dice, sim_result


def custom_hill_climbing(target_mask, base_diff_map, base_rho_map,
                         u_initial, dx, steps, start_d, start_rho,
                         flair_slice, brain_mask):
    """fine-tuning stage: local optimization via hill climbing."""
    curr_d, curr_rho = start_d, start_rho
    step_d, step_rho = 0.2, 0.1

    best_dice, best_sim = evaluate_cost(base_diff_map, base_rho_map, u_initial,
                                        curr_d, curr_rho, dx, steps,
                                        target_mask, flair_slice, brain_mask)

    for _ in range(max_iterations):
        neighbors = [(curr_d + step_d, curr_rho), (curr_d - step_d, curr_rho),
                     (curr_d, curr_rho + step_rho), (curr_d, curr_rho - step_rho)]

        found_better = False
        for td, tr in neighbors:
            if td <= 0.05 or tr <= 0.01 or td > 5.0: continue
            score, sim = evaluate_cost(base_diff_map, base_rho_map, u_initial,
                                       td, tr, dx, steps, target_mask,
                                       flair_slice, brain_mask)
            if score > best_dice:
                best_dice, best_sim = score, sim
                curr_d, curr_rho = td, tr
                found_better = True

        if not found_better:
            step_d *= 0.5
            step_rho *= 0.5
            if step_d < 0.01: break

    return curr_d, curr_rho, best_dice, best_sim


def main():
    """main workflow: coarse scan then fine hill climbing."""
    print(f"--- analyzing patient: {patient_id} ---")

    # 1. data loading
    core_file = f"{patient_folder}_enhancing_cellular_tumor_seg.nii.gz"
    core_3d = nib.load(os.path.join(base_path, core_file)).get_fdata()
    gt_file = f"{patient_folder}_BraTS_tumor_seg.nii.gz"
    gt_3d = nib.load(os.path.join(base_path, gt_file)).get_fdata()

    best_slice_idx = int(np.argmax(np.sum(gt_3d, axis=(0, 1))))
    target_mask = (gt_3d[:, :, best_slice_idx] > 0).astype(float)
    flair_norm = load_normalized_slice(f"{patient_folder}_FLAIR.nii.gz", best_slice_idx)
    brain_mask = (flair_norm > 0.12).astype(float)

    u_initial = (core_3d[:, :, best_slice_idx] > 0).astype(float)
    if np.sum(u_initial) == 0:
        coords = np.argwhere(target_mask > 0)
        if len(coords) > 0:
            c = coords.mean(axis=0).astype(int)
            u_initial[c[0]-2:c[0]+2, c[1]-2:c[1]+2] = 1.0

    brain_mask[u_initial > 0] = 1.0
    print(f"initial pixels: {int(np.sum(u_initial))}")

    # 2. physical maps
    adc_norm = load_normalized_slice(f"{patient_folder}_ADC_vendor.nii.gz", best_slice_idx)
    base_diff_map = 0.01 * (adc_norm ** 2) * brain_mask
    asl_norm = load_normalized_slice(f"{patient_folder}_ASL_CBF.nii.gz", best_slice_idx)
    base_rho_map = 0.1 * asl_norm

    # 3. stage 1: coarse grid search
    print("stage 1: coarse grid search...")
    best_coarse_dice = -1.0
    best_start_d, best_start_rho = 1.0, 1.5

    for td in [0.5, 1.5, 3.0]:
        for tr in [1.0, 2.5, 5.0]:
            score, _ = evaluate_cost(base_diff_map, base_rho_map, u_initial,
                                     td, tr, dx, simulation_steps, target_mask,
                                     flair_norm, brain_mask)
            if score > best_coarse_dice:
                best_coarse_dice = score
                best_start_d, best_start_rho = td, tr

    print(f"coarse search best dice: {best_coarse_dice:.4f}")

    # 4. stage 2: fine hill climbing
    print("stage 2: fine hill climbing...")
    best_d, best_rho, dice, best_sim = custom_hill_climbing(
        target_mask, base_diff_map, base_rho_map, u_initial,
        dx, simulation_steps, best_start_d, best_start_rho, flair_norm, brain_mask
    )

    print(f"\nfinal result. dice: {dice:.4f} | d: {best_d:.2f} | rho: {best_rho:.2f}")

    # 5. visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    fig.suptitle(f"patient no: {patient_id}", fontsize=16, fontweight='bold', y=0.93)

    bg = flair_norm * 0.5

    # plot 1: ground truth
    axes[0].imshow(bg.T, cmap='gray', origin='lower')
    axes[0].imshow(np.ma.masked_where(target_mask == 0, target_mask).T,
                   cmap='autumn', origin='lower', alpha=0.8)
    axes[0].set_title("target (ground truth)", pad=10)

    # plot 2: starting point
    axes[1].imshow(bg.T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(u_initial == 0, u_initial).T,
                   cmap='autumn', origin='lower', alpha=1.0)
    axes[1].set_title("seed (starting point)", pad=10)

    # plot 3: simulation result
    axes[2].imshow(bg.T, cmap='gray', origin='lower')
    res_mask = (best_sim > final_vis_threshold).astype(float)
    axes[2].imshow(np.ma.masked_where(res_mask == 0, res_mask).T,
                   cmap='autumn', origin='lower', alpha=0.8)
    axes[2].set_title(f"sim result (dice: {dice:.2f})", pad=10)

    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.2)

    plt.show()


if __name__ == "__main__":
    main()
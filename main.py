import os
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


# --- KONFIGURACJA ---
PATIENT_ID = "0015_01"
BASE_DIR = "/Users/marysianowakowska/Downloads/TCIA_testdataset/testset_TCIA/"

# --- PARAMETRY SYMULACJI ---
DX = 0.1
MAX_ITERATIONS = 20
SIMULATION_STEPS = 400
FINAL_VIS_THRESHOLD = 0.25

PATIENT_FOLDER = f"UCSD-PTGBM-{PATIENT_ID}"
BASE_PATH = os.path.join(BASE_DIR, PATIENT_FOLDER)


def load_normalized_slice(filename: str, slice_idx: int) -> np.ndarray:
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
    grad_y, grad_x = np.gradient(flair_slice, dx)
    u_grad_y, u_grad_x = np.gradient(u, dx)
    edge_weight = 1.0 / (1.0 + (grad_x**2 + grad_y**2) * 150.0)
    flux_x = diff_map * edge_weight * u_grad_x
    flux_y = diff_map * edge_weight * u_grad_y
    _, d_flux_x = np.gradient(flux_x, dx)
    d_flux_y, _ = np.gradient(flux_y, dx)
    return d_flux_x + d_flux_y


def calculate_dice(sim_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sim_binary = (sim_mask > 0.01).astype(int)
    target_binary = (target_mask > 0.5).astype(int)
    intersection = np.logical_and(sim_binary, target_binary).sum()
    total_area = sim_binary.sum() + target_binary.sum()
    return (2.0 * intersection) / total_area if total_area > 0 else 0.0


def run_simulation(base_diff_map: np.ndarray, base_rho_map: np.ndarray, u_initial: np.ndarray,
                   d_mult: float, rho_mult: float, dx: float, steps: int,
                   flair_slice: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    current_diff_map = base_diff_map * d_mult
    current_rho_map = base_rho_map * rho_mult
    max_d = np.max(current_diff_map)
    dt = (dx ** 2 / (4 * max_d)) * 0.4 if max_d > 0 else 0.01

    u_old = np.copy(u_initial)
    u_new = np.copy(u_initial)

    for _ in range(steps):
        u_core = u_old[1:-1, 1:-1]
        rho_core = current_rho_map[1:-1, 1:-1]
        diff = calculate_diffusion_anisotropic(u_old, dx, current_diff_map, flair_slice)
        react = u_core * rho_core * (1 - u_core)
        u_new[1:-1, 1:-1] = u_core + dt * (diff[1:-1, 1:-1] + react)
        u_new *= brain_mask
        u_old[:] = u_new[:]
    return u_old


def evaluate_cost(base_diff_map: np.ndarray, base_rho_map: np.ndarray, u_initial: np.ndarray,
                  d_mult: float, rho_mult: float, dx: float, steps: int,
                  target_mask: np.ndarray, flair_slice: np.ndarray, brain_mask: np.ndarray) -> tuple:
    sim_result = run_simulation(base_diff_map, base_rho_map, u_initial, d_mult, rho_mult, dx, steps, flair_slice, brain_mask)
    dice = calculate_dice(sim_result, target_mask)
    return dice, sim_result


def custom_hill_climbing(target_mask: np.ndarray, base_diff_map: np.ndarray, base_rho_map: np.ndarray,
                         u_initial: np.ndarray, dx: float, steps: int, start_d: float, start_rho: float,
                         flair_slice: np.ndarray, brain_mask: np.ndarray) -> tuple:
    current_d_mult, current_rho_mult = start_d, start_rho
    step_d, step_rho = 0.25, 0.05
    best_dice, best_sim = evaluate_cost(base_diff_map, base_rho_map, u_initial, current_d_mult, current_rho_mult, dx, steps, target_mask, flair_slice, brain_mask)

    for _ in range(MAX_ITERATIONS):
        neighbors = [(current_d_mult + step_d, current_rho_mult), (current_d_mult - step_d, current_rho_mult),
                     (current_d_mult, current_rho_mult + step_rho), (current_d_mult, current_rho_mult - step_rho)]
        found_better = False
        for test_d, test_rho in neighbors:
            if test_d <= 0.05 or test_rho <= 0.01 or test_d > 4.0: continue
            score, sim_result = evaluate_cost(base_diff_map, base_rho_map, u_initial, test_d, test_rho, dx, steps, target_mask, flair_slice, brain_mask)
            if score > best_dice:
                best_dice, best_sim = score, sim_result
                current_d_mult, current_rho_mult = test_d, test_rho
                found_better = True
        if not found_better:
            step_d *= 0.5; step_rho *= 0.5
            if step_d < 0.02 and step_rho < 0.01: break
    return current_d_mult, current_rho_mult, best_dice, best_sim


def main():
    print(f"--- ANALIZA MEDYCZNA PACJENTA: {PATIENT_ID} ---")

    # 1. SZUKANIE NAJLEPSZEGO PLASTRA
    core_file = f"{PATIENT_FOLDER}_enhancing_cellular_tumor_seg.nii.gz"
    data_core_3d = nib.load(os.path.join(BASE_PATH, core_file)).get_fdata()
    sums_core = np.sum(data_core_3d, axis=(0, 1))

    gt_file = f"{PATIENT_FOLDER}_BraTS_tumor_seg.nii.gz"
    data_gt_3d = nib.load(os.path.join(BASE_PATH, gt_file)).get_fdata()

    if np.max(sums_core) > 0:
        best_slice_idx = int(np.argmax(sums_core))
        print(f"Lekarz wskazał najsilniejszy rdzeń na plastrze: {best_slice_idx}")
    else:
        print("UWAGA: Brak zaznaczonego rdzenia. Szukam największego obszaru guza (BraTS).")
        best_slice_idx = int(np.argmax(np.sum(data_gt_3d, axis=(0, 1))))

    # 2. ŁADOWANIE DANYCH
    target_mask = (data_gt_3d[:, :, best_slice_idx] > 0).astype(float)
    flair_norm = load_normalized_slice(f"{PATIENT_FOLDER}_FLAIR.nii.gz", best_slice_idx)
    brain_mask = (flair_norm > 0.18).astype(float)

    # 3. GENEROWANIE NASIONKA (SEED) - POPRAWKA LOGICZNA
    # Jeśli rdzeń istnieje, weź go. Jeśli nie, weź geometryczny środek masy maski BraTS.
    u_initial = (data_core_3d[:, :, best_slice_idx] > 0).astype(float)

    if np.sum(u_initial) == 0:
        print("Generuję sztuczne nasionko startowe w centrum masy guza.")
        coords = np.argwhere(target_mask > 0)
        if len(coords) > 0:
            center = coords.mean(axis=0).astype(int)
            u_initial[center[0]-2:center[0]+2, center[1]-2:center[1]+2] = 1.0

    u_initial *= brain_mask
    print(f"Start symulacji z: {int(np.sum(u_initial))} aktywnych pikseli.")

    # 4. MAPY FIZYCZNE
    adc_norm = load_normalized_slice(f"{PATIENT_FOLDER}_ADC_vendor.nii.gz", best_slice_idx)
    base_diff_map = 0.01 * (adc_norm ** 2) * brain_mask
    asl_norm = load_normalized_slice(f"{PATIENT_FOLDER}_ASL_CBF.nii.gz", best_slice_idx)
    base_rho_map = 0.1 * asl_norm

    # 5. OPTYMALIZACJA
    print("Dopasowuję model...")
    best_coarse_dice, bsd, bsr = -1.0, 1.0, 1.0
    for td in [0.25, 0.5, 1.0]:
        for tr in [0.5, 1.5, 3.0]:
            s, _ = evaluate_cost(base_diff_map, base_rho_map, u_initial, td, tr, DX, SIMULATION_STEPS, target_mask, flair_norm, brain_mask)
            if s > best_coarse_dice: best_coarse_dice, bsd, bsr = s, td, tr

    best_d_mult, best_rho_mult, best_dice, best_sim = custom_hill_climbing(target_mask, base_diff_map, base_rho_map, u_initial, DX, SIMULATION_STEPS, bsd, bsr, flair_norm, brain_mask)

    # 6. WIZUALIZACJA
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    display_bg = flair_norm * 0.5
    axes[0].imshow(display_bg.T, cmap='gray', origin='lower')
    axes[0].imshow(np.ma.masked_where(target_mask == 0, target_mask).T, cmap='autumn', origin='lower', alpha=0.8)
    axes[0].set_title(f"Target - S:{best_slice_idx}")
    axes[1].imshow(display_bg.T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(u_initial == 0, u_initial).T, cmap='autumn', origin='lower', alpha=1.0)
    axes[1].set_title("Start Seed")
    axes[2].imshow(display_bg.T, cmap='gray', origin='lower')
    sim_mask_display = (best_sim > FINAL_VIS_THRESHOLD).astype(float)
    axes[2].imshow(np.ma.masked_where(sim_mask_display == 0, sim_mask_display).T, cmap='autumn', origin='lower', alpha=0.8)
    axes[2].set_title(f"Sim Result (Dice: {best_dice:.2f})")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
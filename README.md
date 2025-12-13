# CASM Visibility Simulation

This repository contains tools for simulating visibilities for the CASM (Coherent All Sky Monitor) phased array radio experiment.

## Code Overview

The main script `main.py` is a comprehensive simulation pipeline that:
1.  **Generates Array Layout**: Creates the antenna array configuration, 5 x 6 (NS x EW) grid over a 10m x 6m area by default.
2.  **Models the Sky**: Uses `pygdsm` to generate sky maps and also adds a model for the Sun.
3.  **Visualizes the Sky**: Produces sky maps with and without the primary beam pattern opacity mask.
4.  **Computes Visibilities**: Calculates noiseless visibilities using the Van Cittert-Zernike formalism ($V = \int I \cdot A \cdot e^{-2\pi i (ul+vm+wn)} d\Omega$).

## Usage 

### 1. Skymap only (Default)
By default, the script generates sky map images and the antenna layout but **does not** compute visibilities (which can be slow).

```bash
python main.py
```
*   **Outputs**:
    *   `skymaps/`: Directory containing sky map images.
    *   `casm_antenna_layout.png`: Plot of the array configuration.

### 2. Visibility Simulation
To compute visibilities for the specified number of frequency channels, use the `--compvis` flag and specify the number of channels using `--n-channels`.

```bash
python main.py --compvis --n-channels 100
```
*   **Outputs**:
    *   `casm_visibilities.npz`: Complete visibility data.

### 3. Test Baselines Mode
Useful for debugging or analyzing specific baselines. This mode restricts calculation to the maximum North-South and East-West baselines only.

```bash
python main.py --compvis --test-baselines --n-channels 100
```
*   **Features**:
    *   Computes visibilities only for the 2 selected baselines.
    *   Generates `casm_uv_coverage_test.png` highlighting these baselines in UV space.
*   **Outputs**:
    *   `casm_visibilities_test.npz`

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--compvis` | Enable visibility computation (default: False). | `False` |
| `--test-baselines` | Run in test mode (max NS/EW baselines only). | `False` |
| `--n-channels N` | Number of frequency channels to simulate. | `100` |
| `--time "YYYY-MM-DD..."` | Set specific observation time (ISO format). | `Now` |

## Outputs

### Sky Maps
The script generates sky maps at ~400 MHz to visualize the field of view:
*   **`..._nomask.png`**: The raw sky model (Global Sky Model + Sun).
*   **`..._masked.png`**: The sky model overlaid with a **red Gaussian opacity mask** representing the primary beam attenuation (Transparent at Zenith $\rightarrow$ Opaque at Horizon).

### Data Format (`.npz`)
The output `.npz` files contain:
*   `visibilities`: Complex visibility array `(n_baselines, n_freq, 2, 2)`. Units: Flux Density (Jy).
*   `uvw`: UVW coordinates in wavelengths `(n_baselines, n_freq, 3)`.
*   `baselines`: Physical baseline vectors in meters.
*   `baseline_pairs`: Antenna pair indices, shape `(n_baselines, 2)`.
*   `frequencies`: Frequency array in MHz.
*   `antenna_positions`: Antenna ENU coordinates.

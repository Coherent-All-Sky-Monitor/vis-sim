# CASM Visibility Simulation

This script simulates visibilities for the CASM (Coherent All Sky Monitor) array using the Van Cittert-Zernike formalism.

## Usage

### Time-Series Visibility Simulation

Generate a time-series of visibilities for target antennas (4, 6, 7, 8, 9, 10, 11, 12) from the CASM-13 layout:

```bash
python main.py --compvis --layout casm-13.csv --time-series --duration 50 --timestep 2 --n-channels 100 --time 2025-12-19T05:00:00
```

This creates a 50-hour simulation with 2-minute time steps, 100 frequency channels, starting at the specified UTC time.

### Command Line Arguments

| Argument | Description | Example/Default |
|----------|-------------|----------------|
| `--compvis` | Enable visibility computation | Required for visibilities |
| `--layout` | Antenna layout CSV file | `casm-13.csv` |
| `--time-series` | Run time-series simulation | Required |
| `--duration` | Duration in hours | `50` |
| `--timestep` | Time step in minutes | `2` |
| `--n-channels` | Number of frequency channels | `100` |
| `--time` | Start time (UTC ISO format) | `2025-12-19T05:00:00` |

## Timezone Handling

**Important**: Timezone handling can be confusing:

- **Input (`--time`)**: Specify in UTC ISO format (e.g., `2025-12-19T05:00:00`)
- **Output folder name**: Uses PST (Pacific Standard Time) for readability (e.g., `results_20251218_2100_to_20251221_0400`)
- **Stored timestamps (`mod_times` in NPZ)**: UTC ISO strings
- **Usage**: Convert `mod_times` to PST in analysis (see `example.ipynb`)

## Outputs

- **Directory**: `results_YYYYMMDD_HHMM_to_YYYYMMDD_HHMM/` (PST timestamps)
- **File**: `casm_visibilities_YYYYMMDD_HHMM_to_YYYYMMDD_HHMM.npz`

### NPZ Contents

- `visibilities`: Complex array `(n_times, n_baselines, n_freq, 2, 2)` - Flux density in Jy
- `times`: UTC ISO timestamp strings for each time step
- `baselines`, `baseline_pairs`, `frequencies`, `antenna_positions`: Metadata
- `source_names`, `source_alt`, `source_az`: Source positions per time step

## Example Usage

See `example.ipynb` for loading and analyzing the generated visibilities, including timezone conversion and comparison with observed data.

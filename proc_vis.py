import numpy as np
import glob
import re
import time
import os
import gc
from datetime import datetime

HEADER_SIZE = 4096


def has_header(filepath: str) -> bool:
    """Check if a visibility file has a 4096-byte ASCII header."""
    with open(filepath, "rb") as f:
        raw = f.read(HEADER_SIZE)
    text = raw.decode("ascii", errors="ignore")
    return "HDR_SIZE" in text

def parse_corr_header(filepath: str) -> dict[str, str]:
    """
    Read and parse the 4096-byte ASCII header from a visibility file.

    Parameters
    ----------
    filepath : str
        Path to a .dat visibility file with header.

    Returns
    -------
    dict
        Header key-value pairs as strings.
    """
    with open(filepath, "rb") as f:
        raw = f.read(HEADER_SIZE)
    text = raw.decode("ascii", errors="ignore")
    header = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            header[parts[0]] = parts[1].strip()
    return header

def get_header_offset(filepath: str) -> tuple[int, dict | None]:
    """
    Detect header presence and return offset + parsed header.

    Returns
    -------
    tuple of (int, dict or None)
        (4096, header_dict) if header present, (0, None) if not.
    """
    if has_header(filepath):
        return HEADER_SIZE, parse_corr_header(filepath)
    return 0, None

# --- Configuration ---
BASE_PATH = "/mnt/nvme3/data/casm/visibilities_64ant/"
BASE_STR = "2026-03-27-07:56:53"
FILE_PATTERN = BASE_STR + ".*"
OUTPUT_FILE = f"/home/casm/software/casm_xengine/scripts/advait/{BASE_STR}_vis_combined2.npy"
CHECK_INTERVAL = 20 * 60  # 20 minutes
MIN_SIZE_GB = 6.0         # Ignore files smaller than this
START_IDX = 0             # Only process files with index >= START_IDX
STOP_IDX = -1            # Process all files if -1

NSIG = 128
NCHAN = 3072
# IND = np.array([2,4,13,15,16,36,37,44,45,46,48,49])
IND = np.array([8,9,6,27,30,35,32,25,33,26,51,49,55,53,59,56])
NIND = len(IND)
NBASELINE = NSIG * (NSIG + 1) // 2

def reconstruct_vis_mat(nsig, data):
    triu_idxs = np.triu_indices(nsig)
    ntime = data.shape[0]
    nchan = data.shape[1]
    mat = np.zeros((ntime, nchan, nsig, nsig), dtype=np.complex64)
    for itime in range(ntime):
        for ichan in range(nchan):
            cplx_data = data[itime, ichan, :, 0] + 1j*data[itime, ichan, :, 1]
            vis = np.zeros((nsig, nsig), dtype=np.complex64)
            vis[triu_idxs] = cplx_data
            vis[np.diag_indices_from(vis)] /= 2
            mat[itime, ichan, :, :] = vis
            mat[itime, ichan, :, :] += np.conj(mat[itime, ichan, :, :].T)
    return mat

def get_file_size_gb(filepath):
    """ Returns file size in Gigabytes """
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024**3)
    return 0

# --- Main Execution ---
processed_files = set()
print(f"[{datetime.now()}] Watchdog started. Monitoring {BASE_PATH}")
print(f"Filtering for files >= {MIN_SIZE_GB} GB")

while True:
    all_files = glob.glob(os.path.join(BASE_PATH, FILE_PATTERN))
    def extract_index(p):
        m = re.search(r'\.(\d+)$', p)
        return int(m.group(1)) if m else -1
    
    current_files = sorted(all_files, key=extract_index)
    
    # Identify files that are both new and large enough and within index range
    new_files = [
        f for f in current_files 
        if f not in processed_files and get_file_size_gb(f) >= MIN_SIZE_GB and
        (extract_index(f) >= START_IDX) and (STOP_IDX == -1 or extract_index(f) <= STOP_IDX)
    ]
    
    if new_files:
        print(f"[{datetime.now()}] Processing {len(new_files)} new files...")
        newly_binned_chunks = []
        
        for fpath in new_files:
            try:
                # Load raw
                offset, file_header = get_header_offset(fpath)
                raw = np.fromfile(fpath, dtype=np.int32, offset=offset)
                ntime = raw.size // (NCHAN * NBASELINE * 2)
                xcorrs = raw.reshape(-1, NCHAN, NBASELINE, 2)
                
                # Reconstruct
                vis_mat = reconstruct_vis_mat(NSIG, xcorrs)
                del xcorrs 
                
                # Slice and Bin
                sub = vis_mat[:len(vis_mat), :, IND[:, None], IND[None, :]]
                del vis_mat
                
                # binned = sub.reshape(-1, 4, NCHAN, NIND, NIND).mean(1)
                newly_binned_chunks.append(sub.astype(np.complex64))
                
                processed_files.add(fpath)
                print(f"  + Done: {os.path.basename(fpath)} ({get_file_size_gb(fpath):.2f} GB)")
                
                gc.collect()
                
            except Exception as e:
                print(f"  !! Error on {fpath}: {e}")

        if newly_binned_chunks:
            fresh_combined = np.concatenate(newly_binned_chunks, axis=0)
            
            if os.path.exists(OUTPUT_FILE):
                existing_data = np.load(OUTPUT_FILE)
                final_output = np.concatenate([existing_data, fresh_combined], axis=0)
                del existing_data
            else:
                final_output = fresh_combined
            
            np.save(OUTPUT_FILE, final_output)
            
            size_gb = get_file_size_gb(OUTPUT_FILE)
            print(f"[{datetime.now()}] Saved {OUTPUT_FILE}")
            print(f"  - Current Combined File Size: {size_gb:.2f} GB")
            
            del final_output
            del fresh_combined
            del newly_binned_chunks
            gc.collect()
            
    else:
        # Check if there are files that exist but are too small
        pending = [f for f in current_files if f not in processed_files]
        if pending:
            print(f"[{datetime.now()}] {len(pending)} file(s) found but under {MIN_SIZE_GB} GB. Waiting..., size is {get_file_size_gb(pending[0]):.2f} GB")
        else:
            print(f"[{datetime.now()}] No new files. Sleeping...")

    time.sleep(CHECK_INTERVAL)
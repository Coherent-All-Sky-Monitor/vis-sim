#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from astropy.time import Time, TimeDelta
import datetime
plt.style.use('default')
plt.style.use(['science', 'notebook'])
import glob
import os
import re
import pytz
import pandas as pd
from astropy.stats import sigma_clip
from scipy.interpolate import griddata
import cupy as cp


# In[5]:


from main import generate_point_source_visibilities, generate_antenna_mapping, inject_point_source, calculate_uvw, calculate_baselines


# In[6]:


def generate_dirty_map_gpu_chunked(vis, uvw_lambda, res=256, chunk_size=2000):
    """
    GPU DFT imaging with pixel chunking to prevent OutOfMemoryErrors.
    """
    # 1. Setup grid on GPU
    l_axis = cp.linspace(-1, 1, res)
    L, M = cp.meshgrid(l_axis, l_axis)
    dist_sq = L**2 + M**2
    horizon_mask = dist_sq <= 1.0

    l_flat = L[horizon_mask].astype(cp.float32)
    m_flat = M[horizon_mask].astype(cp.float32)
    n_flat = cp.sqrt(1.0 - dist_sq[horizon_mask]).astype(cp.float32)

    # 2. Transfer data to GPU (Using float32/complex64)
    v_gpu = cp.asarray(vis, dtype=cp.complex64)
    uvw_gpu = cp.asarray(uvw_lambda, dtype=cp.float32)

    n_pixels = len(l_flat)
    n_baselines = v_gpu.shape[0]
    n_freqs = v_gpu.shape[1]

    # Final result array on GPU
    sky_brightness = cp.zeros(n_pixels, dtype=cp.complex64)

    # 3. Process in pixel chunks
    for i in range(0, n_pixels, chunk_size):
        end = min(i + chunk_size, n_pixels)
        l_chunk = l_flat[i:end]
        m_chunk = m_flat[i:end]
        n_chunk = n_flat[i:end]

        chunk_sum = cp.zeros(end - i, dtype=cp.complex64)

        # Loop over frequencies for this chunk of pixels
        for f_idx in range(n_freqs):
            u = uvw_gpu[:, f_idx, 0]
            v = uvw_gpu[:, f_idx, 1]
            w = uvw_gpu[:, f_idx, 2]
            v_chan = v_gpu[:, f_idx]

            # Phase calculation for the chunk: (chunk_size, n_baselines)
            # Using exp(j * theta) = cos(theta) + j*sin(theta) can sometimes be faster,
            # but cp.exp is generally well-optimized.
            phase_arg = 2.0 * cp.pi * (
                cp.outer(l_chunk, u) + 
                cp.outer(m_chunk, v) + 
                cp.outer(n_chunk - 1.0, w)
            )

            # Dot product: (chunk_size, baselines) @ (baselines,) -> (chunk_size,)
            chunk_sum += cp.dot(cp.exp(1j * phase_arg.astype(cp.float32)), v_chan)

        sky_brightness[i:end] = chunk_sum

        # Clear the cache occasionally if VRAM is tight
        cp.get_default_memory_pool().free_all_blocks()

    # 4. Final Reconstruction
    dirty_map = cp.zeros((res, res), dtype=cp.float32)
    dirty_map[horizon_mask] = cp.real(sky_brightness) / (n_baselines * n_freqs)

    return cp.asnumpy(dirty_map)


# In[7]:


frequencies = np.linspace(375, 500, 100)
antenna_mapping = generate_antenna_mapping('casm-60.csv')
time_obs = Time.now() - TimeDelta(0.25)


# In[8]:


vis = generate_point_source_visibilities(
    338.845,
    -8.755,
    1e5,
    frequencies,
    antenna_mapping,
    time_obs=time_obs,
    duration_s=1.0,
    Trec_k=50,
    a_eff_m2=0.2,
    beam_fwhm_deg=None,
    Tsky_eff_k=None,
)


# In[9]:


baselines_meters, baseline_pairs = calculate_baselines(
    antenna_mapping["positions"]
)
uvw_coords = calculate_uvw(baselines_meters, frequencies)


# In[14]:


vis = vis[...,0,0]


# In[16]:


dirty_map = generate_dirty_map_gpu_chunked(vis, uvw_coords, res=256)


# In[17]:


plt.figure(figsize=(10, 8))
plt.imshow(dirty_map, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
plt.colorbar(label='Flux Density (arbitrary units)')
plt.title('Wide-Field Dirty Map (DFT Method)')
plt.xlabel('l')
plt.ylabel('m')
plt.show()


# In[ ]:





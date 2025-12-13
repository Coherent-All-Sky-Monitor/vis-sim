"""
CASM Visibility Simulation
Generate simulated visibilities using pygdsm sky model in the CASM band (375-500 MHz)
"""

import numpy as np
import argparse
import os
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from scipy.interpolate import griddata
from astropy.time import Time
from pytz import timezone
from datetime import datetime
import pygdsm
import healpy as hp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# CASM Array Parameters
N_ROWS = 5   # North-South
N_COLS = 6   # East-West
N_ANTENNAS = N_ROWS * N_COLS  # 30 antennas
N_POL = 2    # Two polarizations per antenna
ARRAY_NS_LENGTH = 10.0  # meters (North-South)
ARRAY_EW_LENGTH = 6.0   # meters (East-West)
FREQ_MIN = 375.0  # MHz
FREQ_MAX = 468.0  # MHz (375 + 93 MHz bandwidth)
FREQ_CENTER = (FREQ_MIN + FREQ_MAX) / 2.0  # MHz

C_LIGHT = 299792458.0    # Speed of light in m/s
K_BOLTZMANN = 1.380649e-23 # Boltzmann constant in J/K

# OVRO location (approximate coordinates for Owens Valley Radio Observatory)
OVRO_LAT = 37.2339  # degrees
OVRO_LON = -118.2817  # degrees
OVRO_ELEV = 1222.0  # meters

# Primary beam solid angle (near zenith)
PRIMARY_BEAM_SOLID_ANGLE = 7500.0  # square degrees


# --- SUN MODEL PARAMETERS ---
SUN_BRIGHTNESS_TEMP = 8.0e5  # K (Approximation for Quiet Sun T_b)
SUN_DIAMETER_DEG = 1.2       # degrees (Approximation for size at 400 MHz)
SUN_SIGMA_DEG = SUN_DIAMETER_DEG / 2.355  # FWHM to Gaussian sigma: sigma = FWHM / (2*sqrt(2*ln(2)))
# -----------------------------


def generate_antenna_positions():
    """
    Generate spatial positions for the 43x6 grid of antennas.
    Returns positions in local ENU (East-North-Up) coordinates in meters.
    """
    # Create grid indices
    row_indices = np.arange(N_ROWS)  # 0 to N_ROWS-1
    col_indices = np.arange(N_COLS)  # 0 to N_COLS-1
    
    # Calculate spacing
    ns_spacing = ARRAY_NS_LENGTH / (N_ROWS - 1) if N_ROWS > 1 else 0
    ew_spacing = ARRAY_EW_LENGTH / (N_COLS - 1) if N_COLS > 1 else 0
    
    # Generate positions in local coordinates (East, North, Up)
    # East is x, North is y, Up is z
    positions = np.zeros((N_ANTENNAS, 3))
    
    idx = 0
    for row in row_indices:
        for col in col_indices:
            # Position relative to array center
            # East (x): col * spacing - center_offset
            # North (y): row * spacing - center_offset
            # Up (z): 0.24 m above ground screen
            positions[idx, 0] = col * ew_spacing - ARRAY_EW_LENGTH / 2.0  # East
            positions[idx, 1] = row * ns_spacing - ARRAY_NS_LENGTH / 2.0  # North
            positions[idx, 2] = 0.24  # Up (24 cm above ground screen)
            idx += 1
    
    return positions


def generate_antenna_mapping():
    """
    Create a mapping dictionary that stores antenna information.
    Returns a dictionary with antenna positions and metadata.
    """
    positions = generate_antenna_positions()
    
    mapping = {
        'positions': positions,  # Shape: (258, 3) in ENU coordinates (meters)
        'n_antennas': N_ANTENNAS,
        'n_pol': N_POL,
        'array_shape': (N_ROWS, N_COLS),
        'array_dimensions': (ARRAY_NS_LENGTH, ARRAY_EW_LENGTH),
        'antenna_ids': np.arange(N_ANTENNAS),
        'row_indices': np.repeat(np.arange(N_ROWS), N_COLS),
        'col_indices': np.tile(np.arange(N_COLS), N_ROWS),
    }
    
    return mapping


def get_local_coordinates(nside, time_obs):
    """
    Calculates the equatorial (ICRS) and local (AltAz, ENU) coordinates for all HEALPix pixels.
    This avoids redundant calculations if a sky model is added later.

    Returns:
    --------
    sky_coords_icrs : SkyCoord
        ICRS coordinates of all HEALPix pixels
    pixel_directions : array
        HEALPix pixel directions in ENU coordinates, shape (n_pixels, 3)
    pixel_altaz : array
        Alt-az coordinates for each pixel, shape (n_pixels, 2) [alt, az] in degrees
    """
    # Set up observatory location
    location = EarthLocation(
        lat=OVRO_LAT*u.deg,
        lon=OVRO_LON*u.deg,
        height=OVRO_ELEV*u.m
    )
    
    n_pixels = hp.nside2npix(nside)
    pixel_indices = np.arange(n_pixels)
    
    # Get pixel centers in galactic coordinates (lon, lat) in degrees
    theta, phi = hp.pix2ang(nside, pixel_indices, lonlat=False)
    
    # Convert to galactic coordinates (longitude, latitude)
    gal_lon = np.degrees(phi)  # Galactic longitude in degrees
    gal_lat = 90.0 - np.degrees(theta)  # Galactic latitude in degrees
    
    # Create SkyCoord objects in galactic frame
    sky_coords = SkyCoord(l=gal_lon*u.deg, b=gal_lat*u.deg, frame='galactic')
    
    # Transform to ICRS (equatorial) frame
    sky_coords_icrs = sky_coords.icrs
    
    # Transform to local alt-az coordinates at observation time
    altaz_frame = AltAz(obstime=time_obs, location=location)
    altaz_coords = sky_coords_icrs.transform_to(altaz_frame)
    
    # Extract alt and az in degrees
    alt = altaz_coords.alt.deg
    az = altaz_coords.az.deg
    
    # Store alt-az coordinates
    pixel_altaz = np.column_stack([alt, az])
    
    # Convert alt-az to ENU unit vectors
    alt_rad = np.radians(alt)
    az_rad = np.radians(az)
    
    pixel_directions = np.zeros((n_pixels, 3))
    pixel_directions[:, 0] = np.cos(alt_rad) * np.sin(az_rad)  # East
    pixel_directions[:, 1] = np.cos(alt_rad) * np.cos(az_rad)  # North
    pixel_directions[:, 2] = np.sin(alt_rad)                    # Up
    
    return sky_coords_icrs, pixel_directions, pixel_altaz

def convert_tb_to_intensity(T_b_map, frequency_hz):
    """
    Converts Brightness Temperature (K) to Specific Intensity (W/m²/Hz/sr) 
    using the Rayleigh-Jeans approximation.
    
    Parameters:
    -----------
    T_b_map : array
        Brightness temperature in Kelvin.
    frequency_hz : float
        Frequency in Hz.
        
    Returns:
    --------
    I_map : array
        Specific Intensity in W/m²/Hz/sr.
    """
    # I = (2 * k_B * T_b * nu^2) / c^2
    nu = frequency_hz
    I_map = (2 * K_BOLTZMANN * T_b_map * nu**2) / (C_LIGHT**2)
    return I_map


def add_sun_to_sky_model(sky_maps, frequencies, time_obs, sky_coords_icrs, nside=64):
    """
    Adds a Gaussian-profile Sun source to the sky map.

    Parameters:
    -----------
    sky_maps : array
        Sky brightness temperature map in K, shape (n_freq, n_pixels)
    frequencies : array-like
        Frequencies in MHz (currently unused for T_b, but kept for future spectral index)
    time_obs : Time
        Observation time
    sky_coords_icrs : SkyCoord
        ICRS coordinates of all HEALPix pixels
    nside : int
        HEALPix nside parameter

    Returns:
    --------
    sky_maps_sun : array
        Augmented sky brightness temperature map in K, shape (n_freq, n_pixels)
    """
    n_freq, n_pixels = sky_maps.shape
    
    # 1. Get Sun's position (ICRS)
    sun_coord = get_sun(time_obs)
    
    # 2. Calculate angular distance between Sun and every HEALPix pixel
    # Use astropy's separation function on the ICRS coordinates
    separations = sun_coord.separation(sky_coords_icrs).deg
    
    # 3. Apply Gaussian Brightness Profile
    # The Sun's brightness temperature (T_b) is assumed constant across the band
    Tb_sun_peak = SUN_BRIGHTNESS_TEMP  # K
    sigma_sun = SUN_SIGMA_DEG  # degrees
    
    # Gaussian profile: T_b(theta) = T_peak * exp(-0.5 * (theta / sigma)^2)
    sun_profile_tb = Tb_sun_peak * np.exp(-0.5 * (separations / sigma_sun)**2)
    
    # 4. Add the Sun's profile to the sky maps
    # We assume the Sun is much brighter than the background, and simply add the T_b.
    # We also assume the Sun's brightness is constant across the CASM band (flat spectrum).
    
    # Create the Sun's map (same profile for all frequencies)
    sun_map = np.tile(sun_profile_tb, (n_freq, 1))
    
    # Add to the existing sky maps
    sky_maps_sun = sky_maps + sun_map
    
    # Note: If two bright sources overlap, the brightness temperatures should generally be added.
    print(f"   Added Quiet Sun model (T_peak={Tb_sun_peak:.2e} K, Diameter={SUN_DIAMETER_DEG}°)")

    return sky_maps_sun


def get_sky_model(frequencies, time_obs=None, nside=64):
    """
    Generate sky model using pygdsm, add the Sun, and transform to local coordinates.
    
    Parameters:
    -----------
    frequencies : array-like
        Frequencies in MHz
    time_obs : Time
        Observation time for coordinate transformation
    nside : int
        HEALPix nside parameter (higher = finer resolution)
    
    Returns:
    --------
    sky_map : array
        Sky brightness temperature map in K, shape (n_freq, n_pixels)
    pixel_directions : array
        HEALPix pixel directions in ENU coordinates, shape (n_pixels, 3)
    pixel_altaz : array
        Alt-az coordinates for each pixel, shape (n_pixels, 2) [alt, az] in degrees
    """
    if time_obs is None:
        time_obs = Time.now()
    
    # 1. Get all coordinate transformations (ICRS, ENU, AltAz)
    sky_coords_icrs, pixel_directions, pixel_altaz = get_local_coordinates(nside, time_obs)

    # 2. Generate pygdsm background sky model
    # Initialize pygdsm
    gsm = pygdsm.GlobalSkyModel16(freq_unit='MHz', data_unit='TCMB')
    
    # Generate sky model for each frequency
    sky_maps = []
    for freq in frequencies:
        sky_map = gsm.generate(freq)
        # Get the nside of the generated map
        map_nside = hp.get_nside(sky_map)
        
        # Resample to desired nside if needed
        if map_nside != nside:
            sky_map = hp.ud_grade(sky_map, nside)
        
        sky_maps.append(sky_map)
    
    # Stack into array: (n_freq, n_pixels)
    sky_maps = np.array(sky_maps)
    
    # 3. Add the Sun to the sky model
    sky_maps_sun = add_sun_to_sky_model(sky_maps, frequencies, time_obs, sky_coords_icrs, nside=nside)
    
    # Return the augmented map and coordinates
    return sky_maps_sun, pixel_directions, pixel_altaz


def calculate_baselines(antenna_positions):
    """
    Calculate all baseline vectors between antenna pairs.
    
    Parameters:
    -----------
    antenna_positions : array
        Shape (n_antennas, 3) in meters
    
    Returns:
    --------
    baselines : array
        Shape (n_baselines, 3) baseline vectors in meters
    baseline_pairs : array
        Shape (n_baselines, 2) antenna pair indices
    """
    n_ant = len(antenna_positions)
    baselines = []
    baseline_pairs = []
    
    for i in range(n_ant):
        for j in range(i, n_ant):  # Include auto-correlations
            baseline = antenna_positions[j] - antenna_positions[i]
            baselines.append(baseline)
            baseline_pairs.append([i, j])
    
    baselines = np.array(baselines)
    baseline_pairs = np.array(baseline_pairs)
    
    return baselines, baseline_pairs


def calculate_uvw(baselines_meters, frequencies):
    """
    Convert baselines from meters (ENU) to wavelengths (UVW) for each frequency.
    For a zenith-phased array, the coordinate systems align:
    u -> East (wavelengths)
    v -> North (wavelengths)
    w -> Up (wavelengths) toward phase center (Zenith)
    
    Parameters:
    -----------
    baselines_meters : array
        Baseline vectors in meters, shape (n_baselines, 3)
    frequencies : array
        Frequencies in MHz
        
    Returns:
    --------
    uvw : array
        UVW coordinates, shape (n_baselines, n_freq, 3)
    """
    c = 299792458.0
    lambdas = c / (frequencies * 1e6)  # Wavelengths in meters
    
    # Broadcast division: (B, 1, 3) / (1, F, 1) -> (B, F, 3)
    uvw = baselines_meters[:, np.newaxis, :] / lambdas[np.newaxis, :, np.newaxis]
    
    return uvw


def get_primary_beam_attenuation(pixel_directions):
    """
    Calculate primary beam attenuation P(theta) explicitly.
    Assumes a Gaussian beam centered at zenith with solid angle PRIMARY_BEAM_SOLID_ANGLE.
    
    Parameters:
    -----------
    pixel_directions : array
        Unit vectors pointing to pixels in ENU, shape (n_pixels, 3)
        (l, m, n) where n is toward Zenith.
        
    Returns:
    --------
    attenuation : array
        Power attenuation factor (0 to 1) for each pixel.
    """
    # Calculate theta (angle from zenith)
    # n = cos(theta) -> theta = arccos(n)
    # pixel_directions[:, 2] is the 'Up' component (n)
    n_component = pixel_directions[:, 2]
    # Clip to avoid numerical errors slightly outside [-1, 1]
    n_component = np.clip(n_component, -1.0, 1.0)
    theta = np.arccos(n_component)  # radians
    
    # Calculate sigma from solid angle
    # Omega = 2 * pi * sigma^2  => sigma = sqrt(Omega / 2pi)
    solid_angle_sr = PRIMARY_BEAM_SOLID_ANGLE * (np.pi / 180.0)**2
    sigma = np.sqrt(solid_angle_sr / (2 * np.pi))
    
    # Gaussian profile: P(theta) = exp(-theta^2 / (2*sigma^2))
    attenuation = np.exp(- (theta**2) / (2 * sigma**2))
    
    # Set attenuation to 0 for sources below horizon (n < 0)
    attenuation[n_component <= 0] = 0.0
    
    return attenuation


def plot_uv_coverage(uvw, highlight_indices=None, freq_idx=0, save_path=None):
    """
    Plot UV coverage for a specific frequency channel.
    
    Parameters:
    -----------
    uvw : array
        UVW coordinates, shape (n_baselines, n_freq, 3)
    highlight_indices : list
        List of baseline indices to highlight
    freq_idx : int
        Frequency index to plot
    save_path : str
        Path to save plot
    """
    # Extract u, v for the frequency channel (in wavelengths)
    u_vals = uvw[:, freq_idx, 0]
    v_vals = uvw[:, freq_idx, 1]
    
    # Conjugate points (-u, -v) are symmetric
    u_all = np.concatenate([u_vals, -u_vals])
    v_all = np.concatenate([v_vals, -v_vals])
    
    plt.figure(figsize=(8, 8))
    
    # Plot all points
    plt.scatter(u_all, v_all, s=5, c='gray', alpha=0.5, label='All Baselines')
    
    # Plot highlighted points
    if highlight_indices is not None:
        u_high = u_vals[highlight_indices]
        v_high = v_vals[highlight_indices]
        
        # Add symmetric points
        u_high_all = np.concatenate([u_high, -u_high])
        v_high_all = np.concatenate([v_high, -v_high])
        
        plt.scatter(u_high_all, v_high_all, s=100, c='red', marker='*', label='Test Baselines')
    
    plt.xlabel('u (wavelengths)')
    plt.ylabel('v (wavelengths)')
    plt.title(f'UV Coverage')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved UV coverage plot to {save_path}")
    else:
        plt.show()
    plt.close()


def generate_visibilities(antenna_mapping, frequencies, time_obs=None, n_sky_pixels=None, 
                         custom_baselines=None, custom_baseline_pairs=None):
    """
    Generate simulated visibilities for the CASM array using Van Cittert-Zernike formalism.
    
    Parameters:
    -----------
    antenna_mapping : dict
        Antenna position mapping dictionary
    frequencies : array
        Frequencies in MHz
    time_obs : Time
        Observation time (default: current time)
    n_sky_pixels : int
        (Deprecated) Number of sky pixels is determined by nside within get_sky_model
    custom_baselines : array (optional)
        Pre-calculated or subset of baselines (meters)
    custom_baseline_pairs : array (optional)
        Corresponding antenna pairs
    
    Returns:
    --------
    visibilities : dict
        Dictionary containing visibility data
    """
    if time_obs is None:
        time_obs = Time.now()
    
    positions = antenna_mapping['positions']
    n_ant = antenna_mapping['n_antennas']
    n_freq = len(frequencies)
    
    if custom_baselines is not None and custom_baseline_pairs is not None:
        baselines_meters = custom_baselines
        baseline_pairs = custom_baseline_pairs
    else:
        # Calculate baselines in meters
        baselines_meters, baseline_pairs = calculate_baselines(positions)
    
    n_baselines = len(baselines_meters)
    
    # Calculate UVW coordinates [n_baselines, n_freq, 3]
    # For zenith pointing: u=East, v=North, w=Up
    uvw = calculate_uvw(baselines_meters, frequencies)
    
    # Get sky model
    print(f"Generating sky model for {n_freq} frequencies...")
    # Use nside=64 for ~49,000 pixels (resolution ~1 deg)
    sky_maps, pixel_directions, pixel_altaz = get_sky_model(frequencies, time_obs=time_obs, nside=64)
    n_pixels = sky_maps.shape[1]
    
    # Calculate direction cosines (l, m, n)
    # pixel_directions is (n_pixels, 3) -> (East, North, Up) -> (l, m, n)
    # l points East, m points North, n points Up (toward phase center/zenith)
    lmn = pixel_directions  # Shape (n_pixels, 3)
    
    # Identify visible pixels (horizon check)
    # n = lmn[:, 2] is the Up component
    visible_mask = lmn[:, 2] > 0
    
    # Apply primary beam attenuation
    print("Applying Gaussian primary beam...")
    beam_attenuation = get_primary_beam_attenuation(lmn)
    
    # Combine selection mask: visible AND significant beam support
    # (e.g., beam > 1e-6 to avoid unnecessary computation for nearly zero contribution)
    selection_mask = visible_mask & (beam_attenuation > 1e-6)
    
    selected_indices = np.where(selection_mask)[0]
    n_selected = len(selected_indices)
    
    print(f"   Integrating over {n_selected} pixels (visible & within beam)")
    
    # Filter data for selected pixels
    lmn_selected = lmn[selected_indices]  # (n_selected, 3)
    beam_value_selected = beam_attenuation[selected_indices] # (n_selected,)
    sky_maps_selected = sky_maps[:, selected_indices]  # (n_freq, n_selected)
    
    # Calculate pixel solid angle (dOmega)
    nside = hp.npix2nside(n_pixels)
    pixel_area = hp.nside2pixarea(nside)  # steradians
    
    # Apparent Sky Brightness: I_app(l,m) = I_sky(l,m) * A(l,m)
    # Shape: (n_freq, n_selected)
    apparent_sky = sky_maps_selected * beam_value_selected[np.newaxis, :]
    
    # Initialize visibility array
    # Shape: (n_baselines, n_freq, N_POL, N_POL)
    visibilities = np.zeros((n_baselines, n_freq, N_POL, N_POL), dtype=complex)
    
    print(f"Computing visibilities for {n_baselines} baselines...")

    # Computation Strategy:
    # V(u,v,w) = Sum [ I_app(l,m) * exp(-2pi * i * (ul + vm + wn)) * dOmega ]
    # term (ul + vm + wn) is the dot product of UVW vector and LMN vector.
    
    # We can vectorize over pixels for each frequency-baseline chunk.
    # UVW: (n_baselines, n_freq, 3)
    # LMN: (n_selected, 3)
    
    # To save memory, we invoke loop over frequencies or baselines
    
    for f_idx in range(n_freq):
        freq_hz = frequencies[f_idx] * 1e6 # Convert MHz to Hz
        
        # --- FIX: CONVERT T_B to I_app (Specific Intensity) ---
        # I_app is the apparent intensity I * A (in W/m²/Hz/sr)
        T_b_apparent = apparent_sky[f_idx] # K
        I_apparent = convert_tb_to_intensity(T_b_apparent, freq_hz) # W/m²/Hz/sr
        # -----------------------------------------------------

        # I_app_term is the quantity to be summed: I_apparent * dOmega 
        # Unit: (W/m²/Hz/sr) * (sr) = W/m²/Hz (Flux Density)
        I_app_term = I_apparent * pixel_area
        
        # Get UVW for this freq: shape (n_baselines, 3)
        uvw_f = uvw[:, f_idx, :]
        
        # Calculate phase term: dot(UVW, LMN.T) -> shape (n_baselines, n_selected)
        # argument = ul + vm + wn
        # This matrix multiply can be large: n_bl(3600?) * n_pix(20000?) ~ 72e6 complex64 ~ 500MB. OK.
        phase_arg = np.dot(uvw_f, lmn_selected.T)
        
        # Exponential term
        phasor = np.exp(-2j * np.pi * phase_arg)
        
        # Summation: Vis = dot(phasor, I_app_term)
        # (n_baselines, n_selected) dot (n_selected,) -> (n_baselines,)
        vis_val = np.dot(phasor, I_app_term) * 1e26 # Flux density in Jy
        
        # Assign to XX and YY pols
        visibilities[:, f_idx, 0, 0] = vis_val
        visibilities[:, f_idx, 1, 1] = vis_val
        
        if f_idx % 10 == 0:
            print(f"   Processed frequency channel {f_idx}/{n_freq}")
            
    result = {
        'visibilities': visibilities,
        'baselines': baselines_meters,
        'uvw': uvw,
        'baseline_pairs': baseline_pairs,
        'frequencies': frequencies,
        'time_obs': time_obs,
        'n_antennas': n_ant,
        'n_baselines': n_baselines,
        'n_freq': n_freq,
        'pixel_altaz': pixel_altaz,
    }
    
    return result


def plot_antenna_layout(antenna_mapping, save_path=None):
    """Plot the antenna array layout."""
    positions = antenna_mapping['positions']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Top view (East-North)
    ax1.scatter(positions[:, 0], positions[:, 1], s=50, alpha=0.6)
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('CASM Array Layout (Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Side view (North-Up)
    ax2.scatter(positions[:, 1], positions[:, 2], s=50, alpha=0.6)
    ax2.set_xlabel('North (m)')
    ax2.set_ylabel('Up (m)')
    ax2.set_title('CASM Array Layout (Side View)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved antenna layout plot to {save_path}")
    else:
        plt.show()


def plot_sky_map_with_beam(sky_maps, pixel_altaz, frequencies, time_obs, output_dir='skymaps'):
    """
    Plot sky maps for each frequency. Generates two versions:
    1. Unmasked (full sky)
    2. Masked (with red Gaussian opacity overlay)
    
    Parameters:
    -----------
    sky_maps : array
        Sky brightness maps, shape (n_freq, n_pixels)
    pixel_altaz : array
        Alt-az coordinates, shape (n_pixels, 2)
    frequencies : array
        Frequencies in MHz
    time_obs : Time
        Observation time
    output_dir : str
        Directory to save sky map images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get pixel indices and nside
    n_pixels = len(sky_maps[0])
    
    # Extract alt and az
    alt = pixel_altaz[:, 0]  # Altitude in degrees
    az = pixel_altaz[:, 1]    # Azimuth in degrees
    
    # Filter to only visible hemisphere (alt > 0)
    visible_mask = alt > 0
    visible_alt = alt[visible_mask]
    visible_az = az[visible_mask]
    
    # Set up observatory location for source positions
    location = EarthLocation(
        lat=OVRO_LAT*u.deg,
        lon=OVRO_LON*u.deg,
        height=OVRO_ELEV*u.m
    )
    
    # Define bright sources
    bright_sources = {
        'Cas A': SkyCoord(ra=350.85*u.deg, dec=58.81*u.deg, frame='icrs'),
        'Cyg A': SkyCoord(ra=299.87*u.deg, dec=40.73*u.deg, frame='icrs'),
        'Tau A': SkyCoord(ra=83.63*u.deg, dec=22.01*u.deg, frame='icrs')
    }
    # Add position of the sun
    sun_coord = get_sun(time_obs)
    bright_sources['Sun'] = sun_coord
    
    # Get time string for filenames
    pst = timezone('America/Los_Angeles')
    time_pst = time_obs.to_datetime(timezone=pst)
    time_str = time_pst.strftime('%Y%m%d_%H%M%S')
    
    # Process each frequency separately
    for f_idx, freq in enumerate(frequencies):
        # Get sky map for this frequency
        sky_map = sky_maps[f_idx]
        visible_sky = sky_map[visible_mask]
        
        # Create regular grid in alt-az space for smooth interpolation
        az_grid = np.linspace(0, 360, 360)
        alt_grid = np.linspace(0, 90, 90)
        AZ, ALT = np.meshgrid(az_grid, alt_grid)
        
        # Interpolate sky map onto regular grid
        points = np.column_stack([visible_az, visible_alt])
        sky_interp = griddata(points, visible_sky, (AZ, ALT), method='linear', fill_value=np.nan)
        
        # Convert to polar coordinates for plotting
        az_plot_deg = AZ
        az_plot_rad = np.radians(az_plot_deg)
        radius = 90.0 - ALT
        
        # Generate two plots: one unmasked, one masked
        for use_mask in [False, True]:
            # Create figure with polar projection
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, projection='polar')
            
            # Plot interpolated data as filled contours (smooth, no dots)
            im = ax.contourf(az_plot_rad, radius, np.log10(sky_interp), levels=100, cmap='gray_r', antialiased=True)
            
            if use_mask:
                # Overlay Red Gaussian Mask
                # Calculate sigma in radians from solid angle
                solid_angle_sr = PRIMARY_BEAM_SOLID_ANGLE * (np.pi / 180.0)**2
                sigma_rad = np.sqrt(solid_angle_sr / (2 * np.pi))
                
                # Calculate theta grid (zenith angle in radians)
                theta_rad = np.radians(radius)
                
                # Gaussian beam profile
                beam_profile = np.exp(-0.5 * (theta_rad / sigma_rad)**2)
                
                # Mask opacity (0 where beam is 1, 1 where beam is 0)
                mask_opacity = 1.0 - beam_profile
                
                # Create gradient red colormap (Transparent -> Red)
                # Using RGBA: (1, 0, 0, 0) to (1, 0, 0, 0.9)
                cmap_mask = LinearSegmentedColormap.from_list('red_mask', [(1, 0, 0, 0), (1, 0, 0, 0.9)])
                
                # Plot mask
                ax.contourf(az_plot_rad, radius, mask_opacity, levels=100, cmap=cmap_mask, antialiased=True)
            
            # Add positions of bright sources
            for name, coord in bright_sources.items():
                altaz = coord.transform_to(AltAz(obstime=time_obs, location=location))
                if altaz.alt.deg > 0 and name != 'Sun':
                    az_rad = np.radians(altaz.az.deg)
                    r = 90.0 - altaz.alt.deg
                    ax.plot(az_rad, r, 'o', fillstyle='none', label=name, ms=10)
                if name == 'Sun' and altaz.alt.deg > 0:
                    az_rad = np.radians(altaz.az.deg)
                    r = 90.0 - altaz.alt.deg
                    ax.plot(az_rad, r, marker='*', color='orange', markersize=5, label='Sun')
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            
            # Set up polar plot
            ax.set_theta_zero_location('N')  # North at top
            ax.set_theta_direction(1)  # Counterclockwise
            ax.set_ylim(0, 90)  # From zenith (0) to horizon (90)
            ax.set_yticks([0, 30, 60])
            ax.set_yticklabels(['90°', '60°', '30°'])
            
            # Set azimuth ticks every 15 deg for grid lines
            ax.set_xticks(np.radians(np.arange(0, 360, 45)))
            ax.set_xticklabels(['0° (N)', '45°', '90° (E)', '135°', '180° (S)', '225°', '270° (W)', '315°'])
            
            # Add minor ticks
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))  # Altitude minor ticks every 10 deg
            ax.tick_params(axis='both', which='minor', length=3, width=0.5)
            
            ax.grid(True, alpha=0.5, which='both')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.15, shrink=0.7)
            cbar.set_label('log(Brightness Temperature (K))', rotation=270, labelpad=25)
            
            # Add title
            mask_suffix = " (Masked)" if use_mask else ""
            title = f'{freq:.2f} MHz{mask_suffix}\n{time_pst.strftime("%Y-%m-%d %H:%M:%S %Z")} PST\nPrimary Beam: {PRIMARY_BEAM_SOLID_ANGLE:.0f} deg²'
            ax.set_title(title, pad=20, fontsize=12)
            
            # Save individual image
            file_suffix = "_masked" if use_mask else "_nomask"
            filename = f'{output_dir}/sky_map_{time_str}_{freq:.2f}MHz{file_suffix}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"   Saved: {filename}")
    
    print(f"   Generated {len(frequencies) * 2} sky map images in {output_dir}/")


def main():
    """Main function to generate CASM visibilities."""
    parser = argparse.ArgumentParser(description='CASM Visibility Simulation')
    parser.add_argument('--compvis', action='store_true',
                       help='Compute visibilities (default: False, only generate sky maps)')
    parser.add_argument('--n-channels', type=int, default=100,
                       help='Number of frequency channels (default: 100)')
    parser.add_argument('--time', type=str, default=None,
                       help='Observation time. Formats: HH:MM (PST today) OR ISO-8601 (YYYY-MM-DDTHH:MM:SS)')
    
    parser.add_argument('--test-baselines', action='store_true',
                       help='Run in test mode: select max NS and EW baselines only (default: False)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CASM Visibility Simulation")
    print("=" * 60)
    
    # Set observation time
    pst = timezone('America/Los_Angeles')
    
    if args.time:
        # 1. Try ISO format (Astropy Time)
        try:
            time_obs = Time(args.time)
            print(f"   Parsed explicit time: {args.time}")
        except ValueError:
            # 2. Try HH:MM format (PST Today)
            try:
                # Get current date in PST
                now_pst = datetime.now(pst)
                
                # Parse input time string
                parts = list(map(int, args.time.split(':')))
                if len(parts) == 2:
                    hour, minute = parts
                    second = 0
                elif len(parts) == 3:
                    hour, minute, second = parts
                else:
                    raise ValueError
                
                # Create observation datetime
                obs_dt = now_pst.replace(hour=hour, minute=minute, second=second, microsecond=0)
                time_obs = Time(obs_dt)
                print(f"   Parsed local time (PST): {args.time} -> {obs_dt}")
                
            except ValueError:
                print(f"ERROR: Invalid time format '{args.time}'.")
                print("       Use ISO-8601 (YYYY-MM-DDTHH:MM:SS) OR HH:MM (24-hr PST today).")
                return
    else:
        time_obs = Time.now()
    
    # Display time in PST
    pst = timezone('America/Los_Angeles')
    time_pst = time_obs.to_datetime(timezone=pst)
    time_str = time_pst.strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"\nObservation Time: {time_obs.iso} UTC")
    print(f"Observation Time: {time_str} PST")
    
    # Step 1: Generate antenna position mapping
    print("\n1. Generating antenna position mapping...")
    antenna_mapping = generate_antenna_mapping()
    positions = antenna_mapping['positions']
    print(f"   Generated {antenna_mapping['n_antennas']} antenna positions")
    print(f"   Array dimensions: {ARRAY_NS_LENGTH}m (N-S) x {ARRAY_EW_LENGTH}m (E-W)")
    print(f"   Position range: East [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
    print(f"                    North [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
    print(f"                    Up [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
    
    # Plot antenna layout
    plot_antenna_layout(antenna_mapping, save_path='casm_antenna_layout.png')
    
    # Step 2: Define frequency range
    print("\n2. Setting up frequency range...")
    n_channels = args.n_channels
    frequencies = np.linspace(FREQ_MIN, FREQ_MAX, n_channels)
    print(f"   Frequency range: {FREQ_MIN:.1f} - {FREQ_MAX:.1f} MHz")
    print(f"   Number of channels: {n_channels}")
    print(f"   Channel width: {(FREQ_MAX - FREQ_MIN) / n_channels:.2f} MHz")
    
    # Step 3: Generate sky model for visualization (400 MHz)...
    print("\n3. Generating sky model for visualization (400 MHz)...")
    
    # Generate exact 400 MHz model for visualization
    plot_freqs = np.array([400.0])
    sky_maps_plot, pixel_directions_plot, pixel_altaz_plot = get_sky_model(plot_freqs, time_obs=time_obs, nside=64)
    
    # Visualize sky maps with and without mask
    print("\n4. Visualizing sky maps (Masked and Unmasked)...")
    plot_sky_map_with_beam(sky_maps_plot, pixel_altaz_plot, plot_freqs, time_obs, output_dir='skymaps')
    
    # Step 5: Generate visibilities (if requested)
    if args.compvis:
        print("\n5. Generating visibilities...")
        
        # Prepare inputs
        calc_baselines = None
        calc_pairs = None
        
        # Test Baselines Mode
        if args.test_baselines:
            print("\n   [TEST MODE] Selecting specific baselines...")
            # Calculate all baselines first
            all_baselines, all_pairs = calculate_baselines(positions)
            
            # Identify indices
            # Max EW: Ant 0 (0,0) to Ant 5 (0,5) -> Pair [0, 5]
            # Max NS: Ant 0 (0,0) to Ant 24 (4,0) -> Pair [0, 24]
            test_indices = []
            
            # Ant 0 index = 0
            # Ant EW Max index = N_COLS - 1 = 5
            # Ant NS Max index = (N_ROWS - 1) * N_COLS = 24
            target_pairs = [[0, 5], [0, 24]]
            print(f"   Looking for pairs: {target_pairs} (indices)")
            
            for idx, pair in enumerate(all_pairs):
                p_list = sorted(list(pair))
                if p_list == target_pairs[0] or p_list == target_pairs[1]:
                     test_indices.append(idx)
            
            if len(test_indices) != 2:
                print(f"   WARNING: Could not find all test baselines. Found indices: {test_indices}")
            else:
                print(f"   Found test baselines at indices: {test_indices}")
            
            # Select subset
            calc_baselines = all_baselines[test_indices]
            calc_pairs = all_pairs[test_indices]
            
            # Plot UVW Coverage (Contextual)
            print("   Generating UV coverage plot...")
            # Calculate UVW for all baselines for context
            uvw_all = calculate_uvw(all_baselines, frequencies)
            plot_uv_coverage(uvw_all, highlight_indices=test_indices, freq_idx=len(frequencies)//2, save_path='casm_uv_coverage_test.png')
            
        else:
            print("   Using all baselines.")

        vis_data = generate_visibilities(
            antenna_mapping,
            frequencies,
            time_obs=time_obs,
            n_sky_pixels=1000,
            custom_baselines=calc_baselines,
            custom_baseline_pairs=calc_pairs
        )
        
        print(f"\n   Generated visibilities:")
        print(f"   - Shape: {vis_data['visibilities'].shape}")
        print(f"   - Number of baselines: {vis_data['n_baselines']}")
        print(f"   - Number of frequencies: {vis_data['n_freq']}")
        print(f"   - Polarizations: {N_POL}x{N_POL}")
        
        # Save results
        filename = 'casm_visibilities_test.npz' if args.test_baselines else 'casm_visibilities.npz'
        print(f"\n6. Saving results to {filename}...")
        np.savez_compressed(
            filename,
            visibilities=vis_data['visibilities'],
            baselines=vis_data['baselines'],
            uvw=vis_data['uvw'],
            baseline_pairs=vis_data['baseline_pairs'],
            frequencies=vis_data['frequencies'],
            antenna_positions=positions,
            pixel_altaz=vis_data['pixel_altaz'],
            time_obs=time_obs.iso,
            **antenna_mapping
        )
        print(f"   Saved to {filename}")
        
        # Print summary statistics
        print("\n7. Visibility Statistics:")
        vis_abs = np.abs(vis_data['visibilities'])
        print(f"   Mean |V|: {np.mean(vis_abs):.6e}")
        print(f"   Std |V|: {np.std(vis_abs):.6e}")
        print(f"   Max |V|: {np.max(vis_abs):.6e}")
        print(f"   Min |V|: {np.min(vis_abs):.6e}")
    else:
        print("\n5. Skipping visibility computation (use --compvis to enable)")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


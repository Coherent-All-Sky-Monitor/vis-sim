import pandas as pd

# Load the CSV once
ant_table = pd.read_csv("./casm-13-new.csv")

def fetch_ant(fmt, **kwargs):
    """
    Fetch antenna information and position.
    
    Args:
        fmt: str or list specifying which format the input uses
             One of: 'idx', 'snap', 'feng', 'cond_feng', 'old_feng', 'loc'
        **kwargs: 
            - idx=<int>: antenna index
            - snap=<int>, adc=<int>: snap and ADC pair
            - feng=<int>: feng index
            - cond_feng=<int>: condensed feng index
            - old_feng=<int>: old feng index
            - loc=<tuple>: (x, y, z) position (finds closest match)
    
    Returns:
        dict with keys: antenna, x, y, z, snap, adc, feng_idx, condensed_feng_idx, 
                       old_feng_idx, functional, row, col
    """
    fmt = fmt if isinstance(fmt, str) else fmt[0]
    
    if fmt == 'idx':
        row = ant_table[ant_table['antenna'] == kwargs['idx']]
    elif fmt == 'snap':
        row = ant_table[(ant_table['snap'] == kwargs['snap']) & 
                        (ant_table['adc'] == kwargs['adc'])]
    elif fmt == 'feng':
        row = ant_table[ant_table['feng_idx'] == kwargs['feng']]
    elif fmt == 'cond_feng':
        row = ant_table[ant_table['condensed_feng_idx'] == kwargs['cond_feng']]
    elif fmt == 'old_feng':
        row = ant_table[ant_table['old_feng_idx'] == kwargs['old_feng']]
    elif fmt == 'loc':
        x, y, z = kwargs['loc']
        row = ant_table[(ant_table['x'] - x).abs() < 0.01 &
                        (ant_table['y'] - y).abs() < 0.01 &
                        (ant_table['z'] - z).abs() < 0.01]
    else:
        raise ValueError(f"Unknown format: {fmt}")
    
    if len(row) == 0:
        raise ValueError(f"No antenna found with {fmt}={kwargs}")
    if len(row) > 1:
        raise ValueError(f"Multiple antennas found with {fmt}={kwargs}")
    
    return row.iloc[0].to_dict()


def convert_ant_addr(from_fmt, from_val, to_fmt):
    """
    Convert between different antenna addressing formats.
    
    Args:
        from_fmt: str specifying source format
                 One of: 'idx', 'snap', 'feng', 'cond_feng', 'old_feng', 'row', 'col'
        from_val: the value(s) in source format
                 If from_fmt='snap', pass tuple (snap, adc)
                 If from_fmt='row' or 'col', pass the row/col string
                 Otherwise pass single int
        to_fmt: str specifying target format (same options as from_fmt)
    
    Returns:
        The value in target format (tuple for snap, single value otherwise)
    """
    
    # First resolve the source to get the antenna row
    if from_fmt == 'idx':
        row = ant_table[ant_table['antenna'] == from_val]
    elif from_fmt == 'snap':
        snap, adc = from_val
        row = ant_table[(ant_table['snap'] == snap) & (ant_table['adc'] == adc)]
    elif from_fmt == 'feng':
        row = ant_table[ant_table['feng_idx'] == from_val]
    elif from_fmt == 'cond_feng':
        row = ant_table[ant_table['condensed_feng_idx'] == from_val]
    elif from_fmt == 'old_feng':
        row = ant_table[ant_table['old_feng_idx'] == from_val]
    elif from_fmt == 'row':
        row = ant_table[ant_table['row'] == from_val]
    elif from_fmt == 'col':
        row = ant_table[ant_table['col'] == from_val]
    else:
        raise ValueError(f"Unknown source format: {from_fmt}")
    
    if len(row) == 0:
        raise ValueError(f"No antenna found with {from_fmt}={from_val}")
    if len(row) > 1 and from_fmt in ['row', 'col']:
        # Multiple antennas can have same row or col, just take first
        row = row.iloc[[0]]
    
    ant_row = row.iloc[0]
    
    # Now convert to target format
    if to_fmt == 'idx':
        return int(ant_row['antenna'])
    elif to_fmt == 'snap':
        return (int(ant_row['snap']), int(ant_row['adc']))
    elif to_fmt == 'feng':
        return int(ant_row['feng_idx'])
    elif to_fmt == 'cond_feng':
        return int(ant_row['condensed_feng_idx'])
    elif to_fmt == 'old_feng':
        return int(ant_row['old_feng_idx'])
    elif to_fmt == 'row':
        return ant_row['row']
    elif to_fmt == 'col':
        return ant_row['col']
    else:
        raise ValueError(f"Unknown target format: {to_fmt}")


# Examples:
# ant_info = fetch_ant('snap', snap=0, adc=6)  # Get antenna by snap+adc
# print(ant_info)

# feng_id = convert_ant_addr('snap', (0, 6), 'feng')  # snap 0,6 -> feng
# print(feng_id)

# snap_addr = convert_ant_addr('feng', 6, 'snap')  # feng 6 -> snap,adc
# print(snap_addr)
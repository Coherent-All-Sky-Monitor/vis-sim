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
        tuple (result, in_list) where:
            - result: The value in target format (tuple for snap, single value otherwise)
            - in_list: Boolean indicating if antenna is in CSV (True) or computed (False)
    """
    
    # First resolve the source to get the antenna row
    in_list = True
    
    if from_fmt == 'idx':
        row = ant_table[ant_table['antenna'] == from_val]
    elif from_fmt == 'snap':
        snap, adc = from_val
        row = ant_table[(ant_table['snap'] == snap) & (ant_table['adc'] == adc)]
        if len(row) == 0:
            # Not in CSV, compute feng from snap/adc formula
            if snap > 5 or (snap == 5 and adc > 3) or adc > 11:
                raise ValueError(f"Invalid snap/adc pair: snap={snap}, adc={adc}")
            in_list = False
            # Compute feng from snap/adc mapping: feng = snap * 12 + adc
            computed_feng = snap * 12 + adc
            if to_fmt == 'snap':
                return (snap, adc), in_list
            elif to_fmt == 'feng':
                return computed_feng, in_list
            else:
                raise ValueError(f"Cannot convert computed snap/adc to {to_fmt}")
    elif from_fmt == 'feng':
        row = ant_table[ant_table['feng_idx'] == from_val]
        if len(row) == 0:
            # Not in CSV, compute snap/adc from feng formula
            if from_val < 0 or from_val > 63:
                raise ValueError(f"Invalid feng index: {from_val}")
            in_list = False
            snap = from_val // 12
            adc = from_val % 12
            if to_fmt == 'feng':
                return from_val, in_list
            elif to_fmt == 'snap':
                return (snap, adc), in_list
            else:
                raise ValueError(f"Cannot convert computed feng to {to_fmt}")
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
    
    if len(row) == 0 and from_fmt not in ['snap', 'feng']:
        raise ValueError(f"No antenna found with {from_fmt}={from_val}")
    if len(row) > 1 and from_fmt in ['row', 'col']:
        row = row.iloc[[0]]
    
    ant_row = row.iloc[0]
    
    # Now convert to target format
    if to_fmt == 'idx':
        return int(ant_row['antenna']), in_list
    elif to_fmt == 'snap':
        return (int(ant_row['snap']), int(ant_row['adc'])), in_list
    elif to_fmt == 'feng':
        return int(ant_row['feng_idx']), in_list
    elif to_fmt == 'cond_feng':
        return int(ant_row['condensed_feng_idx']), in_list
    elif to_fmt == 'old_feng':
        return int(ant_row['old_feng_idx']), in_list
    elif to_fmt == 'row':
        return ant_row['row'], in_list
    elif to_fmt == 'col':
        return ant_row['col'], in_list
    else:
        raise ValueError(f"Unknown target format: {to_fmt}")


# Examples:
# result, in_list = convert_ant_addr('snap', (0, 6), 'feng')
# print(f"snap (0,6) -> feng {result}, in_list: {in_list}")

# result, in_list = convert_ant_addr('snap', (1, 5), 'feng')  # snap 1, adc 5
# print(f"snap (1,5) -> feng {result}, in_list: {in_list}")

# result, in_list = convert_ant_addr('feng', 15, 'snap')
# print(f"feng 15 -> snap {result}, in_list: {in_list}")

# result, in_list = convert_ant_addr('feng', 63, 'snap')
# print(f"feng 63 -> snap {result}, in_list: {in_list}")
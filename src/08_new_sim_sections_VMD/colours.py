

def get_mode_colour(mode_energy_index: int) -> str:
    if mode_energy_index == 1:
        return '#F4A261'
    if mode_energy_index == 2:
        return '#E76F51'
    if mode_energy_index == 3:
        return '#2A9D8F'
    if mode_energy_index == 4:
        return '#264653'
    if mode_energy_index == 5:
        return '#E9C46A'
    return '#1D3557'

def get_mode_colour_matplotlib(which_order: int) -> str:
    if which_order == 1:
        return '#d62728'
    elif which_order == 2:
        return '#1f77b4'
    elif which_order == 3:
        return '#2ca02c'
    elif which_order == 4:
        return '#9467bd'
    elif which_order == 5:
        return '#ff7f0e'
    else:
        return 'k'

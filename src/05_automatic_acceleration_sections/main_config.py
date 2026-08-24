MEASUREMENTS_FOLDER =\
    '../../data/csv'

SPEED_UNIT = 'km/h'  # 'm/s' | 'km/h'

SPEED_TOLERANCE = .5  # m/s
TOP_SPEED = 13.8889  # m/s

START_INDEX = 0
SECTIONS_DF_COLUMNS = [
    'beg', 'end', 'type'
]

RESULTS_FOLDER = \
    'result_sections'

ACTUALLY_SAVE = False

# None | 0 | 1 | 2 | ...
SPECIFIC_MEASUREMENT_INDEX = 8
SPECIFIC_START_INDEX = 2717
SPECIFIC_END_INDEX = 3828

PLOT_TYPES = [
    'zero', 'acc',
    'top_speed', 'dec',
    'nothing']
PLOT_Y_LIMS = [0, 65]
PLOT_HIGHLIGHT = [
    'acc', 'dec'
]

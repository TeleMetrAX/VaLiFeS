SECTIONS_JSON_PATH = \
    'sections.json'

SECTIONS_DATA_COLUMNS = [
    'acceleration', 'deceleration'
]
SECTIONS_DATA_IDS = [
    'filename', 'begin', 'end',
    'top_speed', 'driver'
]

CSV_DATA_FOLDER = \
    '../../data/csv'

USE_OLD = False
OLD_RESULTS_PATH =\
    'tmp_old_data.pickle'

SAMPLING_PERIOD_TOLERANCE = .005

STATUS_BAR_ON = True

# VMD parameters
ALPHA = 10000
K = 5
DC = False
INIT = 1
TAU = 0.0
TOL = 1e-6
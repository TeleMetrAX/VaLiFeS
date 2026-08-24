MEASUREMENTS_FOLDER =\
    '../../data/csv'

USE_OLD = True
OLD_RESULTS_PATH =\
    'tmp_old_data_{}.pickle'

SPEED_UNIT = 'km/h'  # 'm/s' | 'km/h'
LIMIT_PLOT1 = 17
SHOW_PLOT_NUMBER_TEXTS = False

SPEED_TOLERANCE = .5  # m/s
TOP_SPEED = 13.8889  # m/s

# use interpolation and smoothing
MAKE_DATA_BETTER = False

START_INDEX = 0
SECTIONS_DF_COLUMNS = [
    'beg', 'end', 'type'
]
SECTIONS_FOLDER = \
    'sections'

RATIO_CALC_COLS = [
    'mode_ind', 'num_around', 'num_all']

ACC_TYPE = 'acc'
DEC_TYPE = 'dec'

SAMPLING_PERIOD_TOLERANCE = .005
PRECISION = 6

STATUS_BAR_ON = True
DEBUG = False
MODE = 'show'  # 'show' | 'save'
SAVE_FOLDER = 'plots'
TO_PLOT = 'highests'  # 'all' | 'highests'
# ORDERS = [1, 2, 3, 4, 5]
ORDERS = [2, 3, 4, 5]
PLOT_LINES = True
COMBINE_ACC_DEC = True
CUSTOM_X_Y_MAX = [100, 150]
# CUSTOM_X_Y_MAX = [200, 200]
# CUSTOM_X_Y_MAX = None

PLOT_AS = [2, 1, 2/3, 2/5]
TARGET = 'paper'  # 'paper' | 'view'
YAXIS_SCALE = 'lin'  # 'lin' | 'log'

# VMD parameters
ALPHA = 10000
K = 5
DC = False
INIT = 1
TAU = 0.0
TOL = 1e-6

RATIOS_DF_COLUMNS = [
    'title', 'num_within_2', 'num_within_1',
    'num_within_2/3', 'num_within_2/5',
    'num_all_2', 'num_all_1',
    'num_all_2/3', 'num_all_2/5'
]
RATIOS_REFERENCE_FACTORS = [2, 1, 2/3, 2/5]

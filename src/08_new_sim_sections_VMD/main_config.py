NUM_SIMULATIONS = 50000
SIMULATION_MODE = 'new'  # 'new' | 'old'

USE_OLD = True
OLD_RESULTS_PATH =\
    'tmp_old_data_sim{}_rtr{}{}.pickle'

# use interpolation and smoothing (only new)
MAKE_DATA_BETTER = False

MODE = 'show'  # 'show' | 'save'
SAVE_FORMAT = 'png'  # 'pdf' | 'png'
TO_PLOT = 'highests'  # 'all' | 'highests'
TARGET = 'paper'  # 'view' | 'paper'
ORDERS = [2, 3, 4, 5]
# ORDERS = [2]
PLOT_LINES = True
PLOT_AS = [2, 2/3, 2/5]
# PLOT_AS = [2, 2/3]
# PLOT_AS = [2]
COMBINE_ACC_DEC = True
CUSTOM_X_Y_MAX = [205, 405]
# CUSTOM_X_Y_MAX = [205, 5]
# CUSTOM_X_Y_MAX = None
PLOT_YAXIS_SCALE = 'lin'  # 'lin' | 'log'

PRINT_DETAILS = False
SMALL_TK_THRESHOLD = 7.0  # s

# VMD parameters
K = 5
ALPHA = 10000
DC = False
INIT = 1
TAU = 0.0
TOL = 1e-6

RATIOS_DF_COLUMNS = [
    'title', 'num_within_2',
    # 'num_within_1',
    'num_within_2/3', 'num_within_2/5',
    'num_all_2',
    # 'num_all_1',
    'num_all_2/3', 'num_all_2/5'
]
# RATIOS_REFERENCE_FACTORS = [2, 1, 2/3, 2/5]
RATIOS_REFERENCE_FACTORS = [2, 2/3, 2/5]


# less useful parameters
########################
NOISE_CSV_FILE = '../../data/noise/speed_last_5min_5Hz.csv'
NOISE_RANDOM_SEED = 42
SPEED_UNIT = 'km/h'  # 'm/s' | 'km/h'
STATUS_BAR_ON = True
DEBUG = False
SAVE_FOLDER = 'plots'
# LIMIT_PLOT1 = 17
LIMIT_PLOT1 = 10
SHOW_PLOT_NUMBER_TEXTS = False
SECTIONS_DF_COLUMNS = [
    'beg', 'end', 'type'
]
ACC_TYPE = 'acc'
DEC_TYPE = 'dec'
# plot section types params:
PLOT_TYPES = [
    'zero', 'acc',
    'top_speed', 'dec',
    'nothing']
PLOT_Y_LIMS = [0, 100]
PLOT_HIGHLIGHT = [
    'acc', 'dec'
]


# old simulation parameters
###########################
SIM_LENGTH = 60  # minutes
SIM_STATUS = [.2, .2, .6]
SPEED_TOLERANCE = .5  # m/s
TOP_SPEED = 13.8889  # m/s
SAMPLING_PERIOD_TOLERANCE = .005
PRECISION = 6
GEN_DATA_T_COL = 'timestamps'
GEN_DATA_V_COL = 'Speed'

SIMULATION_TYPE = 4
# 0 - constant >0 acceleration (acc)
# 1 - constant <0 acceleration (dec)
# 2 - constant acceleration (acc/dec)
# 3 - sine speed (acc)
# 4 - logistic speed (acc)

# mess up the data
RATIO_TO_REMOVE = .05  # from [0.0, 1.0)

USE_RANDOM_SAMPLING_PERIOD = False
SAMPLING_PERIOD_MU = .2  # s
SAMPLING_PERIOD_SIGMA = .2  # s

USE_NOISE = True
NOISE_METHOD = 'bootstrap'  # 'bootstrap' | 'gauss' 
NOISE_MU = 0.0  # m/s
NOISE_SIGMA = 0.2  # m/s

NOISY_SAMPLING_PERIOD = 0.2105  # s

SAMPLING_PERIOD = 0.2  # s

# maximum [acc/dec]eleration (BokareM2016)
A_MAX = 2.87  # m/s^2
A_MIN = -4.33  # m/s^2

# minimum/maximum speed
V_MIN = 0.0  # m/s
V_MAX = 13.8889  # m/s (50 km/h)

# minimum/maximum duration of a section
D_MIN = 10  # s
D_MAX = 200  # s

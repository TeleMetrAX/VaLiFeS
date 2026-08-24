import numpy as np
import glob
import matplotlib.pyplot as plt

import main_config as conf
import importer as imp


if __name__ == '__main__':
    data_paths = glob.glob(f'{conf.MEASUREMENTS_FOLDER}/*.csv')

    all_num_higher_diffs = []
    all_ratio_higher_diffs = []

    # print(data_paths[17])  # outlier with 5,03% of missing data points

    for path in data_paths:
        # get data
        t, v, units, measurement_name = \
            imp.read_import_measurement_data(path, False)
        d = np.diff(t)

        thd_multiplier = 1
        num_higher_diffs = 0
        while True:
            limit = conf.SAMPLING_PERIOD + thd_multiplier * conf.DIFF_THRESHOLD
            mask = d > limit
            count = int(mask.sum())
            if count == 0 or limit > conf.LIMIT_OF_MISSING:
                break
            num_higher_diffs += count
            thd_multiplier += 1

        all_num_higher_diffs.append(num_higher_diffs)
        all_ratio_higher_diffs.append(float(np.round(num_higher_diffs / len(d), 4)))

    # all_num_higher_diffs = np.array(all_num_higher_diffs)
    all_ratio_higher_diffs = np.array(all_ratio_higher_diffs)

    print()
    print(f'Number of missing timesteps assuming a sampling period of {conf.SAMPLING_PERIOD} s '
          f'per dataset:\n{all_num_higher_diffs}\n')
    print()
    print(f'Percentage of missing timesteps assuming a sampling period of {conf.SAMPLING_PERIOD} s '
          f'per dataset, relative to the data length:\n{all_ratio_higher_diffs*100}\n')
    print()
    print(f'Highest percentage of missing timesteps assuming a sampling period of {conf.SAMPLING_PERIOD} s: '
          f'{np.round(np.max(all_ratio_higher_diffs)*100, 2)}%\n')
    print()
    print(f'Average percentage of missing timesteps assuming a sampling period of {conf.SAMPLING_PERIOD} s: '
          f'{np.round(np.average(all_ratio_higher_diffs)*100, 2)}%\n')
    print()
    print(f'Number of datasets analyzed: {len(data_paths)}\n')

    plt.bar([p for p in data_paths], all_ratio_higher_diffs*100)
    plt.xticks(rotation=90)
    plt.ylabel('Percentage of missing data points')
    plt.grid()
    plt.tight_layout()
    plt.show()

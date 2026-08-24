import matplotlib.pyplot as plt
import importer as imp
import main_config as conf
import data_processor as proc


INPUT_FILEPATH = 'c:/Downloads/8.csv '


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 1)

    t, v, units, measurement_name = \
        imp.read_import_measurement_data(INPUT_FILEPATH, False)

    axs[0].plot(t, v, label='Original')
    differences = t[1:] - t[:-1]
    axs[1].plot(differences, label='Original')

    sampling_period, t, v = \
        proc.infer_sampling_period(t, v, conf.SAMPLING_PERIOD_TOLERANCE, conf.PRECISION)

    axs[0].plot(t, v, label='Inferred')
    differences = t[1:] - t[:-1]
    axs[1].plot(differences, label='Inferred')

    t, v = proc.smooth_timeseries(t, v, rate=1/sampling_period)

    axs[0].plot(t, v, label='Smoothed')
    differences = t[1:] - t[:-1]
    axs[1].plot(differences, label='Smoothed')

    for ax in axs:
        ax.grid()
        ax.legend()
    plt.show()

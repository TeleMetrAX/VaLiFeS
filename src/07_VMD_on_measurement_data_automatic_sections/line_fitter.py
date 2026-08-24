import numpy as np
import pandas as pd

import main_config as conf


class RatiosCalculator:
    def __init__(self):
        df_cols = [conf.RATIO_CALC_COLS[0]]
        for begstr in conf.RATIO_CALC_COLS[1:]:
            for factor in conf.RATIOS_REFERENCE_FACTORS:
                df_cols.append(f'{begstr}_{get_a_str(factor)}')
        self.df = pd.DataFrame(columns=df_cols)
        for order in np.sort(conf.ORDERS):
            self.df.loc[len(self.df)] = [order] + (len(df_cols) - 1) * [0]
        self.reference_factors = conf.RATIOS_REFERENCE_FACTORS

    def add_line(self, x_t_diffs, y_iter_times, label: str):
        if label[0] == '$':
            return

        mode_ind = int(label[0])

        if mode_ind not in self.df.iloc[:, 0].values:
            ind = len(self.df)
        else:
            ind = self.df[self.df.iloc[:, 0] == mode_ind].index[0]

        num_within_bands = list()
        num_all = list()
        for factor in self.reference_factors:
            # within bands is in the 10% range of the line y = factor * x
            num_within_bands.append(
                np.sum((y_iter_times >= 0.9 * factor * x_t_diffs) & (y_iter_times <= 1.1 * factor * x_t_diffs)))
            num_all.append(len(x_t_diffs))

        for i, num in enumerate(num_within_bands):
            self.df.iloc[ind, 1 + i] += num
        for i, num in enumerate(num_all):
            self.df.iloc[ind, len(self.reference_factors) + 1 + i] += num
        return

    def get_ratios(self):
        all_ratios = []
        for _, row in self.df.iloc[::-1].iterrows():
            title = row.iloc[0]
            ratios = list()
            for i in range(len(self.reference_factors)):
                ratios.append(
                    row.iloc[1 + i] / row.iloc[len(self.reference_factors) + 1 + i])
            all_ratios.append([title] + ratios)
        return all_ratios


def get_a_str(a: float):
    if int(a) == a:
        return str(int(a))
    elif a == .5:
        return '1/2'
    elif a == 2 / 3:
        return '2/3'
    elif a == 2 / 5:
        return '2/5'
    else:
        return str(round(a, 2))


def get_ratio_str(ratio: float) -> str:
    return f'{round(100.0 * ratio, 1)}\\%'


def get_line(beg: float, end: float, a: float, b: float = 0) -> tuple:
    """
    Get line defined by y = a*x + b between beg and end.
    :param beg: beginning of the line
    :param end: end of the line
    :param a: slope
    :param b: y-intercept
    :return: x, y
    """
    x = np.linspace(beg, end, 100)
    y = a * x + b
    return x, y


def get_a_s_by_order(which_order: int) -> list:
    order_a_correspondence = {
        2: [2],
        3: [2, 1, 2 / 3],
        4: [2, 1, 2 / 3],
        5: [2, 1, 2 / 3, 1 / 2]
    }
    if which_order in order_a_correspondence.keys():
        return order_a_correspondence[which_order]
    else:
        return None

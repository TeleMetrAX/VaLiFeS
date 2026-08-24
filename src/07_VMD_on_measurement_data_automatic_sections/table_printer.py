import numpy as np


HEADER_STR = (
    r'\begin{table}[htbp!]' + '\n'
    r'\centering' + '\n'
    r'\caption{Ratio of mode inverse centre frequencies conforming to the three lines with a 10\% tolerance}' + '\n'
    r'\begin{tabular}{c|cccc}' + '\n'
    r'\toprule' + '\n'
    r'\textbf{Mode} & \multicolumn{4}{c}{Ratio of points in the 10\% range of line} \\' + '\n'
    r'\textbf{energy} & \multicolumn{4}{c}{$T_{k_i} = c(k_i) \cdot \tau_\mathrm{acc}$, where $c(k_i)=$} \\' + '\n'
    r'\textbf{index (i)} & $2$ & $1$ & $2/3$ & $2/5$ \\' + '\n'
    r'\midrule'
)

END_STR = (
    r'\bottomrule' + '\n'
    r'\end{tabular}' + '\n'
    r'\label{tab:real_ratios_in_10_percent_bands}' + '\n'
    r'\end{table}'
)

LINE_TEMPLATE = '${}$ & {}\\% & {}\\% & {}\\% & {}\\% \\\\'
HIGHLIGHTED_LINE_TEMPLATE = '${}$ & \\textbf{{{}\\%}} & {}\\% & {}\\% & {}\\% \\\\'


def print_table(all_ratios: np.ndarray, orders: list):
    lines = []
    sum_percentages = []
    for i, order in enumerate(np.sort(orders)):
        percentages = [np.round(100 * r, 1) for r in all_ratios[i]]
        sum_percentages.append(np.sum(percentages[1:]))

        if order == 2:
            lines.append(HIGHLIGHTED_LINE_TEMPLATE.format(
                order, percentages[1], percentages[2], percentages[3], percentages[4])
            )
        else:
            lines.append(LINE_TEMPLATE.format(
                order, percentages[1], percentages[2], percentages[3], percentages[4])
        )

    print(HEADER_STR)
    for line in lines:
        print(line)
    print(END_STR)

    # for sp in sum_percentages:
    #     print(np.round(sp, 1), '\n')

import os
import csv

import numpy as np
import matplotlib.pyplot as plt

EXPECTED_LEN = 42


def process_file(filename, results_object):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)

        for row in reader:
            if len(row) != EXPECTED_LEN:
                continue  # something went wrong with this row

            results_object.append(row)

        return results_object, header


results = []
for entry in os.scandir('output'):
    path = entry.path
    if path[-4:] != '.csv':
        continue
    results, headers = process_file(path, results)


def data_for_header(header_value, data):
    return data[:, index_of_header(header_value)].astype(np.float64)


def index_of_header(header_value):
    return headers.index(header_value)


print(headers)

results = np.array(results)

max_group = data_for_header("max_group", results)

keys = ["discount", "horizon", "lr_initial", "lr_final", "lr_decay", "lr_steps", "explore_initial", "explore_final", "explore_decay", "explore_steps"]
len_keys = len(keys)

fig, axs = plt.subplots(len_keys, figsize=(8, 2*len_keys), sharex='all')
fig.suptitle('Hyperparameter Results')
for (idx, key) in enumerate(keys):
    data = data_for_header(key, results)
    axs[idx].scatter(max_group, data)
    axs[idx].set(ylabel=key)
    if max(data) < 1:
        axs[idx].set(yscale='log')
    axs[idx].grid(True)

axs[-1].set(xlabel='Max Group Score')

# plt.show()
plt.savefig('fig')

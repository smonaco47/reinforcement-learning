import os
import csv

import numpy as np

EXPECTED_LEN = 42
root_dir = "output/"


def process_file(filename, results_object):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)

        for row in reader:
            if len(row) != EXPECTED_LEN:
                continue  # something went wrong with this row

            results_object.append(row)

        return results_object, header


def index_of_header(header_value):
    return headers.index(header_value)


results = []
headers = []
for entry in os.scandir(root_dir):
    path = entry.path
    if path[-4:] != '.csv':
        continue
    results, headers = process_file(path, results)

results = np.array(results)
header = np.array(headers)
results_with_header = np.vstack([header, results])

np.savetxt(f"{root_dir}consolidated_output.csv", results_with_header, '%s', delimiter=",")

# Remove colunns that don't really change or are dependent on other columns
cleaned_results = np.copy(results_with_header)
index_of_hit_goal = index_of_header("hit_goal")

cleaned_results = cleaned_results[:, :index_of_hit_goal + 1]
idx_to_remove = [index_of_header(val) for val in
                 ["lr_decay", "lr_type", "explore_decay", "explore_type", "max", "max_group"]]
cleaned_results = np.delete(cleaned_results, idx_to_remove, axis=1)

np.savetxt(f"{root_dir}output_for_dt.csv", cleaned_results, '%s', delimiter=",")

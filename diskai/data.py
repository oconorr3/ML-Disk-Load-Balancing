"""
Data processing methods.
"""
################################################
# Imports
################################################
# Global project imports:
import numpy as np
# Local project imports:
import config

################################################
# Constants
################################################
# The Scaler to use for normalization:
scaler = config.scaler

################################################
# Functions
################################################
def parse_input_row(row, previous_rows):
    """
    Parse information from a row of input. Includes previous rows up to config.num_input_lines.
    PARAMETERS:
        row : the input row to parse data from
        previous_rows : the previous rows to parse data from
    RETURNS: list of information important to this particular row of input
    """
    row_list = [0, 0, row['isWrite'], row['size']]
    for i in range(config.num_input_lines - 1):
        row_list = row_list + [0, 0, previous_rows[i]['isWrite'], previous_rows[i]['size']]
    return row_list

def parse_output_row(row):
    """
    Parse information from an output row.
    PARAMETERS:
        row : the output row to parse data from
    RETURNS: size information important to this particular row of output
    """
    return config.size_classes(row['size'])

def normalize_data(inputs):
    """
    Normalize the data from inputs using the scaler given at config.scaler.
    PARAMETERS:
        inputs : list of inputs whose data to normalize
    RETURNS: inputs, normalized using the given scaler
    """
    np_inputs = np.asarray(inputs)
    scaler.fit(np_inputs[:, [i * 4 - 1 for i in range(1, config.num_input_lines + 1)]])
    inputs_normalized = scaler.transform(
        np_inputs[:, [i * 4 - 1 for i in range(1, config.num_input_lines + 1)]])
    all_normed = np.concatenate((np_inputs[:, [0, 1, 2]], inputs_normalized[:, [0]]), axis=1)
    # all_normed = np.concatenate((all_normed, inputs_normalized[:, [0]]), axis=1)
    for i in range(2, config.num_input_lines + 1):
        x = i * 4
        all_normed = np.concatenate((all_normed, np_inputs[:, [x - 4, x - 3, x - 2]]), axis=1)
        all_normed = np.concatenate((all_normed, inputs_normalized[:, [i - 1]]), axis=1)
    return all_normed

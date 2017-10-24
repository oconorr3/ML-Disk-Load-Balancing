#!/usr/bin/env python
"""
CS 4284 Capstone Project: Machine learning for disk allocation

Authors: Philip Guzik and Robert Schofield
"""
################################################
# Imports
################################################
# Global project imports:
import getopt
import sys

# Local project imports:
import config
from diskai import start


################################################
# Constants
################################################
# The string to print to explain usage
USAGE_STRING = "usage: mldisk.py --mode=[train|test] --mlfile=<filename> [args]"

################################################
# Functions
################################################
def main():
    """
    Main method for the mldisk program.
    """
    # Retrieve the arguments from the
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv",
            ["help", "mode=", "mlfile=", "files=", "output="])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)

    # Initialize option information:
    file_list = None
    ml_data_file = None
    mode = None
    output_file = None
    verbose = False

    # Parse the arguments:
    if len(opts) < 2:
        usage()
    for opt, args in opts:
        if opt == "-v":
            verbose = True
        elif opt in ("-h", "--help"):
            display_help()
            sys.exit(0)
        elif opt in ("--mode="):
            print("Mode: %s" % args)
            mode = args
        elif opt in ("--files="):
            print("Files: %s" % args)
            file_list = args.split(',')
        elif opt in ("--mlfile="):
            print("ML data: %s" % args)
            ml_data_file = args
        elif opt in ("--output="):
            print("Output file: %s" % args)
            output_file = args
        else:
            assert False, "unhandled option"

    # Send the information from the arguments to the start module:
    start.initialize(mode, ml_data_file, file_list, verbose, config.size_classes)

def usage():
    """
    Print out the usage string.
    """
    print(USAGE_STRING)

def display_help():
    """
    Displays the help message for the program.
    """
    print(usage() + \
"""Modes for mldisk:
train      : train the neural network based on the data from the given file names
test       : test the neural network based on the saved network based on the given
             dataset on the given file names

Options for mldisk:
mlfile     : the name of the input/output file for the neural network
files      : the filenames to train/test on. Example: --files=file1,file2,file3
output     :
-v verbose : train the neural network in verbose mode
""")


if __name__ == '__main__':
    main()

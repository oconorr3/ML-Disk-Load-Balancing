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
import diskai.skl as skl
import diskai.tf as tf


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
        opts, args = getopt.getopt(sys.argv[1:], "hvm:o:f:s:t:",
            [
                "help", "mode=", "mlfile=", "files=", "output=", "sklmethods=", "tfmethods=",
                "verbose"
            ])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)

    # Initialize option information:
    file_list = None
    ml_data_file = "mldata.pkl"
    mode = None
    output_file = None
    verbose = False
    use_sklearn = False
    sklmethods = []
    use_tflow = False
    tfmethods = []

    # Parse the arguments:
    if len(opts) < 2:
        usage()
    for opt, args in opts:
        if opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("--mode="):
            print("Mode: %s" % args)
            mode = args
        elif opt in ("-h", "--help"):
            display_help()
            sys.exit(0)
        elif opt in ("-f", "--files="):
            print("Files: %s" % args)
            file_list = args.split(',')
        elif opt in ("-l", "--mlfile="):
            print("ML data: %s" % args)
            ml_data_file = args
        elif opt in ("-o", "--output="):
            print("Output file: %s" % args)
            output_file = args
        elif opt in ("-s", "--sklmethods="):
            print("Scikit-learn methods: %s" % args)
            use_sklearn = True
            for method in args.split(','):
                sklmethods.append(method)
        elif opt in ("-t", "--tfmethods="):
            print("Tensorflow methods: %s" % args)
            use_sklearn = True
            for method in args.split(','):
                tfmethods.append(method)
        else:
            assert False, "unhandled option"

    # A mode must be selected:
    if mode is None:
        assert False, "You must select a mode using --mode=<mode>"

    # Give each method the proper verbose flag:
    if use_sklearn:
        for method in sklmethods:
            config.sklmethods[method].verbose = verbose
            print(config.sklmethods[method])
        skl.initialize(mode, file_list, sklmethods)
    if use_tflow:
        assert False, "Tensorflow is not yet supported."
    # Send the information from the arguments to the start module:
    # start.initialize(mode, ml_data_file, file_list, verbose)

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
-f files      : the filenames to train/test on. Example: --files=file1,file2,file3
-l mlfile     : the name of the input/output file for the neural network
-o outputdir  : output directory for network print statements
-s sklmethods : the sklearn methods to use, as defined in config.py
-t tfmethods  : the tensorflow methods to use, as defined in config.py
-v verbose    : train the neural network in verbose mode (output to STDOUT)
""")


if __name__ == '__main__':
    main()

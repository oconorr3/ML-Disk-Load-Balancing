#!/usr/bin/env python
"""
Machine learning for disk allocation

Authors: Philip Guzik and Robert Schofield
"""
import getopt
import sys

def main():
    """
    Main method for the mldisk program.
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv",
            ["help", "mlfile=", "files="])
    except getopt.GetoptError as err:
        print str(err)  
        usage()
        sys.exit(2)
    output = None
    verbose = False
    if len(opts) < 2:
        usage()
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            display_help()
            sys.exit()
        elif o in ("-o", "--output"):
            output = a
        else:
            assert False, "unhandled option"


usage_string = "usage: mldisk.py --mode=[train|test] --mlfile=<filename> [args]"
def usage():
    print(usage_string)

def display_help():
    """
    Displays the help message for the program.
    """
    print(
"""Modes for mldisk:
train      : train the neural network based on the data from the given file names
test       : test the neural network based on the saved network based on the given
             dataset on the given file names

Options for mldisk:
mlfile     : the name of the input/output file for the neural network
files      : the filenames to train/test on. Example: --files=file1,file2,file3
-v verbose :
""")


if __name__ == '__main__':
    main()

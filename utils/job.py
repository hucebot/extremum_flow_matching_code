import os
import io
import sys
import atexit
import time
import argparse
import importlib
import yaml
import inspect
import traceback
import contextlib
import matplotlib

from utils.logger import LoggerWriter
from utils import string

def init(params, description=""):
    """Initialize job and retrieve parameters from 
    either job launcher or command line.
    params: Dict of default parameters name and values pairs.
    Description: Optional text description for command line help.
    Return updated parameters dict and output logger.
    """

    #Check inputs
    assert isinstance(description, str)
    assert isinstance(params, dict)

    #Retrieve the module name of caller
    frame = inspect.stack()[1].frame
    path_module = inspect.getmodule(frame).__file__
    module_name = os.path.basename(path_module)[:-3]

    #Parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    #Declare command line arguments as optional with default value
    parser.add_argument(
        "--prefix_out", default="/tmp/" + module_name + "_", 
        help="Path prefix to output directory. default=%(default)s")
    parser.add_argument(
        "--label", default="", 
        help="Additional label to append to path_out directory. default=%(default)s")
    parser.add_argument(
        "--print_params", action='store_true', 
        help="Print default parameters in YAML format and exit")
    parser.add_argument(
        "--detached", action='store_true', 
        help="Detached mode, write standard output in path")
    for key, value in params.items():
        if isinstance(value, bool):
            parser.add_argument("--"+key, default=value, 
                type=lambda x: (str(x).lower() == 'true'))
        elif isinstance(value, list):
            def tmp_list_of_strings(arg):
                return arg.split(',')
            parser.add_argument("--"+key, default=value, type=tmp_list_of_strings, help="default=%(default)s")
        else:
            parser.add_argument("--"+key, default=value, type=type(value), help="default=%(default)s")
    #Parse arguments
    args = parser.parse_args()
    #Implement special commands
    if args.print_params:
        print(yaml.dump(params, default_flow_style=False))
        exit()
    #Retrieve parameter values from command line
    for key, value in params.items():
        params[key] = vars(args)[key]
    path_out = args.prefix_out
    path_out += string.now_to_str()
    if len(args.label) > 0:
        path_out += "_" + args.label
    #Initialize logger
    logger = LoggerWriter(path_out)
    #Implement detached mode
    params["detached"] = args.detached
    if args.detached:
        print("Running in detached mode")
        #Set matplotlib in headless mode with alternative 
        #back-end to prevent memory leak
        matplotlib.use("agg")
        #Redirect standard and error outputs
        sys.stdout = logger._stream
        sys.stderr = logger._stream
        #Add verbose info to output and flush logger at exit
        logger.add_print("Begin job: " + string.now_to_str())
        time_start = time.time()
        def tmp_atexit_handler():
            logger.add_print("End job: " + string.now_to_str() + \
                " length: " + str(time.time()-time_start))
            logger.close()
        atexit.register(tmp_atexit_handler)
    #Return parameter and logger
    return params, logger


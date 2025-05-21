import os
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

from utils.logger import LoggerReader

#Define command line arguments and parse user inputs
parser = argparse.ArgumentParser(description="Logs viewer")
parser.add_argument("path_folders", nargs="+", help="List of log folders")
args = parser.parse_args()

#Check input paths
for path in args.path_folders:
    if not os.path.isdir(path):
        print("Invalid log path: " + path)
        exit()

#Load logs as reader
print("Log directories:")
dict_loggers = {}
for path in args.path_folders:
    dirname = os.path.basename(os.path.join(path, "")[:-1])
    dict_loggers[dirname] = LoggerReader(path)
    print(dirname)
names_logs = list(dict_loggers.keys())

dict_group_scalars = {}
list_roots = []
dict_data_X_epoch = {}
dict_data_X_time = {}
dict_data_Y = {}
def load_data():
    
    global dict_group_scalars, list_roots
    global dict_data_X_epoch, dict_data_X_time, dict_data_Y
    dict_group_scalars = {}
    list_roots = []
    dict_data_X_epoch = {}
    dict_data_X_time = {}
    dict_data_Y = {}
    
    #Retrieve scalar data and names
    dict_data_scalars = {}
    names_scalars = set()
    for name, logger in dict_loggers.items():
        print("Loading:", name)
        dict_data_scalars[name] = logger.get_scalars()
        names_scalars = names_scalars.union(set(dict_data_scalars[name].keys()))
    names_scalars = list(names_scalars)
    names_scalars.sort()

    #split the scalar names into groups
    for name in names_scalars:
        index = name.find("/")
        root = ""
        if index != -1:
            root = name[:index]
            name = name[index+1:]
        if not root in dict_group_scalars:
            dict_group_scalars[root] = []
        dict_group_scalars[root].append(name)
    list_roots = list(dict_group_scalars.keys())
    list_roots.sort()

    #Group data scalars by epoch index such 
    #that values are averaged over mini batches
    for name_log, data in dict_data_scalars.items():
        dict_data_X_epoch[name_log] = {}
        dict_data_X_time[name_log] = {}
        dict_data_Y[name_log] = {}
        for name_scalar, array in data.items():
            array_sorted = array
            array_epoch = array_sorted[:,0].astype(int)
            dict_data_X_epoch[name_log][name_scalar] = np.unique(array_epoch)
            indices_not_zero = np.bincount(array_epoch) > 0
            dict_data_X_time[name_log][name_scalar] = \
                (np.bincount(array_epoch, weights=array_sorted[:,2]) / \
                (np.bincount(array_epoch)+1e-6))[indices_not_zero]
            dict_data_Y[name_log][name_scalar] = \
                (np.bincount(array_epoch, weights=array_sorted[:,3]) / \
                (np.bincount(array_epoch)+1e-6))[indices_not_zero]

#Preload data
load_data()

#Plot scalars with selected prefix root
fig = plt.figure(constrained_layout=True)
button1 = None
button2 = None
def generate_scalar_plot(root, list_names, mode_axis):
    global fig, button1, button2
    #Reload data
    load_data()
    #Clear figure
    fig.clf()
    #Compute required rows
    size_col = 5
    size_row = max((len(list_names))//size_col+1,1)
    width = 22.0
    height = (size_row+1)*4.0
    #Generate subplot grid
    gs = fig.add_gridspec(size_row+1, size_col)
    #Define header row (root selector)
    ax_header1 = fig.add_subplot(gs[0, 0])
    ax_header1.set_title("Scalar Categories:")
    button1 = matplotlib.widgets.RadioButtons(
        ax=ax_header1, 
        labels=list_roots, 
        active=list_roots.index(root), 
        activecolor="red")
    def func_on_click1(label):
        generate_scalar_plot(label, dict_group_scalars[label], mode_axis)
    button1.on_clicked(func_on_click1)
    #Define header row (legend)
    ax_header2 = fig.add_subplot(gs[0, 1:])
    for name_log in names_logs:
        ax_header2.plot([], [], label=name_log)
    ax_header2.legend(loc="upper left", title="Logs")
    ax_header2.axis("off")
    #Define header row (axis selection)
    ax_header2 = fig.add_subplot(gs[0, 3])
    ax_header2.set_title("Axis Selection:")
    button2 = matplotlib.widgets.RadioButtons(
        ax=ax_header2, 
        labels=["Epoch", "Time"],
        active=mode_axis,
        activecolor="red")
    def func_on_click2(label):
        generate_scalar_plot(root, list_names, (mode_axis+1)%2)
    button2.on_clicked(func_on_click2)
    #Define subplot rows
    ax_plot_all = fig.add_subplot(gs[len(list_names)//size_col+1, len(list_names)%size_col])
    for index, name_scalar in enumerate(list_names):
        index_row = index//size_col+1
        index_col = index%size_col
        ax_plot = fig.add_subplot(gs[index_row, index_col])
        for name_log in names_logs:
            prefix = root+"/" if len(root) > 0 else ""
            if mode_axis == 0 and name_log in dict_data_X_epoch and \
                name_log in dict_data_Y and \
                prefix+name_scalar in dict_data_X_epoch[name_log] and \
                prefix+name_scalar in dict_data_Y[name_log]:
                ax_plot.plot(
                    dict_data_X_epoch[name_log][prefix+name_scalar],
                    dict_data_Y[name_log][prefix+name_scalar],
                    label=name_log,
                    marker=".")
                ax_plot_all.plot(
                    dict_data_X_epoch[name_log][prefix+name_scalar],
                    dict_data_Y[name_log][prefix+name_scalar],
                    label=name_scalar,
                    marker=".")
            if mode_axis == 1 and name_log in dict_data_X_time and \
                name_log in dict_data_Y and \
                prefix+name_scalar in dict_data_X_time[name_log] and \
                prefix+name_scalar in dict_data_Y[name_log]:
                ax_plot.plot(
                    dict_data_X_time[name_log][prefix+name_scalar],
                    dict_data_Y[name_log][prefix+name_scalar],
                    label=name_log,
                    marker=".")
                ax_plot_all.plot(
                    dict_data_X_time[name_log][prefix+name_scalar],
                    dict_data_Y[name_log][prefix+name_scalar],
                    label=name_scalar,
                    marker=".")
        ax_plot.set_title(name_scalar)
        ax_plot.get_yaxis().get_major_formatter().set_useOffset(False)
        ax_plot.grid()
    ax_plot_all.set_title("all")
    ax_plot_all.legend()
    ax_plot_all.grid()
    #Configure figure display
    fig.set_size_inches(width, height)
    #Update figure display
    fig.show()

generate_scalar_plot(list_roots[0], dict_group_scalars[list_roots[0]], 0)
plt.show()
exit()


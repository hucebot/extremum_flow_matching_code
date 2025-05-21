import datetime

def dict_to_str(d):
    """Convert given dict to string compatible with filesystem name"""

    sss = ""
    for name, value in d.items():
        sss += name + "_" + str(value) + "_"
    if len(sss) != 0:
        sss = sss[0:-1]
    return sss

def now_to_str():
    """Return current date and time as string"""

    today = datetime.datetime.now()
    return today.strftime("%Y%m%d_%H%M%S")


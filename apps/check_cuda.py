import torch
import platform
is_cuda = torch.cuda.is_available()
if is_cuda:
    print("{}:".format(platform.node()), "Yes ({})".format(torch.cuda.get_device_name(torch.cuda.current_device())))
else:
    print("{}:".format(platform.node()), "No")


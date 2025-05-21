import numpy as np
import matplotlib
import pickle

from utils import check_type

def generate_data_scalar_from_func(
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100):
    """Generate and return (x,y,z) data for 2d heatmap and surface.
    Given function with batched (x,y)->z is used to generate the data within given min and max range.
    """
    
    #Check inputs
    assert callable(func)
    assert check_type.is_scalar(min_x) and check_type.is_scalar(max_x) and min_x < max_x
    assert isinstance(steps_x, int) and steps_x > 0
    assert check_type.is_scalar(min_y) and check_type.is_scalar(max_y) and min_y < max_y
    assert isinstance(steps_y, int) and steps_y > 0

    #Generate grid
    x = np.linspace(min_x, max_x, steps_x)
    y = np.linspace(min_y, max_y, steps_y)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    #Compute value by converting 2d grid to batched 1d input and back
    tmp_x = xx.reshape((x.shape[0]*y.shape[0], 1))
    tmp_y = yy.reshape((x.shape[0]*y.shape[0], 1))
    tmp_z = func(tmp_x, tmp_y)
    zz = tmp_z.reshape((x.shape[0], y.shape[0]))

    return xx, yy, zz

def generate_data_field_from_func(
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100,
    use_color=False):
    """Generate and return (x,y,u,v) data for 2d vector field.
    Given function with batched (x,y)->(u,v) is used to generate the data within given min and max range.
    use_color: If True, assume the function also output a color scalar 
    (x,y)->(u,v,c) and (x,y,u,v) data is returned.
    """
    
    #Check inputs
    assert callable(func)
    assert check_type.is_scalar(min_x) and check_type.is_scalar(max_x) and min_x < max_x
    assert isinstance(steps_x, int) and steps_x > 0
    assert check_type.is_scalar(min_y) and check_type.is_scalar(max_y) and min_y < max_y
    assert isinstance(steps_y, int) and steps_y > 0

    #Generate grid
    x = np.linspace(min_x, max_x, steps_x)
    y = np.linspace(min_y, max_y, steps_y)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    #Compute value by converting 2d grid to batched 1d input and back
    tmp_x = xx.reshape((x.shape[0]*y.shape[0], 1))
    tmp_y = yy.reshape((x.shape[0]*y.shape[0], 1))
    if use_color:
        tmp_u, tmp_v, tmp_c = func(tmp_x, tmp_y)
        uu = tmp_u.reshape((x.shape[0], y.shape[0]))
        vv = tmp_v.reshape((x.shape[0], y.shape[0]))
        cc = tmp_c.reshape((x.shape[0], y.shape[0]))
        return xx, yy, uu, vv, cc
    else:
        tmp_u, tmp_v = func(tmp_x, tmp_y)
        uu = tmp_u.reshape((x.shape[0], y.shape[0]))
        vv = tmp_v.reshape((x.shape[0], y.shape[0]))
        return xx, yy, uu, vv

def plot_heatmap_from_func(
    ax, fig,
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100,
    no_colorbar=False,
    **params):
    """Plot a 2d heatmap on given matplotlib axes and figure.
    Given function with batched (x,y)->z is used to generate the data within given min and max range.
    Additional matplotlib arguments can ge passed to pcolormesh.
    """
    
    #Check inputs
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(fig, matplotlib.figure.Figure)

    #Generate the data
    xx, yy, zz = generate_data_scalar_from_func(
        func,
        min_x=min_x, max_x=max_x, steps_x=steps_x, 
        min_y=min_y, max_y=max_y, steps_y=steps_y)

    #Plotting
    im = ax.pcolormesh(xx, yy, zz, **params, shading="gouraud")
    if not no_colorbar:
        fig.colorbar(im, ax=ax)

def plot_surface_from_func(
    ax, fig,
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100,
    no_colorbar=False,
    **params):
    """Plot a 2d -> 1d surface on given matplotlib axes and figure.
    Given function with batched (x,y)->z is used to generate the data within given min and max range.
    Additional matplotlib arguments can ge passed to plot_surface.
    """
    
    #Check inputs
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(fig, matplotlib.figure.Figure)
    
    #Generate the data
    xx, yy, zz = generate_data_scalar_from_func(
        func,
        min_x=min_x, max_x=max_x, steps_x=steps_x, 
        min_y=min_y, max_y=max_y, steps_y=steps_y)

    #Plotting
    im = ax.plot_surface(xx, yy, zz, **params)
    if not no_colorbar:
        fig.colorbar(im, ax=ax)

def plot_wireframe_from_func(
    ax, fig,
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100,
    **params):
    """Plot a 2d -> 1d surface wireframe on given matplotlib axes and figure.
    Given function with batched (x,y)->z is used to generate the data within given min and max range.
    Additional matplotlib arguments can ge passed to plot_wireframe.
    """
    
    #Check inputs
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(fig, matplotlib.figure.Figure)
    
    #Generate the data
    xx, yy, zz = generate_data_scalar_from_func(
        func,
        min_x=min_x, max_x=max_x, steps_x=steps_x, 
        min_y=min_y, max_y=max_y, steps_y=steps_y)

    #Plotting
    im = ax.plot_wireframe(xx, yy, zz, **params)

def plot_stream_from_func(
    ax, fig,
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100,
    use_color=False,
    **params):
    """Plot a 2d -> 2d vector stream on given matplotlib axes and figure.
    Given function with batched (x,y)->(u,v) is used to generate the data within given min and max range.
    use_color: If True, assume the function also output a color scalar 
    (x,y)->(u,v,c) and (x,y,u,v,c) data is returned.
    Additional matplotlib arguments can ge passed to streamplot.
    """
    
    #Check inputs
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(fig, matplotlib.figure.Figure)
    
    if use_color:
        #Generate the data
        xx, yy, uu, vv, cc = generate_data_field_from_func(
            func,
            min_x=min_x, max_x=max_x, steps_x=steps_x, 
            min_y=min_y, max_y=max_y, steps_y=steps_y,
            use_color=True)
        #Plotting
        ax.streamplot(xx, yy, uu, vv, color=cc, **params)
    else:
        #Generate the data
        xx, yy, uu, vv = generate_data_field_from_func(
            func,
            min_x=min_x, max_x=max_x, steps_x=steps_x, 
            min_y=min_y, max_y=max_y, steps_y=steps_y,
            use_color=False)
        #Plotting
        ax.streamplot(xx, yy, uu, vv, **params)

def plot_quiver_from_func(
    ax, fig,
    func, 
    min_x=-1.0, max_x=1.0, steps_x=100, 
    min_y=-1.0, max_y=1.0, steps_y=100,
    use_color=False,
    **params):
    """Plot a 2d -> 2d vector field with arrows on given matplotlib axes and figure.
    Given function with batched (x,y)->(u,v) is used to generate the data within given min and max range.
    use_color: If True, assume the function also output a color scalar 
    (x,y)->(u,v,c) and (x,y,u,v,c) data is returned.
    Additional matplotlib arguments can ge passed to quiver.
    """
    
    #Check inputs
    assert isinstance(ax, matplotlib.axes.Axes)
    assert isinstance(fig, matplotlib.figure.Figure)
    
    if use_color:
        #Generate the data
        xx, yy, uu, vv, cc = generate_data_field_from_func(
            func,
            min_x=min_x, max_x=max_x, steps_x=steps_x, 
            min_y=min_y, max_y=max_y, steps_y=steps_y,
            use_color=True)
        #Plotting
        ax.quiver(xx, yy, uu, vv, cc, **params)
    else:
        #Generate the data
        xx, yy, uu, vv = generate_data_field_from_func(
            func,
            min_x=min_x, max_x=max_x, steps_x=steps_x, 
            min_y=min_y, max_y=max_y, steps_y=steps_y,
            use_color=False)
        #Plotting
        ax.quiver(xx, yy, uu, vv, **params)

def save_raw_figure(path, fig):
    """Write to given filename the given matplotlib 
    figure as raw pickle format."""
    
    assert isinstance(path, str)
    assert isinstance(fig, matplotlib.figure.Figure)
    pickle.dump(fig, open(path, "wb"))

def load_raw_figure(path):
    """Load from given path to raw pickle format a matplotlib 
    figure and return it as (fig, axs)."""
    
    assert isinstance(path, str)
    fig = pickle.load(open(path, "rb"))
    assert isinstance(fig, matplotlib.figure.Figure)
    return fig, fig.axes


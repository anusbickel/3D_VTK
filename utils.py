from matplotlib import colors
import matplotlib.cm
import numpy as np
from postcactus.integral_curves import curves_scalar_log10
from postcactus.unit_abbrev import *
import postcactus.viz_vtk as viz
import h5py
import sys
import os
import vtk
from contextlib import contextmanager
from vtk import (
    VTK_MAJOR_VERSION,
    vtkColorTransferFunction,
    vtkWindowToImageFilter,
    vtkPNGWriter,
    vtkPoints,
    vtkCellArray,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkActor
)
from vtk.util import numpy_support


### Colormap Utils ###
def make_cmap_magnetic():
    """Create custom magnetic field colormap."""
    clrs = [
        (0, (0, 0, 0.5)),
        (0.3, (0.5, 0, 0.5)),
        (0.4, (1.0, 0, 0)),
        (0.6, (1.0, 0.8, 0.0)),
        (0.7, (1.0, 1.0, 0)),
        (1.0, (1, 1, 1))
    ]
    cdict = {
        'red': [(y, r, r) for y, (r, g, b) in clrs],
        'green': [(y, g, g) for y, (r, g, b) in clrs],
        'blue': [(y, b, b) for y, (r, g, b) in clrs]
    }
    name = 'magnetic'
    return colors.LinearSegmentedColormap(name, cdict)


additional_colormaps = [make_cmap_magnetic()]
additional_colormaps = {c.name: c for c in additional_colormaps}


def to_rgb(clr):
    """Convert color to RGB tuple."""
    if clr is None:
        return None
    cc = colors.ColorConverter()
    return cc.to_rgb(clr)


def get_cmap(name):
    """Get colormap by name, including custom colormaps."""
    m = additional_colormaps.get(name)
    if m is None:
        return matplotlib.cm.get_cmap(name)
    return m


### VTK Utils ###
def vtkConnectDataInput(src, dst):
    """Connect VTK data source to destination (version-compatible)."""
    if VTK_MAJOR_VERSION <= 5:
        dst.SetInput(src)
    else:
        dst.SetInputData(src)


def vtkConnectOutputInput(src, dst):
    """Connect VTK output to input (version-compatible)."""
    if VTK_MAJOR_VERSION <= 5:
        dst.SetInput(src.GetOutput())
    else:
        dst.SetInputConnection(src.GetOutputPort())


def make_vtkColorTransferFunction(bnds, clrs):
    """
    Construct a vtkColorTransferFunction.
    
    Args:
        bnds: Value boundaries
        clrs: Colors in range [0,1]
        
    Returns:
        vtkColorTransferFunction
    """
    vmin, vmax = min(bnds), max(bnds)
    ctf = vtkColorTransferFunction()
    ctf.range = [vmin, vmax]
    for v, (r, g, b) in zip(bnds, clrs):
        ctf.AddRGBPoint(v, r, g, b)
    return ctf


def get_vtkColorTransferFunction(name, vmin=0, vmax=1.0, nsegments=256):
    """
    Returns a vtkColorTransferFunction by name.
    
    Args:
        name: Colormap name (matplotlib or custom)
        vmin, vmax: Value range
        nsegments: Number of color segments
        
    Returns:
        vtkColorTransferFunction
    """
    cm = get_cmap(name)
    sk = np.linspace(0.0, 1.0, nsegments)
    vk = np.linspace(vmin, vmax, nsegments)
    ck = cm(sk)[:, :3]
    return make_vtkColorTransferFunction(vk, ck)


def write_png(win, fname, scale=1):
    """
    Write render window to PNG file.
    
    Args:
        win: RenderWindow object
        fname: Output filename (without .png extension)
        scale: Image scale factor
    """
    windowToImage = vtkWindowToImageFilter()
    win.renderWin.Render()
    windowToImage.SetInput(win.renderWin)
    windowToImage.SetScale(scale)
    windowToImage.SetInputBufferTypeToRGB()
    windowToImage.ReadFrontBufferOff()
    writer = vtkPNGWriter()
    windowToImage.Update()
    if not fname.endswith('.png'):
        fname = fname + '.png'
    writer.SetFileName(fname)
    writer.SetInputConnection(windowToImage.GetOutputPort())
    writer.Write()


def RhoData_to_vtkImageData(rhodata, dtype=float):
    """
    Converts numpy array to VTK image data.
    
    Args:
        rhodata: Density data numpy array
        dtype: Data type
        
    Returns:
        vtkImageData object
    """
    dat0 = rhodata
    if dtype is None:
        dtype = dat0.dtype
    else:
        dtype = np.dtype(dtype)
    
    vtype = numpy_support.get_vtk_array_type(dtype)
    dat = np.log(np.asarray(dat0))[::, ::, ::]
    flt = dat.ravel(order='F')
    varray = numpy_support.numpy_to_vtk(
        num_array=flt,
        deep=True,
        array_type=vtype
    )
    
    resx = len(dat)
    resy = len(dat[0])
    resz = len(dat[0][0])
    xmax = len(dat) / 4
    
    dx = 2 * xmax / (resx - 1)
    
    imgdat = vtk.vtkImageData()
    imgdat.GetPointData().SetScalars(varray)
    imgdat.SetDimensions(resx, resy, resz)
    imgdat.SetOrigin(-xmax, -xmax, -xmax)
    imgdat.SetSpacing(dx, dx, dx)
    
    return imgdat, resx


def grid(hrho, grid_size=[2, 2, 2], spacing=20, renderer=None, X=None):
    """
    Add coordinate grid to renderer.
    
    Args:
        hrho: Density filename (for extracting distance)
        grid_size: Grid dimensions [nx, ny, nz]
        spacing: Grid spacing
        renderer: VTK renderer
        X: Grid resolution
    """
    dis = int(hrho.split("_")[2][1:-1])
    conversion = X // 2 * (1 / dis) * KM_CU
    spacing = spacing * conversion
    
    points = vtkPoints()
    lines = vtkCellArray()
    
    # Generate grid points
    for i in range(-grid_size[0], grid_size[0] + 1):
        for j in range(-grid_size[1], grid_size[1] + 1):
            for k in range(-grid_size[2], grid_size[2] + 1):
                points.InsertNextPoint(i * spacing, j * spacing, k * spacing)
    
    def add_lines(start, end, lines):
        """Helper to add line between two points."""
        lines.InsertNextCell(2)
        lines.InsertCellPoint(start)
        lines.InsertCellPoint(end)
    
    # Calculate grid dimensions
    num_points_x = 2 * grid_size[0] + 1
    num_points_y = 2 * grid_size[1] + 1
    num_points_z = 2 * grid_size[2] + 1
    
    # Add lines along x direction
    for j in range(num_points_y):
        for k in range(num_points_z):
            for i in range(num_points_x - 1):
                start_index = i + j * num_points_x + k * num_points_x * num_points_y
                end_index = (i + 1) + j * num_points_x + k * num_points_x * num_points_y
                add_lines(start_index, end_index, lines)
    
    # Add lines along y direction
    for i in range(num_points_x):
        for k in range(num_points_z):
            for j in range(num_points_y - 1):
                start_index = i + j * num_points_x + k * num_points_x * num_points_y
                end_index = i + (j + 1) * num_points_x + k * num_points_x * num_points_y
                add_lines(start_index, end_index, lines)
    
    # Add lines along z direction
    for i in range(num_points_x):
        for j in range(num_points_y):
            for k in range(num_points_z - 1):
                start_index = i + j * num_points_x + k * num_points_x * num_points_y
                end_index = i + j * num_points_x + (k + 1) * num_points_x * num_points_y
                add_lines(start_index, end_index, lines)
    
    # Create polydata and actor
    grid_polydata = vtkPolyData()
    grid_polydata.SetPoints(points)
    grid_polydata.SetLines(lines)
    
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(grid_polydata)
    
    actor = vtkActor()
    actor.SetMapper(mapper)
    
    renderer.AddActor(actor)
    return renderer


def print_actor_info(renderer):
    """
    Print information about actors in renderer.
    
    Args:
        renderer: VTK renderer
        
    Returns:
        Number of actors
    """
    actors = renderer.GetActors()
    num_actors = actors.GetNumberOfItems()
    print(f"Total actors: {num_actors}")
    
    actors.InitTraversal()
    for i in range(num_actors):
        actor = actors.GetNextItem()
        visibility = "visible" if actor.GetVisibility() else "hidden"
        opacity = actor.GetProperty().GetOpacity()
        print(f"  Actor {i}: {visibility}, opacity={opacity:.2f}")
    
    return num_actors


### Data Utils ###
def data_range(data, vmin=None, vmax=None):
    """
    Get data range with optional overrides.
    
    Args:
        data: Data array
        vmin, vmax: Optional min/max overrides
        
    Returns:
        Tuple of (vmin, vmax)
    """
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    return vmin, vmax


def load_h5(fname):
    """
    Load data from HDF5 file.
    
    Args:
        fname: Filename
        
    Returns:
        Data array
    """
    f = h5py.File(fname, "r")
    a_group_key = list(f.keys())[0]
    h5 = list(f[a_group_key])
    return h5


### Math Utils ###
def log_data_range(vmax, magnitudes, scalar):
    """
    Calculate logarithmic data range.
    
    Args:
        vmax: Maximum value
        magnitudes: Number of orders of magnitude
        scalar: Scalar data
        
    Returns:
        Tuple of (log_scalar, log_min, log_max, num_labels)
    """
    vmin = vmax / 10**magnitudes
    logsc = curves_scalar_log10(scalar, vmin / 100.0)
    lgmin, lgmax = np.log10(vmin), np.log10(vmax)
    lgmin, lgmax, numlbl = viz.align_range_decimal(lgmin, lgmax)
    return logsc, lgmin, lgmax, numlbl


def abs_cos_theta(pos):
    """
    Calculate absolute cosine of theta angle.
    
    Args:
        pos: Position array [x, y, z]
        
    Returns:
        Absolute cosine of theta
    """
    x, y, z = pos
    d = np.sqrt(x**2 + y**2)
    cth = np.cos(np.arctan2(d, np.abs(z)))
    return cth


### Other Utils ###
@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def natural_sort_key(filename):
    """
    Extract number from figure_*.png format for natural sorting.
    
    Args:
        filename: Filename string or Path
        
    Returns:
        Integer for sorting
    """
    import re
    from pathlib import Path
    
    match = re.search(r'figure_(\d+)\.png', str(filename))
    return int(match.group(1)) if match else 0


def sort_figure_files(directory, pattern="figure_*.png"):
    """
    Sort figure files in numerical order.
    
    Args:
        directory: Directory containing files
        pattern: Glob pattern for files
        
    Returns:
        Sorted list of Path objects
    """
    from pathlib import Path
    return sorted(Path(directory).glob(pattern), key=natural_sort_key)

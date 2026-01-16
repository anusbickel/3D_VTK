import logging
import numpy as np
import vtk
from math import pi, cos, sin
from postcactus import viz_vtk as viz
from postcactus.unit_abbrev import *
from postcactus.simdir import SimDir
import matplotlib as mpl

from utils import (
    RhoData_to_vtkImageData, get_vtkColorTransferFunction,
    make_vtkColorTransferFunction, data_range, to_rgb,
    log_data_range, write_png, grid, suppress_stdout,
    vtkConnectDataInput, vtkConnectOutputInput
)
from data_loader import SimulationData
from config import RenderConfig

logger = logging.getLogger(__name__)


class DensityRenderer:
    """Handles density volume rendering."""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize density renderer.
        
        Args:
            renderer: VTK renderer to add actors to
            
        Example:
            density_renderer = DensityRenderer(vtk_renderer)
        """
        self.renderer = renderer
        logger.debug("Initialized DensityRenderer")
    
    def render_density_levels(self, rho: np.ndarray, levels: list,
                              cmap: str = None, vmin: float = None,
                              vmax: float = None, show_cbar=True,
                              opacity: float = 1.0, diff_cmap: bool = False) -> tuple:
        """
        Render density isosurfaces at specified levels.
        
        This creates 3D surfaces where the density equals the specified values,
        using the marching cubes algorithm.
        
        Args:
            rho: Density data array
            levels: List of density values to render as isosurfaces
            cmap: Colormap name ('plasma', 'viridis', 'real', etc.)
            vmin, vmax: Color scale limits
            show_cbar: Boolean to show colorbar
            opacity: Surface opacity (0.0 to 1.0)
            diff_cmap: Whether to use different colorbar position (for jets)
            
        Returns:
            Tuple of (actor, color_transfer_function) or just actor if no colormap
            
        Example:
            actor = density_renderer.render_density_levels(
                rho=rho_data,
                levels=[1e-3, 1e-4, 1e-5],
                cmap='plasma',
                opacity=0.5
            )
        """
        logger.info(f"Rendering density levels: {levels}")
        
        # Convert density to VTK format
        vdata,X = RhoData_to_vtkImageData(rho)
        vdata_array = vdata.GetPointData().GetScalars()
        vdata_range = vdata_array.GetRange()
        logger.debug(f"VTK data range: {vdata_range}")
        
        # Setup marching cubes algorithm
        dmc = vtk.vtkMarchingCubes()
        vtkConnectDataInput(vdata, dmc)
        
        # Set density levels to extract
        for i, level in enumerate(levels):
            dmc.SetValue(i, level)
        dmc.Update()
        
        # Get output mesh
        output = dmc.GetOutput()
        logger.debug(f"Marching cubes generated: {output.GetNumberOfPoints()} points, "
                    f"{output.GetNumberOfCells()} cells")
        
        # Setup mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        vtkConnectOutputInput(dmc, mapper)
        actor = vtk.vtkActor()
        
        # Configure coloring
        if cmap is None:
            # Single color mode
            mapper.ScalarVisibilityOff()
            actor.GetProperty().SetColor(*to_rgb('g'))
            rval = actor
        elif cmap == "real":
            # Custom plasma+magma colormap
            vmin, vmax = data_range(rho, vmin=vmin, vmax=vmax)
            vk = np.linspace(vmin, vmax, 256)
            top = mpl.colormaps['plasma']
            bottom = mpl.colormaps['magma']
            clrs = np.vstack((
                top(np.linspace(0, 1, 256))[:128, :3],
                bottom(np.linspace(0, 1, 256))[128:, :3]
            ))
            ctf = make_vtkColorTransferFunction(vk, clrs)
            
            if show_cbar:
                barx = 0.05 if diff_cmap else 0.9
                title = None
                viz.color_bar(ctf, renderer=self.renderer, barx=barx, title=title)
            
            mapper.ScalarVisibilityOn()
            mapper.SetLookupTable(ctf)
            rval = (actor, ctf)
        else:
            # Standard matplotlib colormap
            vmin, vmax = data_range(rho, vmin=vmin, vmax=vmax)
            ctf = get_vtkColorTransferFunction(cmap, vmin=vmin, vmax=vmax)
            
            if show_cbar:
                barx = 0.05 if diff_cmap else 0.9
                title = None
                viz.color_bar(ctf, renderer=self.renderer, barx=barx, title=title)
            
            mapper.ScalarVisibilityOn()
            mapper.SetLookupTable(ctf)
            rval = (actor, ctf)
        
        # Set opacity and add to scene
        actor.GetProperty().SetOpacity(opacity)
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)
        
        logger.info(f"Added density actor with opacity {opacity}")
        return rval, X


class BlackHoleRenderer:
    """Handles black hole and spin visualization."""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize black hole renderer.
        
        Args:
            renderer: VTK renderer to add actors to
            
        Example:
            bh_renderer = BlackHoleRenderer(vtk_renderer)
        """
        self.renderer = renderer
        logger.debug("Initialized BlackHoleRenderer")
    
    def render_black_holes(self, data_dir: str, rho_file: str,
                          X: int, show_spin: bool = False):
        """
        Render black hole horizons and optionally spin arrows.
        
        Reads black hole position and radius data from the simulation
        and renders them as spheres. Optionally adds arrows to show
        spin direction.
        
        Args:
            data_dir: Path to simulation data directory
            rho_file: Name of density file (used to extract time/distance)
            X: Global grid size
            show_spin: Whether to show spin direction arrows
            
        Example:
            bh_renderer.render_black_holes(
                data_dir='/home/user/HUK_data/',
                rho_file='rho_t100M_d100M',
                show_spin=True
            )
        """
        logger.info("Rendering black holes")
        
        # Load simulation directory
        with suppress_stdout():
            sd = SimDir(data_dir)
        
        bbh = sd.ahoriz.horizons
        if len(bbh) < 2:
            logger.warning("Less than 2 black holes found")
            return
        
        bh0, bh1 = bbh[0], bbh[1]
        
        # Extract time and distance from filename
        time = float(rho_file.split("_")[1][1:-1])
        distance = float(rho_file.split("_")[2][1:-1])
        logger.info(f"Black hole data at time {time} M, distance {distance} M")
        
        # Find closest time index
        tms = bh0.ah.rmean.t
        idx = np.argmin(np.abs(tms - time))
        
        # Get conversion factor (assuming X is global grid size)
        conversion = X // 2 * (1 / distance) * KM_CU
        
        # Get black hole properties
        bh0_r = bh0.ah.rmean.y[idx] * conversion
        bh1_r = bh1.ah.rmean.y[idx] * conversion
        bh0_pos = np.array([
            bh0.ah.pos_x.y[idx] * conversion,
            bh0.ah.pos_y.y[idx] * conversion,
            bh0.ah.pos_z.y[idx] * conversion
        ])
        bh1_pos = np.array([
            bh1.ah.pos_x.y[idx] * conversion,
            bh1.ah.pos_y.y[idx] * conversion,
            bh1.ah.pos_z.y[idx] * conversion
        ])
        
        logger.info(f"BH0: r={bh0_r:.2f}, pos={bh0_pos}")
        logger.info(f"BH1: r={bh1_r:.2f}, pos={bh1_pos}")
        
        # Create sphere representations
        self._create_sphere(bh0_pos, bh0_r)
        self._create_sphere(bh1_pos, bh1_r)
        
        logger.info("Added black hole actors")
        
        # Add spin arrows if requested
        if show_spin:
            self._add_spin_arrows(sd, time, bh0_pos, bh1_pos)
    
    def _create_sphere(self, position: np.ndarray, radius: float):
        """
        Create a sphere actor at given position.
        
        Args:
            position: [x, y, z] position array
            radius: Sphere radius
        """
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*position)
        sphere.SetRadius(radius)
        sphere.SetPhiResolution(100)
        sphere.SetThetaResolution(100)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*to_rgb('k'))
        
        self.renderer.AddActor(actor)
        logger.debug(f"Created sphere at {position} with radius {radius}")
    
    def _add_spin_arrows(self, sd: SimDir, time: float,
                        bh0_pos: np.ndarray, bh1_pos: np.ndarray):
        """
        Add spin direction arrows to black holes.
        
        Args:
            sd: SimDir object with scalar data
            time: Simulation time
            bh0_pos: Position of first black hole
            bh1_pos: Position of second black hole
        """
        logger.info("Adding spin arrows")
        
        # Find closest time in spin data
        stms = sd.ts.scalar['qlm_coordspinx[0]'].t
        sidx = np.argmin(np.abs(stms - time))
        
        # Get spin components for both black holes
        x0 = sd.ts.scalar['qlm_coordspinx[0]'].y[sidx]
        x1 = sd.ts.scalar['qlm_coordspinx[1]'].y[sidx]
        y0 = sd.ts.scalar['qlm_coordspiny[0]'].y[sidx]
        y1 = sd.ts.scalar['qlm_coordspiny[1]'].y[sidx]
        z0 = sd.ts.scalar['qlm_coordspinz[0]'].y[sidx]
        z1 = sd.ts.scalar['qlm_coordspinz[1]'].y[sidx]
        
        r0 = np.array([x0, y0, z0])
        r1 = np.array([x1, y1, z1])
        
        logger.info(f"BH0 spin: {r0}")
        logger.info(f"BH1 spin: {r1}")
        
        # Normalize directions
        d0 = r0 / np.linalg.norm(r0)
        d1 = r1 / np.linalg.norm(r1)
        
        # Create arrows
        arrow0 = self._create_arrow(bh0_pos, d0, scale_factor=10)
        arrow1 = self._create_arrow(bh1_pos, d1, scale_factor=10)
        
        self.renderer.AddActor(arrow0)
        self.renderer.AddActor(arrow1)
        
        logger.info("Added spin arrow actors")
    
    def _create_arrow(self, start_point: np.ndarray, direction: np.ndarray,
                     scale_factor: float = 1.0) -> vtk.vtkActor:
        """
        Create an arrow actor pointing in a specific direction.
        
        Args:
            start_point: [x, y, z] starting position
            direction: [x, y, z] direction vector
            scale_factor: Scale factor for arrow size
            
        Returns:
            VTK actor representing the arrow
        """
        arrow_source = vtk.vtkArrowSource()
        
        transform = vtk.vtkTransform()
        transform.Translate(start_point)
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Create rotation matrix to align arrow with direction
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
            y_axis = np.array([0.0, 1.0, 0.0])
        else:
            y_axis = np.cross(z_axis, direction)
            y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, direction)
        
        matrix = vtk.vtkMatrix4x4()
        for i, axis in enumerate([x_axis, y_axis, direction]):
            for j in range(3):
                matrix.SetElement(i, j, axis[j])
        
        transform.Concatenate(matrix)
        transform.Scale(scale_factor, scale_factor, scale_factor)
        
        # Apply transform
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(transform)
        tfilter.SetInputConnection(arrow_source.GetOutputPort())
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tfilter.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*to_rgb('c'))
        
        return actor


class FieldLineRenderer:
    """Handles magnetic field line rendering."""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize field line renderer.
        
        Args:
            renderer: VTK renderer to add actors to
            
        Example:
            fl_renderer = FieldLineRenderer(vtk_renderer)
        """
        self.renderer = renderer
        logger.debug("Initialized FieldLineRenderer")
    
    def render_field_lines(self, curves: list, scalar: np.ndarray,
                          weight: np.ndarray, vmax: float,
                          magnitudes: float, tradius: float,
                          cam_phi: float, cmap: str = 'plasma_r'):
        """
        Render magnetic field lines as tubes.
        
        Args:
            curves: Field line curve data (list of coordinate arrays)
            scalar: Scalar values along curves (e.g., field strength)
            weight: Curve weights
            vmax: Maximum scalar value for color scale
            magnitudes: Number of orders of magnitude for log scaling
            tradius: Tube radius
            cam_phi: Camera phi angle (for length scale indicator)
            cmap: Colormap name
            
        Example:
            fl_renderer.render_field_lines(
                curves=field_lines,
                scalar=field_strength,
                weight=weights,
                vmax=1e15,
                magnitudes=2.0,
                tradius=1.0,
                cam_phi=-0.5236,
                cmap='plasma_r'
            )
        """
        logger.info("Rendering field lines")
        
        # Calculate logarithmic data range
        logsc, lgmin, lgmax, numlbl = log_data_range(vmax, magnitudes, scalar)
        logger.info(f"Field line scalar range: {lgmin:.2f} to {lgmax:.2f}")
        
        # Render field lines as tubes with color
        fls, ctf = viz.field_lines(
            curves, scalar=logsc, vmin=lgmin, vmax=lgmax,
            cmap=cmap, tradius=tradius, renderer=self.renderer
        )
        
        logger.info(f"Added {len(curves)} field line tubes")
        
        # Add length scale indicator
        self._add_length_scale(cam_phi)
    
    def _add_length_scale(self, cam_phi: float):
        """
        Add a length scale indicator to the scene.
        
        Args:
            cam_phi: Camera phi angle for positioning
        """
        lensc_r = 200 * KM_CU
        lensc = 50 * KM_CU
        
        lscx = lensc_r * cos(cam_phi - pi / 2)
        lscy = lensc_r * sin(cam_phi - pi / 2)
        lscz = lensc
        
        lensc_curves = [np.array([[lscx] * 3, [lscy] * 3, [-lscz, 0, lscz]])]
        viz.tubes(lensc_curves, color='k', fixed_radius=0.001 * KM_CU,
                 renderer=self.renderer)
        
        logger.debug("Added length scale indicator")


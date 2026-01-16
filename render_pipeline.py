"""
Main rendering pipeline that orchestrates the entire rendering process.
"""
import logging
import numpy as np
import vtk
from pathlib import Path
from postcactus import viz_vtk as viz
from postcactus.unit_abbrev import *

from config import RenderConfig
from data_loader import load_simulation_data, SimulationData
from scene_builder import DensityRenderer, BlackHoleRenderer, FieldLineRenderer
from utils import write_png, grid

logger = logging.getLogger(__name__)

class RenderPipeline:
    """
    Main rendering pipeline that coordinates all rendering operations.
    
    This class provides a clean, chainable interface for loading data,
    setting up scenes, and rendering output.
    """
    
    def __init__(self, config: RenderConfig):
        """
        Initialize render pipeline.
        
        Args:
            config: Configuration object containing all rendering settings
            
        Example:
            config = RenderConfig.from_json('config.json')
            pipeline = RenderPipeline(config)
        """
        self.config = config
        self.data = None
        self.data_size = None
        self.renderer = None
        self.window = None
        
        logger.info("Initialized RenderPipeline")
        logger.debug(f"Configuration:\n{config}")
    
    def load_data(self) -> 'RenderPipeline':
        """
        Load all required simulation data.
        
        This loads density, horizon, and optionally field line data
        based on the configuration settings.
        
        Returns:
            Self for method chaining
            
        Example:
            pipeline.load_data()
        """
        logger.info("Loading simulation data")
        
        self.data = load_simulation_data(
            data_dir=self.config.paths.data_dir,
            rho_file=self.config.paths.rho_file,
            bfield_file=self.config.paths.bfield_file,
            smallb2_file=self.config.paths.smallb2_file,
            nkeep=self.config.field_lines.nkeep,
            rcut=self.config.field_lines.rcut * KM_CU,
            load_field_lines=self.config.field_lines.show,
            load_smallb2=self.config.jets.smallb2_rho
        )
        
        logger.info("Data loading complete")
        return self
    
    def setup_scene(self) -> 'RenderPipeline':
        """
        Create and configure the rendering scene.
        
        This sets up the renderer and adds all requested visual elements:
        - Density isosurfaces
        - Jets (if enabled)
        - Black holes (if enabled)
        - Field lines (if enabled)
        - Coordinate grid (if enabled)
        - Apparent horizon patches
        
        Returns:
            Self for method chaining
            
        Example:
            pipeline.setup_scene()
        """
        logger.info("Setting up scene")
        
        # Create renderer with black background
        self.renderer = viz.make_renderer(bgcolor='k')
        
        # Render density isosurfaces
        self._render_density()
        
        # Render jets if enabled
        if self.config.jets.show:
            self._render_jets()
        
        # Render black holes if enabled
        if self.config.black_holes.show:
            self._render_black_holes()
        
        # Render field lines if enabled
        if self.config.field_lines.show and self.data.has_field_lines():
            self._render_/field_lines()
        
        # Add coordinate grid if enabled
        if self.config.grid.show:
            self._add_grid()
        
        # Add apparent horizon patches if available
        if self.data.has_horizon():
            logger.info("Adding apparent horizon patches")
            viz.show_ah_patches(self.data.ah, color='g', renderer=self.renderer)
        
        logger.info("Scene setup complete")
        return self
    
    def _render_density(self):
        """Render density isosurfaces."""
        logger.info("Rendering density isosurfaces")
        
        density_renderer = DensityRenderer(self.renderer)
        density_levels = self.config.density.get_levels()
        opacity = self.config.density.opacity

        print(type(opacity))
        
        if isinstance(opacity, float) or isinstance(opacity, int):
            # Single opacity for all levels
            logger.info(f"Using single opacity: {opacity}")
            for levels in density_levels:
                rval, X = density_renderer.render_density_levels(
                            rho=self.data.rho,
                            levels=levels,
                            cmap=self.config.density.colormap,
                            vmin=self.config.density.vmin,
                            vmax=self.config.density.vmax,
                            show_cbar=self.config.density.show_cbar,
                            opacity=opacity
                          )
            self.data_size=X
        else:
            # List of opacities for different level sets
            logger.info(f"Using opacity list: {opacity}")
            if len(opacity) != len(density_levels):
                logger.warning(f"Opacity list length ({len(opacity)}) doesn't match "
                             f"density levels ({len(density_levels)})")
            for levels, op in zip(density_levels, opacity):
                density_renderer.render_density_levels(
                    rho=self.data.rho,
                    levels=levels,
                    cmap=self.config.density.colormap,
                    vmin=self.config.density.vmin,
                    vmax=self.config.density.vmax,
                    show_cbar=self.config.density.show_cbar,
                    opacity=op
                )
    
    def _render_jets(self):
        """Render jet isosurfaces."""
        logger.info("Rendering jets")
        
        density_renderer = DensityRenderer(self.renderer)
        jet_levels = self.config.jets.get_levels()
        
        # Use different colorbar position if jet colormap differs from density
        diff_cmap = (self.config.density.colormap != self.config.jets.colormap)

        if self.config.jets.smallb2_rho:
            data = np.divide(self.data.smallb2, self.data.rho)
        else:
            data = self.data.rho
        
        density_renderer.render_density_levels(
            rho=data,
            levels=jet_levels,
            cmap=self.config.jets.colormap,
            vmin=self.config.jets.vmin,
            vmax=self.config.jets.vmax,
            show_cbar=self.config.jets.show_cbar,
            opacity=self.config.jets.opacity,
            diff_cmap=diff_cmap
        )
    
    def _render_black_holes(self):
        """Render black holes and optionally spin arrows."""
        logger.info("Rendering black holes")
        
        bh_renderer = BlackHoleRenderer(self.renderer)
        bh_renderer.render_black_holes(
            data_dir=self.config.paths.scalar_data_dir,
            rho_file=self.config.paths.rho_file,
            X=self.data_size,
            show_spin=self.config.black_holes.show_spin
        )
    
    def _render_field_lines(self):
        """Render magnetic field lines."""
        logger.info("Rendering field lines")
        
        fl_renderer = FieldLineRenderer(self.renderer)
        fl_renderer.render_field_lines(
            curves=self.data.curves,
            scalar=self.data.scalar,
            weight=self.data.weight,
            vmax=self.config.field_lines.vmax_bfield,
            magnitudes=self.config.field_lines.magnitudes,
            tradius=self.config.field_lines.tradius * KM_CU,
            cam_phi=self.config.camera.phi,
            cmap='plasma_r'
        )
    
    def _add_grid(self):
        """Add coordinate grid to scene."""
        logger.info("Adding coordinate grid")
        
        # Import global grid size X from automated_render
        # (This is a temporary solution - ideally X should be in config)
        grid(
            hrho=self.config.paths.rho_file,
            grid_size=self.config.grid.size,
            spacing=self.config.grid.spacing,
            renderer=self.renderer,
            X=self.data_size
        )
    
    def setup_camera(self) -> 'RenderPipeline':
        """
        Configure camera position.
        
        Sets the camera to the position specified in the configuration.
        
        Returns:
            Self for method chaining
            
        Example:
            pipeline.setup_camera()
        """
        r, theta, phi = self.config.camera.as_tuple()
        logger.info(f"Setting camera at (r={r}, theta={theta:.3f}, phi={phi:.3f})")
        
        viz.set_camera(self.renderer, r, theta, phi)
        return self
    
    def setup_window(self) -> 'RenderPipeline':
        """
        Create render window and configure it.
        
        This creates the VTK render window, adds the renderer,
        optionally adds a background image, and adds time text.
        
        Returns:
            Self for method chaining
            
        Example:
            pipeline.setup_window()
        """
        logger.info("Setting up render window")
        
        # Create window (offscreen if not interactive)
        self.window = viz.RenderWindow(
            size=self.config.output.fig_size,
            offscreen=not self.config.output.interactive
        )
        
        # Add renderer to window
        self.window.add_renderer(self.renderer, viewport=(0, 0, 1.0, 1.0))
        
        # Add time text overlay
        time = float(self.config.paths.rho_file.split("_")[1][1:-1])
        viz.text(f'{time:.2e} M', posx=0.7, color='w', renderer=self.renderer)
        
        # Add background image if specified
        if self.config.output.background_image:
            self._add_background_image()
        
        logger.info("Render window setup complete")
        return self
    
    def _add_background_image(self):
        """Add background image to render window."""
        try:
            import cv2 as cv
            from vtkplotlib.image_io import vtkimagedata_from_array, vtkimagedata_to_array
        except ImportError:
            logger.error("opencv-python and vtkplotlib required for background images")
            return
        
        bg_path = self.config.output.background_image
        logger.info(f"Adding background image: {bg_path}")
        
        renwin = self.window.renderWin
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(bg_path)
        reader.Update()
        image_vtk_data = reader.GetOutput()
        
        # Convert and crop image
        image_np_data = vtkimagedata_to_array(image_vtk_data)
        midx, midy = len(image_np_data), len(image_np_data[0])
        scale = 4
        size = self.config.output.fig_size
        
        image_data = vtkimagedata_from_array(
            image_np_data[
                midx - (scale * size[1]) // 2:midx + (scale * size[1]) // 2,
                midy - (scale * size[0]) // 2:midy + (scale * size[0]) // 2
            ]
        )
        
        # Create background actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(image_data)
        
        # Create background renderer
        bg_rend = vtk.vtkRenderer()
        bg_rend.AddActor(image_actor)
        
        # Layer setup
        self.renderer.SetLayer(1)
        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(bg_rend)
        
        # Setup background camera
        origin = image_data.GetOrigin()
        spacing = image_data.GetSpacing()
        extent = image_data.GetExtent()
        
        camera = bg_rend.GetActiveCamera()
        camera.ParallelProjectionOn()
        
        xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
        yd = (extent[3] - extent[2] + 1) * spacing[1]
        
        d = camera.GetDistance()
        camera.SetParallelScale(0.5 * yd)
        camera.SetFocalPoint(xc, yc, 0.0)
        camera.SetPosition(xc, yc, d)
        
        logger.info("Background image added successfully")
    
    def render(self, output_path: str = None) -> 'RenderPipeline':
        """
        Render the scene to file or display interactively.
        
        Args:
            output_path: Output file path (uses config if not provided)
            
        Returns:
            Self for method chaining
            
        Example:
            pipeline.render('output/my_figure')
        """
        if output_path is None:
            output_path = self.config.output.fig_name
        
        logger.info(f"Rendering to {output_path}")
        
        if self.config.output.interactive:
            logger.info("Opening interactive window")
            self.window.show_interactive(screensh_file=output_path)
        else:
            write_png(
                self.window,
                output_path,
                scale=self.config.output.scale
            )
            logger.info(f"Saved to {output_path}.png")
        
        return self
    
    def render_movie(self, movie_function) -> 'RenderPipeline':
        """
        Render a movie using provided movie function.
        
        Args:
            movie_function: Function with signature:
                           func(window, fname, camera, rho_file, scale)
            
        Returns:
            Self for method chaining
            
        Example:
            def rotate_movie(win, fname, cam, rho_file, scale):
                # Render rotating frames
                ...
            
            pipeline.render_movie(rotate_movie)
        """
        logger.info("Rendering movie")
        
        movie_function(
            self.window,
            self.config.output.fig_name,
            self.config.camera.as_tuple(),
            self.config.paths.rho_file,
            self.config.output.scale
        )
        
        logger.info("Movie rendering complete")
        return self
    
    def run(self):
        """
        Execute the complete rendering pipeline.
        
        This is a convenience method that chains all steps together:
        - Load data
        - Setup scene
        - Setup camera
        - Setup window
        - Render output
        
        Example:
            config = RenderConfig.from_json('config.json')
            pipeline = RenderPipeline(config)
            pipeline.run()
        """
        logger.info("Starting render pipeline")
        
        self.load_data()
        self.setup_scene()
        self.setup_camera()
        self.setup_window()
        
        if self.config.output.movie:
            logger.warning("Movie rendering requires movie_function parameter")
            logger.warning("Use pipeline.render_movie(movie_func) instead")
        else:
            self.render()
        
        logger.info("Render pipeline complete")
    
    def cleanup(self):
        """
        Clean up resources.
        
        Call this when done with the pipeline to free VTK resources.
        """
        if self.window:
            self.window.renderWin.Finalize()
        logger.info("Pipeline cleanup complete")


def create_default_config() -> RenderConfig:
    """
    Create a default configuration and save to JSON.
    
    Returns:
        Default RenderConfig instance
        
    Example:
        config = create_default_config()
        # Creates 'default_config.json' in current directory
    """
    config = RenderConfig()
    config.to_json('default_config.json')
    logger.info("Created default_config.json")
    return config

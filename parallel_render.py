"""
Parallel rendering module for distributing single frame rendering across multiple processors.

This module uses MPI to split the rendering workload spatially with domain decomposition,
where each processor loads only its portion of the data to minimize memory usage.
"""
import logging
import numpy as np
import h5py
from typing import Tuple, Optional
from pathlib import Path

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    
import vtk
from postcactus import viz_vtk as viz
from postcactus.unit_abbrev import *

from config import RenderConfig
from render_pipeline import RenderPipeline
from data_loader import SimulationData
from utils import write_png

logger = logging.getLogger(__name__)


class DomainDecomposition:
    """
    Handles spatial domain decomposition for parallel data loading.
    
    Each MPI rank loads and processes only its portion of the 3D domain.
    """
    
    def __init__(self, domain_shape: Tuple[int, int, int], 
                 decomposition: Tuple[int, int, int] = None):
        """
        Initialize domain decomposition.
        
        Args:
            domain_shape: Full domain shape (nx, ny, nz)
            decomposition: Number of subdivisions in each direction (px, py, pz)
                          If None, automatically determined from MPI size
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.domain_shape = domain_shape
        
        # Determine decomposition
        if decomposition is None:
            self.decomposition = self._auto_decompose()
        else:
            self.decomposition = decomposition
            
        # Validate decomposition
        px, py, pz = self.decomposition
        if px * py * pz != self.size:
            raise ValueError(
                f"Decomposition {decomposition} doesn't match MPI size {self.size}"
            )
        
        # Calculate this rank's subdomain
        self.subdomain_bounds = self._calculate_subdomain()
        self.physical_bounds = None  # Set later when we know physical coordinates
        
        logger.info(f"[Rank {self.rank}] Subdomain indices: {self.subdomain_bounds}")
    
    def _auto_decompose(self) -> Tuple[int, int, int]:
        """
        Automatically determine domain decomposition.
        
        Tries to create roughly cubic subdomains.
        """
        size = self.size
        
        # Try to factorize into 3 roughly equal factors
        best_decomp = (size, 1, 1)
        best_ratio = float('inf')
        
        for pz in range(1, int(size**(1/3)) + 2):
            if size % pz != 0:
                continue
            remaining = size // pz
            for py in range(1, int(remaining**0.5) + 2):
                if remaining % py != 0:
                    continue
                px = remaining // py
                
                # Calculate how cubic this decomposition is
                ratio = max(px, py, pz) / min(px, py, pz)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_decomp = (px, py, pz)
        
        logger.info(f"Auto-decomposition: {best_decomp} (ratio: {best_ratio:.2f})")
        return best_decomp
    
    def _calculate_subdomain(self) -> Tuple[slice, slice, slice]:
        """
        Calculate subdomain bounds for this rank.
        
        Returns:
            Tuple of slices (x_slice, y_slice, z_slice)
        """
        px, py, pz = self.decomposition
        nx, ny, nz = self.domain_shape
        
        # Calculate rank indices in 3D grid
        rank_z = self.rank % pz
        rank_y = (self.rank // pz) % py
        rank_x = self.rank // (pz * py)
        
        # Calculate subdomain sizes
        nx_sub = nx // px
        ny_sub = ny // py
        nz_sub = nz // pz
        
        # Handle remainders
        nx_rem = nx % px
        ny_rem = ny % py
        nz_rem = nz % pz
        
        # Calculate bounds
        x_start = rank_x * nx_sub + min(rank_x, nx_rem)
        x_end = x_start + nx_sub + (1 if rank_x < nx_rem else 0)
        
        y_start = rank_y * ny_sub + min(rank_y, ny_rem)
        y_end = y_start + ny_sub + (1 if rank_y < ny_rem else 0)
        
        z_start = rank_z * nz_sub + min(rank_z, nz_rem)
        z_end = z_start + nz_sub + (1 if rank_z < nz_rem else 0)
        
        return (
            slice(x_start, x_end),
            slice(y_start, y_end),
            slice(z_start, z_end)
        )
    
    def calculate_physical_bounds(self, extent: float) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate physical coordinate bounds for this rank's subdomain.
        
        Args:
            extent: Physical extent of domain (from -extent to +extent)
            
        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax) in physical coordinates
        """
        nx, ny, nz = self.domain_shape
        x_slice, y_slice, z_slice = self.subdomain_bounds
        
        # Convert grid indices to physical coordinates
        x_min = -extent + (x_slice.start / nx) * (2 * extent)
        x_max = -extent + (x_slice.stop / nx) * (2 * extent)
        
        y_min = -extent + (y_slice.start / ny) * (2 * extent)
        y_max = -extent + (y_slice.stop / ny) * (2 * extent)
        
        z_min = -extent + (z_slice.start / nz) * (2 * extent)
        z_max = -extent + (z_slice.stop / nz) * (2 * extent)
        
        self.physical_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
        logger.info(f"[Rank {self.rank}] Physical bounds: "
                   f"X=[{x_min:.2f}, {x_max:.2f}], "
                   f"Y=[{y_min:.2f}, {y_max:.2f}], "
                   f"Z=[{z_min:.2f}, {z_max:.2f}]")
        
        return self.physical_bounds
    
    def load_subdomain_h5(self, filename: str, dataset_name: str = None) -> np.ndarray:
        """
        Load only this rank's subdomain from HDF5 file.
        
        Args:
            filename: HDF5 file path
            dataset_name: Dataset name (if None, uses first dataset)
            
        Returns:
            Subdomain data array
        """
        try:
            # Try parallel HDF5 first
            return self._load_parallel_h5(filename, dataset_name)
        except (ImportError, AttributeError, OSError) as e:
            logger.warning(f"[Rank {self.rank}] Parallel HDF5 not available ({e}), "
                          "using sequential read")
            return self._load_sequential_h5(filename, dataset_name)
    
    def _load_parallel_h5(self, filename: str, dataset_name: str) -> np.ndarray:
        """Load using parallel HDF5."""
        with h5py.File(filename, 'r', driver='mpio', comm=self.comm) as f:
            if dataset_name is None:
                dataset_name = list(f.keys())[0]
            
            dset = f[dataset_name]
            
            # Read only this rank's subdomain
            x_slice, y_slice, z_slice = self.subdomain_bounds
            data = dset[x_slice, y_slice, z_slice]
            
            logger.info(f"[Rank {self.rank}] Loaded subdomain shape {data.shape} "
                       f"from parallel HDF5")
            return data
    
    def _load_sequential_h5(self, filename: str, dataset_name: str) -> np.ndarray:
        """Load using sequential reads (one rank at a time)."""
        data = None
        
        # Each rank reads in turn to avoid file access conflicts
        for i in range(self.size):
            if i == self.rank:
                with h5py.File(filename, 'r') as f:
                    if dataset_name is None:
                        dataset_name = list(f.keys())[0]
                    
                    dset = f[dataset_name]
                    x_slice, y_slice, z_slice = self.subdomain_bounds
                    data = dset[x_slice, y_slice, z_slice]
                    
                    logger.info(f"[Rank {self.rank}] Loaded subdomain shape {data.shape}")
            
            # Synchronize to prevent file access conflicts
            self.comm.Barrier()
        
        return data


class ParallelRenderPipeline(RenderPipeline):
    """
    Extended rendering pipeline with domain decomposition support.
    
    This class extends RenderPipeline to support memory-efficient parallel rendering:
    1. Each processor loads only its portion of the volume data
    2. Rendering is done independently with spatial clipping
    3. Results are composited to create the final image
    """
    
    def __init__(self, config: RenderConfig, comm=None, use_decomposition: bool = True):
        """
        Initialize parallel render pipeline.
        
        Args:
            config: Configuration object
            comm: MPI communicator (defaults to MPI.COMM_WORLD)
            use_decomposition: If True, use domain decomposition to save memory
        """
        super().__init__(config)
        
        if HAS_MPI:
            self.comm = comm if comm is not None else MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        self.is_parallel = self.size > 1
        self.use_decomposition = use_decomposition and self.is_parallel
        self.decomp = None
        
        logger.info(f"Initialized ParallelRenderPipeline: rank {self.rank}/{self.size}, "
                   f"decomposition={'enabled' if self.use_decomposition else 'disabled'}")
    
    def load_data(self) -> 'ParallelRenderPipeline':
        """
        Load simulation data with domain decomposition.
        
        If decomposition is enabled, each rank loads only its subdomain.
        Otherwise, all ranks load the full dataset.
        """
        if self.use_decomposition:
            logger.info(f"[Rank {self.rank}] Loading data with domain decomposition")
            self._load_data_decomposed()
        else:
            logger.info(f"[Rank {self.rank}] Loading full data on all ranks")
            super().load_data()
        
        return self
    
    def _load_data_decomposed(self):
        """
        Load data with domain decomposition - each rank loads only its subdomain.
        """
        # First, determine full domain shape (only rank 0 opens file)
        rho_file = self.config.paths.rho_file
        if not rho_file.endswith(".h5"):
            rho_file = rho_file + ".h5"
        filepath = Path(self.config.paths.data_dir) / rho_file
        
        if self.rank == 0:
            with h5py.File(filepath, 'r') as f:
                dataset_name = list(f.keys())[0]
                domain_shape = f[dataset_name].shape
                logger.info(f"[Rank 0] Detected domain shape: {domain_shape}")
        else:
            domain_shape = None
        
        # Broadcast domain shape to all ranks
        domain_shape = self.comm.bcast(domain_shape, root=0)
        
        # Create domain decomposition
        self.decomp = DomainDecomposition(domain_shape)
        
        # Load subdomain density data
        logger.info(f"[Rank {self.rank}] Loading density subdomain")
        rho_subdomain = self.decomp.load_subdomain_h5(str(filepath))
        
        # Load other data (horizon, field lines) - only on rank 0 to save memory
        # These are typically much smaller than the volume data
        if self.rank == 0:
            logger.info("[Rank 0] Loading horizon and auxiliary data")
            
            # Extract time from filename
            try:
                time_str = self.config.paths.rho_file.split("_")[1][1:-1]
                time = float(time_str)
            except (IndexError, ValueError):
                time = 0.0
            
            # Load horizon
            from data_loader import DataLoader
            loader = DataLoader(self.config.paths.data_dir)
            ah = loader.load_horizon(time)
            
            # Load field lines if requested
            curves, scalar, weight = None, None, None
            if self.config.field_lines.show and self.config.paths.bfield_file:
                curves, scalar, weight, time = loader.load_field_lines(
                    self.config.paths.bfield_file,
                    self.config.field_lines.nkeep,
                    self.config.field_lines.rcut * KM_CU
                )
            
            # Load B**2 if requested
            smallb2 = None
            if self.config.jets.smallb2_rho and self.config.paths.smallb2_file:
                # For jets, we need full volume - load subdomain instead
                smallb2_file = self.config.paths.smallb2_file
                if not smallb2_file.endswith(".h5"):
                    smallb2_file = smallb2_file + ".h5"
                smallb2_path = Path(self.config.paths.data_dir) / smallb2_file
                smallb2_subdomain = self.decomp.load_subdomain_h5(str(smallb2_path))
        else:
            ah = None
            curves, scalar, weight = None, None, None
            time = 0.0
            smallb2_subdomain = None
        
        # Broadcast time to all ranks
        time = self.comm.bcast(time, root=0)
        
        # Load smallb2 subdomain on all ranks if needed for jets
        if self.config.jets.smallb2_rho and self.config.paths.smallb2_file:
            if self.rank != 0:
                smallb2_file = self.config.paths.smallb2_file
                if not smallb2_file.endswith(".h5"):
                    smallb2_file = smallb2_file + ".h5"
                smallb2_path = Path(self.config.paths.data_dir) / smallb2_file
                smallb2_subdomain = self.decomp.load_subdomain_h5(str(smallb2_path))
            
            smallb2 = smallb2_subdomain
        else:
            smallb2 = None
        
        # Create SimulationData with subdomain
        self.data = SimulationData(
            rho=rho_subdomain,
            ah=ah,
            smallb2=smallb2,
            curves=curves,
            scalar=scalar,
            weight=weight,
            time=time
        )
        
        logger.info(f"[Rank {self.rank}] Data loading complete, "
                   f"subdomain shape: {rho_subdomain.shape}")
    
    def setup_scene(self) -> 'ParallelRenderPipeline':
        """
        Setup scene with spatial decomposition for parallel rendering.
        
        Each processor renders only its subdomain.
        """
        logger.info(f"[Rank {self.rank}] Setting up scene")
        
        # Create renderer
        self.renderer = viz.make_renderer(bgcolor='k')
        
        # Render density with domain decomposition
        self._render_density_decomposed()
        
        # Render jets if enabled (also needs decomposition)
        if self.config.jets.show:
            self._render_jets_decomposed()
        
        # Other elements - only on rank 0 to avoid duplication
        if self.rank == 0:
            if self.config.black_holes.show:
                self._render_black_holes()
            
            if self.config.field_lines.show and self.data.has_field_lines():
                self._render_field_lines()
            
            if self.config.grid.show:
                self._add_grid()
            
            if self.data.has_horizon():
                logger.info("[Rank 0] Adding apparent horizon patches")
                viz.show_ah_patches(self.data.ah, color='g', renderer=self.renderer)
        
        logger.info(f"[Rank {self.rank}] Scene setup complete")
        return self
    
    def _render_density_decomposed(self):
        """
        Render density subdomain on each processor.
        
        Uses VTK's spatial clipping to ensure each rank only shows its portion.
        """
        logger.info(f"[Rank {self.rank}] Rendering density subdomain")
        
        from scene_builder import DensityRenderer
        from utils import RhoData_to_vtkImageData
        
        # Convert subdomain to VTK format
        vdata, X = RhoData_to_vtkImageData(self.data.rho)
        self.data_size = X
        
        # Calculate physical bounds for clipping
        if self.use_decomposition:
            extent = X / 4  # From RhoData_to_vtkImageData
            physical_bounds = self.decomp.calculate_physical_bounds(extent)
        
        # Render density
        density_renderer = DensityRenderer(self.renderer)
        density_levels = self.config.density.get_levels()
        opacity = self.config.density.opacity
        
        if isinstance(opacity, (float, int)):
            for levels in density_levels:
                rval, _ = density_renderer.render_density_levels(
                    rho=self.data.rho,
                    levels=levels,
                    cmap=self.config.density.colormap,
                    vmin=self.config.density.vmin,
                    vmax=self.config.density.vmax,
                    show_cbar=self.config.density.show_cbar and self.rank == 0,
                    opacity=opacity
                )
                
                # Apply spatial clipping to show only this rank's subdomain
                if self.use_decomposition:
                    self._apply_spatial_clip(rval, physical_bounds)
        else:
            for levels, op in zip(density_levels, opacity):
                rval, _ = density_renderer.render_density_levels(
                    rho=self.data.rho,
                    levels=levels,
                    cmap=self.config.density.colormap,
                    vmin=self.config.density.vmin,
                    vmax=self.config.density.vmax,
                    show_cbar=self.config.density.show_cbar and self.rank == 0,
                    opacity=op
                )
                
                if self.use_decomposition:
                    self._apply_spatial_clip(rval, physical_bounds)
    
    def _render_jets_decomposed(self):
        """Render jet isosurfaces from subdomain."""
        logger.info(f"[Rank {self.rank}] Rendering jets from subdomain")
        
        from scene_builder import DensityRenderer
        
        density_renderer = DensityRenderer(self.renderer)
        jet_levels = self.config.jets.get_levels()
        
        diff_cmap = (self.config.density.colormap != self.config.jets.colormap)
        
        if self.config.jets.smallb2_rho:
            if self.data.smallb2 is not None:
                data = np.divide(self.data.smallb2, self.data.rho)
            else:
                logger.warning(f"[Rank {self.rank}] smallb2 data not available")
                return
        else:
            data = self.data.rho
        
        rval, _ = density_renderer.render_density_levels(
            rho=data,
            levels=jet_levels,
            cmap=self.config.jets.colormap,
            vmin=self.config.jets.vmin,
            vmax=self.config.jets.vmax,
            show_cbar=self.config.jets.show_cbar and self.rank == 0,
            opacity=self.config.jets.opacity,
            diff_cmap=diff_cmap
        )
        
        # Apply spatial clipping
        if self.use_decomposition and self.decomp.physical_bounds:
            self._apply_spatial_clip(rval, self.decomp.physical_bounds)
    
    def _apply_spatial_clip(self, actor_or_tuple, bounds: Tuple):
        """
        Apply spatial clipping to an actor to render only within bounds.
        
        Args:
            actor_or_tuple: VTK actor or tuple of (actor, ctf)
            bounds: Spatial bounds (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        # Extract actor from tuple if necessary
        if isinstance(actor_or_tuple, tuple):
            actor = actor_or_tuple[0]
        else:
            actor = actor_or_tuple
        
        # Create clipping planes for all 6 faces
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        
        planes = vtk.vtkPlaneCollection()
        
        # X planes
        plane_xmin = vtk.vtkPlane()
        plane_xmin.SetOrigin(xmin, 0, 0)
        plane_xmin.SetNormal(1, 0, 0)
        planes.AddItem(plane_xmin)
        
        plane_xmax = vtk.vtkPlane()
        plane_xmax.SetOrigin(xmax, 0, 0)
        plane_xmax.SetNormal(-1, 0, 0)
        planes.AddItem(plane_xmax)
        
        # Y planes
        plane_ymin = vtk.vtkPlane()
        plane_ymin.SetOrigin(0, ymin, 0)
        plane_ymin.SetNormal(0, 1, 0)
        planes.AddItem(plane_ymin)
        
        plane_ymax = vtk.vtkPlane()
        plane_ymax.SetOrigin(0, ymax, 0)
        plane_ymax.SetNormal(0, -1, 0)
        planes.AddItem(plane_ymax)
        
        # Z planes
        plane_zmin = vtk.vtkPlane()
        plane_zmin.SetOrigin(0, 0, zmin)
        plane_zmin.SetNormal(0, 0, 1)
        planes.AddItem(plane_zmin)
        
        plane_zmax = vtk.vtkPlane()
        plane_zmax.SetOrigin(0, 0, zmax)
        plane_zmax.SetNormal(0, 0, -1)
        planes.AddItem(plane_zmax)
        
        # Apply clipping to mapper
        mapper = actor.GetMapper()
        mapper.SetClippingPlanes(planes)
        
        logger.debug(f"[Rank {self.rank}] Applied spatial clipping")
    
    def setup_window(self) -> 'ParallelRenderPipeline':
        """
        Create render window for parallel rendering.
        
        All ranks create their own window for local rendering.
        """
        logger.info(f"[Rank {self.rank}] Setting up render window")
        
        # Create window (always offscreen in parallel mode)
        self.window = viz.RenderWindow(
            size=self.config.output.fig_size,
            offscreen=True
        )
        
        # Add renderer to window
        self.window.add_renderer(self.renderer, viewport=(0, 0, 1.0, 1.0))
        
        # Add time text overlay (only on rank 0)
        if self.rank == 0:
            time = float(self.config.paths.rho_file.split("_")[1][1:-1])
            viz.text(f'{time:.2e} M', posx=0.7, color='w', renderer=self.renderer)
        
        logger.info(f"[Rank {self.rank}] Render window setup complete")
        return self
    
    def render_parallel(self, output_path: str = None) -> 'ParallelRenderPipeline':
        """
        Render in parallel and composite results.
        
        Args:
            output_path: Output file path (uses config if not provided)
        """
        if output_path is None:
            output_path = self.config.output.fig_name
        
        logger.info(f"[Rank {self.rank}] Starting parallel render")
        
        # Each rank renders its portion
        self.window.renderWin.Render()
        
        # Capture image from each rank
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.window.renderWin)
        w2if.SetScale(self.config.output.scale)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()
        w2if.Update()
        
        # Get image data
        image = w2if.GetOutput()
        
        if self.is_parallel:
            # Composite images
            final_image = self._composite_images_parallel(image)
            
            # Only rank 0 writes output
            if self.rank == 0:
                self._write_composite_image(final_image, output_path)
                logger.info(f"[Rank 0] Saved composite image to {output_path}.png")
        else:
            # Serial mode - just write normally
            write_png(self.window, output_path, scale=self.config.output.scale)
            logger.info(f"Saved to {output_path}.png")
        
        return self
    
    def _composite_images_parallel(self, local_image):
        """
        Composite images from all ranks using alpha blending.
        
        Args:
            local_image: VTK image data from this rank
            
        Returns:
            Composited image (only valid on rank 0)
        """
        logger.info(f"[Rank {self.rank}] Compositing images")
        
        # Convert VTK image to numpy array
        from vtk.util import numpy_support
        
        dims = local_image.GetDimensions()
        scalars = local_image.GetPointData().GetScalars()
        local_array = numpy_support.vtk_to_numpy(scalars).reshape(dims[1], dims[0], -1)
        
        # Gather all images to rank 0
        all_images = self.comm.gather(local_array, root=0)
        
        if self.rank == 0:
            # Alpha compositing - blend all images
            composite = np.zeros_like(all_images[0], dtype=np.float32)
            alpha_acc = np.zeros(all_images[0].shape[:2], dtype=np.float32)
            
            for img in all_images:
                img_float = img.astype(np.float32) / 255.0
                
                # Detect non-background pixels (has data)
                has_data = np.sum(img_float, axis=2) > 0.01
                
                # Alpha blend
                blend_weight = has_data.astype(np.float32)[:, :, np.newaxis]
                composite += img_float * blend_weight * (1.0 - alpha_acc[:, :, np.newaxis])
                alpha_acc += has_data * (1.0 - alpha_acc)
            
            # Convert back to uint8
            composite = (composite * 255).clip(0, 255).astype(np.uint8)
            
            # Convert to VTK image
            composite_vtk = vtk.vtkImageData()
            composite_vtk.SetDimensions(dims[0], dims[1], 1)
            
            vtk_array = numpy_support.numpy_to_vtk(
                composite.reshape(-1, composite.shape[-1]),
                deep=True
            )
            composite_vtk.GetPointData().SetScalars(vtk_array)
            
            return composite_vtk
        
        return None
    
    def _write_composite_image(self, image, output_path: str):
        """
        Write composited VTK image to PNG file.
        
        Args:
            image: VTK image data
            output_path: Output file path
        """
        writer = vtk.vtkPNGWriter()
        writer.SetInputData(image)
        
        if not output_path.endswith('.png'):
            output_path += '.png'
        
        writer.SetFileName(output_path)
        writer.Write()
    
    def run_parallel(self):
        """
        Execute complete parallel rendering pipeline with domain decomposition.
        """
        logger.info(f"[Rank {self.rank}] Starting parallel render pipeline")
        
        self.load_data()
        self.setup_scene()
        self.setup_camera()
        self.setup_window()
        self.render_parallel()
        
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("PARALLEL RENDERING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            if self.use_decomposition:
                # Report memory savings
                full_size = np.prod(self.decomp.domain_shape) * 8 / (1024**3)  # GB for float64
                subdomain_size = np.prod([s.stop - s.start for s in self.decomp.subdomain_bounds]) * 8 / (1024**3)
                logger.info(f"Memory efficiency: {subdomain_size:.2f} GB per rank vs {full_size:.2f} GB full domain")
                logger.info(f"Memory reduction: {(1 - subdomain_size*self.size/full_size)*100:.1f}% "
                           f"({self.size}x decomposition)")


def main_parallel(args):
    """
    Main function for parallel rendering.
    
    Usage:
        mpirun -np 4 python parallel_render.py --config config.json
    """
    from config import RenderConfig
    
    # Setup logging
    if not HAS_MPI:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.error("mpi4py not available. Install with: pip install mpi4py")
        return
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Setup logging (separate file per rank)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s - [Rank {rank}] - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'parallel_render_rank{rank}.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Load configuration
        config = RenderConfig.from_json(args.config)
        
        # Force offscreen rendering in parallel mode
        config.output.interactive = False
        
        # Create and run parallel pipeline
        pipeline = ParallelRenderPipeline(config, comm)
        pipeline.run_parallel()
        
    except Exception as e:
        logger.error(f"[Rank {rank}] Rendering failed: {e}", exc_info=True)
        raise
    finally:
        if rank == 0:
            logger.info("Cleaning up")


if __name__ == "__main__":
    main_parallel()

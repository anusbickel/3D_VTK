import os
import logging
import numpy as np
from pathlib import Path
from postcactus import simdir
from postcactus.unit_abbrev import *
from postcactus.integral_curves import *
from utils import load_h5, abs_cos_theta

logger = logging.getLogger(__name__)

class DataResampler:
    """Handles resampling of simulation data onto uniform grids."""
    
    def __init__(self, sd: simdir.SimDir):
        """
        Initialize data resampler.
        
        Args:
            sd: SimDir object for the simulation
        """
        self.sd = sd
        logger.info("Initialized DataResampler")
    
    def _get_origin(self, time: float, center_on_ah: bool) -> np.ndarray:
        """
        Get origin coordinates, optionally centered on apparent horizon.
        
        Args:
            time: Time in code units
            center_on_ah: Whether to center on apparent horizon
            
        Returns:
            Origin coordinate array [x, y, z]
        """
        if not center_on_ah:
            return np.zeros((3,))
        
        origin = np.zeros((3,))
        ahl = self.sd.ahoriz.largest
        
        if ahl:
            ahx, ahy = ahl.ah.pos_x, ahl.ah.pos_y
            ahi = np.argmin(np.abs(time - ahx.t))
            logger.info(f"Shifting origin to AH position at t={ahx.t[ahi] / MS_CU:.2f} ms")
            origin = np.array([ahx.y[ahi], 0.0, 0.0])
        else:
            logger.warning("No apparent horizon data available for centering")
        
        logger.debug(f"Origin: {origin}")
        return origin
    
    def resample_data(self, var: str, xmax: float, res: int, time: float,
                      center_on_ah: bool = True) -> tuple:
        """
        Resample data onto a uniform grid.
        
        Args:
            var: Variable to resample.
            xmax: Domain size in code units (KM_CU)
            res: Resolution (number of grid points per dimension)
            time: Time in code units
            center_on_ah: Whether to center grid on apparent horizon
            
        Returns:
            Tuple of (density_data, actual_time, origin)
            - data: GridData object with resampled data
            - actual_time: Actual time of the data snapshot
            - origin: Origin coordinates used
            
        Example:
            rho_data, time, origin = resampler.resample_density(
                var='rho_b', xmax=105*KM_CU, res=200, time=100*MS_CU
            )
        """
        logger.info(f"Resampling density at time {time / MS_CU:.2f} ms")
        
        # Get origin
        origin = self._get_origin(time, center_on_ah)
        
        # Create uniform grid geometry
        g = gd.RegGeom(
            [res] * 3,
            origin + [-xmax, -xmax, -xmax],
            x1=origin + [xmax, xmax, xmax]
        )
        
        # Bind geometry and setup grid reader
        src = self.sd.grid.xyz.bind_geom(g, order=2)
        src = GridReaderUndoSymRefl(src, [2])
        
        # Find closest iteration to requested time
        its = self.sd.grid.xyz.get_iters('rho_b')
        tms = self.sd.grid.xyz.get_times('rho_b')
        it = its[np.argmin(np.abs(tms - time))]
        actual_time = tms[np.argmin(np.abs(tms - time))]
        
        logger.info(f"Using iteration {it} at time {actual_time / MS_CU:.2f} ms")
        
        # Read data
        data = src.read(var, it)
        
        logger.info(f"Resampled {var} to {res}^3 grid")
        return data, actual_time, origin
    

class DataLoader:
    """Handles loading and processing of simulation data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to simulation data directory
            
        Example:
            loader = DataLoader('../../h5/HUK/')
        """
        self.data_dir = Path(data_dir)
        self.sd = simdir.SimDir(str(data_dir))
        logger.info(f"Initialized DataLoader for {data_dir}")
    
    def load_density(self, rho_file: str) -> np.ndarray:
        """
        Load density data from HDF5 file.
        
        Args:
            rho_file: Name of density file (without .h5 extension)
            
        Returns:
            Density data array
            
        Example:
            rho = loader.load_density('rho_t100M_d100M')
        """
        if not rho_file.endswith(".h5"):
            rho_file = rho_file + ".h5"
        filepath = self.data_dir / rho_file
        logger.info(f"Loading density from {filepath}")
        rho = load_h5(str(filepath))
        return rho
    
    def load_horizon(self, time: float, tolerance: float = 4.0) -> dict:
        """
        Load apparent horizon data at given time.
        
        Args:
            time: Simulation time
            tolerance: Time tolerance in M (solar masses)
            
        Returns:
            Horizon patch data dictionary or None if not available
            
        Example:
            ah = loader.load_horizon(time=100.0)
        """
        lah = self.sd.ahoriz.largest
        if lah is None:
            logger.warning("No apparent horizon data available")
            return None
        
        logger.info(f"Loading horizon at time {time} M")
        ah = lah.shape.get_ah_patches(time, tol=tolerance * MS_CU)[0]
        logger.info(f"Loaded {len(ah)} horizon patches")
        return ah
    
    def load_field_lines(self, fncurv: str, nkeep: int, rcut: float,
                        origin: np.ndarray = None) -> tuple:
        """
        Load and process magnetic field lines.
        
        Args:
            fncurv: Field line file name
            nkeep: Number of field lines to keep after filtering
            rcut: Radial cutoff distance for filtering
            origin: Origin offset for coordinates (default: [0,0,0])
            
        Returns:
            Tuple of (curves, scalar, weight, time)
            - curves: List of field line coordinate arrays
            - scalar: Scalar values along field lines
            - weight: Weight values for each field line
            - time: Simulation time
            
        Example:
            curves, scalar, weight, time = loader.load_field_lines(
                'bfield_t100M_d100M_lines', nkeep=60, rcut=500.0
            )
        """
        filepath = self.data_dir / fncurv
        logger.info(f"Loading field lines from {filepath}")
        
        curves, seed, time, meta = load_curves(str(filepath))
        
        if origin is None:
            origin = meta.get('origin', np.zeros((3,)))
        logger.debug(f"Origin: {origin}")
        
        # Filter field lines with cylindrical mask
        logger.info("Filtering field lines with cylindrical mask")
        msph2 = cylinder_mask(rcut)
        curves = curves_cut(msph2, *curves)
        
        # Filter field lines with spherical mask
        logger.info("Filtering field lines with spherical mask")
        msph = sphere_mask(rcut)
        curves = curves_cut(msph, *curves)
        
        logger.info(f"Remaining curves after filtering: {len(curves[0])}")
        
        # Weight and select curves
        logger.info("Calculating curve weights")
        cweight = weight_order_binned(
            curves[0], curves[1], abs_cos_theta, nbins=20
        )
        
        logger.info(f"Selecting top {nkeep} curves")
        curves, cweight, scalar = keep_num_curves(
            curves[0], cweight, nkeep, curves[1]
        )
        
        # Normalize weights
        cweight /= np.max(cweight)
        
        logger.info(f"Kept {nkeep} field lines at time {time} M")
        return curves, scalar, cweight, time

    def load_smallb2(self, smallb2_file: str) -> np.ndarray:
        """
        Load B**2 data from HDF5 file.

        Args:
            smallb2_file: Name of B**2 file

        Returns:
            B**2 data array

        Example:
            smallb2 = loader.load_smallb2('smallb2_t128M_d100M')
        """
        if not smallb2_file.endswith(".h5"):
            smallb2_file = smallb2_file + ".h5"
        filepath = self.data_dir / smallb2_file
        logger.info(f"Loading B**2 from {filepath}")
        smallb2 = load_h5(str(filepath))
        return smallb2

    def load_or_resample(self, var: str, data_file: str = None,
                         resample_config: dict = None) -> tuple:
        """
        Load data from file or resample from simulation data.

        Args:
            var: Name of variable to resample
            data_file: Name of pre-resampled data file (optional)
            resample_config: Dict with keys: 'xmax', 'res', 'time', 'center_on_ah'
                           If provided, resampling will be performed

        Returns:
            Tuple of (data, time, origin)
            - If loading from file: (np.ndarray, extracted_time, None)
            - If resampling: (GridData, actual_time, origin)
        """

        if resample_config:
            # Resample simulation data
            logger.info(f"Resampling {var} from simulation data")
            return self.resampler.resample_data(var, **resample_config) 
        if data_file:
            if var=='rho_b':
                # Load pre-resampled densitydata
                logger.info("Loading pre-resampled density from file")
                data = self.load_density(data_file)
            elif var=='smallb2':
                # Load pre-resampled B**2 data
                logger.info("Loading pre-resampled B**2 from file")
                data = self.load_smallb2(data_file)
            else:
                logger.warning(f"Variable {var} not recognized.  Attempting density loading scheme.")
                try:
                  data = self.load_density(data_file)
                except:
                  logger.warning(f"Density loading scheme failed.  Please use correct loading scheme.")

            # Extract time from filename
            try:
                time = extract_time_from_filename(data_file)
            except ValueError:
                logger.warning("Could not extract time from filename")
                time = 0.0
        
    
    def adjust_horizon_for_origin(self, ah: dict, origin: np.ndarray):
        """
        Adjust horizon coordinates for given origin offset.
        
        This modifies the horizon data in-place by subtracting the origin
        offset from each horizon patch coordinate.
        
        Args:
            ah: Horizon data dictionary
            origin: Origin offset array [x, y, z]
            
        Example:
            loader.adjust_horizon_for_origin(ah, np.array([10, 0, 0]))
        """
        if ah is None:
            logger.warning("No horizon data to adjust")
            return
        
        logger.info(f"Adjusting horizon for origin offset: {origin}")
        for ahcd in ah.values():
            for ahp, o in zip(ahcd, origin):
                ahp -= o


class SimulationData:
    """Container for loaded simulation data."""
    
    def __init__(self, rho: np.ndarray, ah: dict = None,
                 curves: list = None, scalar: np.ndarray = None,
                 weight: np.ndarray = None, time: float = None,
                 smallb2: np.ndarray = None):
        """
        Initialize simulation data container.
        
        Args:
            rho: Density data array
            ah: Apparent horizon data dictionary
            curves: Field line curves list
            scalar: Field line scalar values
            weight: Field line weights
            time: Simulation time in M
            
        Example:
            data = SimulationData(
                rho=rho_array,
                ah=horizon_dict,
                time=100.0
            )
        """
        self.rho = rho
        self.smallb2 = smallb2
        self.ah = ah
        self.curves = curves
        self.scalar = scalar
        self.weight = weight
        self.time = time
        
        logger.info(f"SimulationData initialized at time {time} M")
        if rho is not None:
            rho_array = np.array(rho)
            logger.info(f"Density shape: {rho_array.shape}")
            logger.info(f"Density range: {np.min(rho):.2e} to {np.max(rho):.2e}")

        if smallb2 is not None:
            smallb2_array = np.array(smallb2)
            logger.info(f"B**2 shape: {smallb2_array.shape}")
            logger.info(f"B**2 range: {np.min(smallb2):.2e} to {np.max(smallb2):.2e}")
        
        if curves is not None:
            logger.info(f"Horizon: {len(ah)} patches loaded")
    
    def has_field_lines(self) -> bool:
        """Check if field line data is available."""
        return self.curves is not None and len(self.curves) > 0
    
    def has_horizon(self) -> bool:
        """Check if horizon data is available."""
        return self.ah is not None


def load_simulation_data(data_dir: str, rho_file: str,
                        bfield_file: str = None,
                        smallb2_file: str = None,
                        nkeep: int = 60, rcut: float = 500.0,
                        load_field_lines: bool = False,
                        load_smallb2: bool = False,
                        resample_config: dict = None) -> SimulationData:


    """
    Load all required simulation data.
    
    This is the main convenience function for loading data. It handles:
    - Loading density data
    - Extracting time from filename
    - Loading horizon data
    - Optionally loading field line data
    
    Args:
        data_dir: Path to data directory
        rho_file: Density file name (format: rho_t{TIME}M_d{DISTANCE}M)
        bfield_file: Field line file name (optional)
        smallb2_file: B**2 file name (optional)
        nkeep: Number of field lines to keep
        rcut: Radial cutoff for field lines (in KM_CU units)
        load_field_lines: Whether to load field line data
        resample_config: Configuration dictionary for resampling
        
    Returns:
        SimulationData object with all loaded data
        
    Example:
        # Load just density and horizon
        data = load_simulation_data(
            data_dir='../../h5/HUK/',
            rho_file='rho_t100M_d100M'
        )
        
        # Load with field lines
        data = load_simulation_data(
            data_dir='../../h5/HUK/',
            rho_file='rho_t100M_d100M',
            bfield_file='bfield_t100M_d100M_lines',
            nkeep=60,
            rcut=500.0,
            load_field_lines=True
        )
    """
    logger.info(f"Loading simulation data from {data_dir}")
    loader = DataLoader(data_dir)
    
    # Load density
    logger.info("Loading density data")
    rho = loader.load_density(rho_file)
    
    # Extract time from filename (format: rho_t{TIME}M_d{DISTANCE}M)
    try:
        time_str = rho_file.split("_")[1][1:-1]  # Extract time from 't100M'
        time = float(time_str)
        logger.info(f"Extracted time: {time} M")
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not extract time from filename: {e}")
        time = 0.0
    
    # Load horizon
    logger.info("Loading horizon data")
    ah = loader.load_horizon(time)
    
    # Load field lines if requested
    curves, scalar, weight = None, None, None
    if load_field_lines and bfield_file:
        logger.info("Loading field line data")
        curves, scalar, weight, time = loader.load_field_lines(
            bfield_file, nkeep, rcut
        )
        
        # Adjust horizon for origin if needed
        if ah is not None:
            origin = np.zeros((3,))
            loader.adjust_horizon_for_origin(ah, origin)
    elif load_field_lines:
        logger.warning("Field lines requested but no bfield_file provided")

    smallb2=None
    if load_smallb2 and smallb2_file:
        logger.info("Loading B**2 data")
        smallb2 = loader.load_smallb2(smallb2_file)
    elif load_smallb2:
        logger.warning("B**2 requested but no bfield_file provided")
        
    
    return SimulationData(
        rho=rho,
        ah=ah,
        smallb2=smallb2,
        curves=curves,
        scalar=scalar,
        weight=weight,
        time=time
    )


def extract_time_from_filename(filename: str) -> float:
    """
    Extract time value from standard filename format.
    
    Args:
        filename: Filename in format 'rho_t{TIME}M_d{DISTANCE}M'
        
    Returns:
        Time value as float
        
    Raises:
        ValueError: If time cannot be extracted
        
    Example:
        time = extract_time_from_filename('rho_t100M_d100M')  # Returns 100.0
    """
    try:
        time_str = filename.split("_")[1][1:-1]
        return float(time_str)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract time from filename {filename}: {e}")


def extract_distance_from_filename(filename: str) -> float:
    """
    Extract distance value from standard filename format.
    
    Args:
        filename: Filename in format 'rho_t{TIME}M_d{DISTANCE}M'
        
    Returns:
        Distance value as float
        
    Raises:
        ValueError: If distance cannot be extracted
        
    Example:
        dist = extract_distance_from_filename('rho_t100M_d100M')  # Returns 100.0
    """
    try:
        dist_str = filename.split("_")[2][1:-1]
        return float(dist_str)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract distance from filename {filename}: {e}")


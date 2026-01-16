import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Union
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera positioning configuration."""
    r: float = 550.0
    theta: float = 0.7068  # pi*1.8/8
    phi: float = -0.5236   # -pi/6
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Return camera configuration as tuple (r, theta, phi)."""
        return (self.r, self.theta, self.phi)

@dataclass
class ResamplingConfig:
    """Configuration for resampling onto a uniform grid."""
    enabled: bool = False
    rmax: float = 500
    nsample: int = 600
    center_on_ah: bool = True
    time: float = 0.0

    def get_xmax(self) -> float:
        """Get domain size"""
        return self.rmax



@dataclass
class DensityConfig:
    """Density rendering configuration."""
    levels_start: float = -13.0
    levels_stop: float = -4.0
    levels_num: int = 20
    colormap: str = 'plasma'
    vmin: float = -12.0
    vmax: float = -8.0
    show_cbar: bool = True
    opacity: Union[float, List[float]] = 0.999
    
    def get_levels(self) -> List[np.ndarray]:
        """Generate density level arrays."""
        return [np.linspace(self.levels_start, self.levels_stop, self.levels_num)]


@dataclass
class JetConfig:
    """Jet rendering configuration."""
    show: bool = False
    smallb2_rho: bool = False
    levels_start: float = -20.0
    levels_stop: float = -18.0
    levels_num: int = 5
    colormap: str = 'summer'
    vmin: float = -20.0
    vmax: float = -18.0
    show_cbar: bool = True
    opacity: float = 0.05
    
    def get_levels(self) -> np.ndarray:
        """Generate jet level array."""
        return np.linspace(self.levels_start, self.levels_stop, self.levels_num)


@dataclass
class FieldLineConfig:
    """Field line rendering configuration."""
    show: bool = False
    rcut: float = 500.0  # in KM_CU
    nkeep: int = 60
    tube_radius: float = 1.0  # in KM_CU
    vmax_bfield: float = 10e15
    magnitudes: float = 1.0
    tradius: float = 0.00001  # in KM_CU
    lensc_r: float = 10.0  # in KM_CU
    lensc: float = 10.0  # in KM_CU


@dataclass
class BlackHoleConfig:
    """Black hole rendering configuration."""
    show: bool = True
    show_spin: bool = True
    timestep: int = 32


@dataclass
class GridConfig:
    """Grid rendering configuration."""
    show: bool = False
    size: List[int] = field(default_factory=lambda: [2, 2, 2])
    spacing: float = 50.0


@dataclass
class OutputConfig:
    """Output configuration."""
    fig_name: str = "figure"
    fig_size: Tuple[int, int] = (1024, 768)
    scale: int = 1
    interactive: bool = False
    movie: bool = False
    background_image: Optional[str] = None


@dataclass
class PathConfig:
    """File path configuration."""
    data_dir: str = "../../h5/HUK/"
    scalar_data_dir: str = "/home/zn9800/HUK_data/"
    rho_file: str = "rho_t0M_d100M"
    bfield_file: str = ""
    smallb2_file: str = ""
    
    def get_rho_path(self) -> Path:
        """Get full path to density file."""
        return Path(self.data_dir) / self.rho_file
    
    def get_bfield_path(self) -> Optional[Path]:
        """Get full path to magnetic field file."""
        if self.bfield_file:
            return Path(self.data_dir) / self.bfield_file
        return None

    def get_smallb2_path(self) -> Optional[Path]:
        """Get full path to smallb2 file."""
        if self.smallb2_file:
            return Path(self.data_dir) / self.smallb2_file
        return None


@dataclass
class RenderConfig:
    """Main configuration container for all rendering settings."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    jets: JetConfig = field(default_factory=JetConfig)
    field_lines: FieldLineConfig = field(default_factory=FieldLineConfig)
    black_holes: BlackHoleConfig = field(default_factory=BlackHoleConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'RenderConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            RenderConfig instance
            
        Example:
            config = RenderConfig.from_json('my_config.json')
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            camera=CameraConfig(**data.get('camera', {})),
            resampling=ResamplingConfig(**data.get('resampling', {})),
            density=DensityConfig(**data.get('density', {})),
            jets=JetConfig(**data.get('jets', {})),
            field_lines=FieldLineConfig(**data.get('field_lines', {})),
            black_holes=BlackHoleConfig(**data.get('black_holes', {})),
            grid=GridConfig(**data.get('grid', {})),
            output=OutputConfig(**data.get('output', {})),
            paths=PathConfig(**data.get('paths', {}))
        )
    
    def to_json(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Output JSON file path
            
        Example:
            config.to_json('my_config.json')
        """
        data = {
            'camera': asdict(self.camera),
            'resampling': asdict(self.resampling),
            'density': asdict(self.density),
            'jets': asdict(self.jets),
            'field_lines': asdict(self.field_lines),
            'black_holes': asdict(self.black_holes),
            'grid': asdict(self.grid),
            'output': asdict(self.output),
            'paths': asdict(self.paths)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_args(cls, args) -> 'RenderConfig':
        """
        Create configuration from command line arguments.
        
        Args:
            args: argparse.Namespace object with command line arguments
            
        Returns:
            RenderConfig instance
            
        Example:
            args = parser.parse_args()
            config = RenderConfig.from_args(args)
        """
        return cls(
            camera=CameraConfig(
                r=args.camr,
                theta=args.camth,
                phi=args.camph
            ),
            paths=PathConfig(
                data_dir=args.dr,
                scalar_data_dir=args.dd,
                rho_file=args.rho,
                bfield_file=args.bfield,
                smallb2_file=args.smallb2
            ),
            output=OutputConfig(
                fig_name=args.fig_name,
                fig_size=args.fig_size,
                interactive=args.interactive,
                movie=args.movie
            ),
            grid=GridConfig(
                show=args.grid
            )
        )
    
    def __str__(self) -> str:
        """Return a readable string representation of the configuration."""
        lines = [
            "RenderConfig:",
            f"  Camera: r={self.camera.r}, θ={self.camera.theta:.3f}, φ={self.camera.phi:.3f}",
            f"  Resampling: {'enabled' if self.resampling.enabled else 'disabled'}",
            f"  Density: {self.density.colormap}, opacity={self.density.opacity}",
            f"  Jets: {'enabled' if self.jets.show else 'disabled'}",
            f"  Field lines: {'enabled' if self.field_lines.show else 'disabled'}",
            f"  Black holes: {'enabled' if self.black_holes.show else 'disabled'}",
            f"  Grid: {'enabled' if self.grid.show else 'disabled'}",
            f"  Output: {self.output.fig_name}, size={self.output.fig_size}",
            f"  Data: {self.paths.rho_file}"
        ]
        return "\n".join(lines)



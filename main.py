import os
import argparse
import logging
import numpy as np
from math import pi
from pathlib import Path
from postcactus.unit_abbrev import *
from postcactus import viz_vtk as viz

from config import RenderConfig
from render_pipeline import RenderPipeline, create_default_config
from utils import write_png
from parallel_render import main_parallel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def movie_scheme(win, fname, cam, hrho, scale):
    """
    Movie rendering scheme that rotates camera around scene.
    
    This function generates a series of frames by rotating the camera
    360 degrees around the scene at a fixed elevation.
    
    Args:
        win: VTK render window
        fname: Output filename prefix
        cam: Camera configuration tuple (r, theta, phi)
        hrho: Density filename (for time text)
        scale: Image scale factor
        
    Example:
        movie_scheme(window, 'output/frame', (550, 0.7, 0), 'rho_t100M_d100M', 1)
    """
    rend = win.all_renderers[0]
    time = int(hrho.split("_")[1][1:-1])
    viz.text(f'{time:.2e} M', posx=0.7, color='w', renderer=rend)
    
    r, th, ph = cam
    
    logger.info("Starting movie rendering with camera rotation")
    num_frames = 15
    for n, phi in enumerate(np.linspace(0, 2 * np.pi, num_frames)):
        logger.info(f'Rendering frame {n+1}/{num_frames}: '
                   f'(r={r}, theta={th:.3f}, phi={phi:.3f})')
        viz.set_camera(rend, r, th, phi)
        write_png(win, f'{fname}_{n:03d}', scale=scale)
    
    logger.info(f"Movie rendering complete: {num_frames} frames saved")


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Render simulation data with 3D density visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default configuration file
  python main.py --create-config
  
  # Render using configuration file
  python main.py --config my_config.json
  
  # Render using command line arguments
  python main.py --dr ../../h5/HUK/ --rho rho_t100M_d100M --camr 600
  
  # Render interactively
  python main.py --config my_config.json --interactive
  
  # Render movie
  python main.py --config my_config.json --movie
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help="Path to JSON configuration file")
    
    parser.add_argument('--create-config', action='store_true',
                       help="Create default config file and exit")
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument('--fig-name', type=str,
                             default="movie_test/figure",
                             help="Output filename (without .png extension)")
    output_group.add_argument('--fig-size', type=tuple,
                             default=(1024, 768),
                             help="Image size as (width, height)")
    output_group.add_argument('--interactive', action='store_true',
                             help="Open interactive VTK window")
    output_group.add_argument('--movie', action='store_true',
                             help="Render rotating movie")
    output_group.add_argument('--scale', type=int, default=1,
                             help="Image scale factor")
    
    # Path options
    path_group = parser.add_argument_group('path options')
    path_group.add_argument('--dr', type=str,
                           default="../../h5/HUK/",
                           help="Path to data directory")
    path_group.add_argument('--dd', type=str,
                           default='/home/zn9800/HUK_data/',
                           help="Path to scalar data directory (for black holes)")
    path_group.add_argument('--rho', type=str,
                           default='rho_t0M_d100M',
                           help="Density file name (format: rho_t{TIME}M_d{DISTANCE}M)")
    path_group.add_argument('--bfield', type=str,
                           default='',
                           help="Magnetic field file name")
    
    # Camera options
    cam_group = parser.add_argument_group('camera options')
    cam_group.add_argument('--camr', type=float,
                          default=550.0,
                          help="Camera radial distance")
    cam_group.add_argument('--camth', type=float,
                          default=pi * 1.8 / 8,
                          help="Camera theta angle (polar)")
    cam_group.add_argument('--camph', type=float,
                          default=-pi / 6,
                          help="Camera phi angle (azimuthal)")
    
    # Rendering options
    render_group = parser.add_argument_group('rendering options')
    render_group.add_argument('--grid', action='store_true',
                             help="Show coordinate grid")
    render_group.add_argument('--no-black-holes', action='store_true',
                             help="Disable black hole rendering")
    render_group.add_argument('--no-spin', action='store_true',
                             help="Disable spin arrow rendering")
    render_group.add_argument('--show-jets', action='store_true',
                             help="Enable jet rendering")
    render_group.add_argument('--show-field-lines', action='store_true',
                             help="Enable field line rendering")
    render_group.add_argument('--parallel', action='store_true',
                             help="Enable parallel rendering across processors")
    
    # Logging options
    log_group = parser.add_argument_group('logging options')
    log_group.add_argument('--verbose', '-v', action='store_true',
                          help="Enable verbose (DEBUG) logging")
    log_group.add_argument('--quiet', '-q', action='store_true',
                          help="Quiet mode (WARNING level logging)")
    
    return parser.parse_args()


def setup_logging(args):
    """
    Configure logging based on command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def create_config_from_args(args) -> RenderConfig:
    """
    Create RenderConfig from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        RenderConfig instance
    """
    config = RenderConfig.from_args(args)
    
    # Apply additional command line options
    if args.scale:
        config.output.scale = args.scale
    
    if args.no_black_holes:
        config.black_holes.show = False
    
    if args.no_spin:
        config.black_holes.show_spin = False
    
    if args.show_jets:
        config.jets.show = True

    if args.show_field_lines:
        config.field_lines.show = True
    
    return config


def ensure_output_directory(output_path: str):
    """
    Ensure output directory exists.
    
    Args:
        output_path: Output file path
    """
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")


def main(args):
    """
    Main execution function.
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads or creates configuration
    4. Creates and runs the rendering pipeline
    """
    # Setup logging
    setup_logging(args)
    
    # Create default config if requested
    if args.create_config:
        logger.info("Creating default configuration file")
        create_default_config()
        logger.info("Created default_config.json")
        logger.info("Edit this file to customize your rendering, then run:")
        logger.info("  python main.py --config default_config.json")
        return
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            return
        config = RenderConfig.from_json(args.config)
        
        # Override with command line options if provided
        if args.interactive:
            config.output.interactive = True
        if args.movie:
            config.output.movie = True
    else:
        logger.info("Using command line arguments (no config file specified)")
        config = create_config_from_args(args)
    
    # Log configuration summary
    logger.info("Configuration summary:")
    logger.info(f"  Data file: {config.paths.rho_file}")
    logger.info(f"  Output: {config.output.fig_name}")
    logger.info(f"  Camera: r={config.camera.r}, θ={config.camera.theta:.3f}, φ={config.camera.phi:.3f}")
    logger.info(f"  Interactive: {config.output.interactive}")
    logger.info(f"  Movie: {config.output.movie}")
    
    # Ensure output directory exists
    ensure_output_directory(config.output.fig_name)
    
    # Create and run pipeline
    pipeline = RenderPipeline(config)
    
    try:
        pipeline.load_data()
        pipeline.setup_scene()
        pipeline.setup_camera()
        pipeline.setup_window()
        
        if config.output.movie:
            logger.info("Starting movie rendering")
            pipeline.render_movie(movie_scheme)
        else:
            logger.info("Starting single frame rendering")
            pipeline.render()
        
        logger.info("=" * 60)
        logger.info("RENDERING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        if not config.output.interactive and not config.output.movie:
            output_file = config.output.fig_name
            if not output_file.endswith('.png'):
                output_file += '.png'
            logger.info(f"Output saved to: {output_file}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check that all data files exist")
    except KeyError as e:
        logger.error(f"Missing data key: {e}")
        logger.error("Please check your data files are in the correct format")
    except Exception as e:
        logger.error(f"Rendering failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up resources
        pipeline.cleanup()


if __name__ == "__main__":
    args = parse_args()

    if args.parallel:
        main_parallel(args)
    else:
        main(args)

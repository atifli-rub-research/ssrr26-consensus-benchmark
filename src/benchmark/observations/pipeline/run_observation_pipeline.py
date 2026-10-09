from PIL import Image
from pathlib import Path
from typing import List
from datetime import datetime
from tqdm import tqdm
from loguru import logger
import shutil
import json

from benchmark.observations.core.types import Drone
from benchmark.observations.cropping.grid_cropper import GridCropper
from benchmark.observations.corruptions.factory import apply_corruptions
from benchmark.observations.pipeline.config import ObservationPipelineConfig, parse_agents
from benchmark.observations.detectors.yolo26_detector import Yolo26Detector
from benchmark.observations.evaluation.metrics import compute_observation_metrics
from benchmark.observations.ground_truth.yolo_ground_truth import load_tile_ground_truths


def run_pipeline(config_path_str: str):
    """Run the mine detection consensus benchmark pipeline.
    
    This function orchestrates the entire pipeline by calling smaller, focused functions
    that each handle one logical step. It reads the configuration, sets up outputs,
    processes the image through multiple drone models with corruptions, and saves results.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    config_path = Path(config_path_str)

    config = ObservationPipelineConfig()
    config.load(config_path)

    logger.info("Running mine detection consensus benchmark pipeline...")

    from pprint import pformat
    logger.info("Configuration:\n{}", pformat(config.get_raw()))
    logger.info("=" * 80)

    # =====================================
    # Setup and Configuration Phase
    # =====================================
    
    run_output_dir = _setup_output_directory(config)
    shutil.copy2(config_path, run_output_dir / "config.yaml")
    logger.info("Run output directory: {}", run_output_dir)

    config_params = _get_pipeline_configuration(config)

    # =====================================
    # Parsing and Preprocessing Phase
    # =====================================
    
    drones = parse_agents(config)
    config_params["image"], config_params["image_crops"] = _load_and_preprocess_image(
        image_path=config_params["image_path"],
        tile_size=config_params["tile_size"],
        stride=config_params["stride"],
        drones=drones,
    )

    logger.info("Done parsing")
    logger.info("Total drones: {}", len(drones))

    # =====================================
    # Ground Truth Preparation Phase
    # =====================================
    
    config_params["ground_truths"] = load_tile_ground_truths(
        image_path=config_params["image_path"],
        config=config,
        image_crops=config_params["image_crops"],
    )

    ground_truth_dir = run_output_dir / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    with (ground_truth_dir / "tile_ground_truths.json").open("w") as f:
        json.dump(config_params["ground_truths"], f, indent=2)

    # =====================================
    # Processing and Detection Phase
    # =====================================
    
    all_predictions = _process_all_drone_pipelines(
        config=config,
        config_params=config_params,
        run_output_dir=run_output_dir,
        drones=drones,
        ground_truths=config_params["ground_truths"],
        image_crops=config_params["image_crops"],
    )

    logger.info("Done corruptions and detection")

def _setup_output_directory(config: ObservationPipelineConfig) -> Path:
    """Setup experiment output directory and copy config.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Path: run_output_dir path
    """
    scenario_name = config.get("scenario_name", "experiment")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{scenario_name}"
    
    base_output_dir = Path(config.get("output.output_dir"))
    run_output_dir = base_output_dir / run_name
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    return run_output_dir


def _get_pipeline_configuration(config: ObservationPipelineConfig) -> dict:
    """Extract configuration parameters needed for pipeline execution.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        dict: Configuration parameters with defaults
    """
    return {
        "image_path": Path(config.get("input.image_path")),
        "tile_size": config.get("tiling.tile_size"),
        "stride": config.get("tiling.stride"),
        "base_seed": config.get("experiment.seed", 42),
        "save_corrupted_tiles": config.get("output.save_corrupted_tiles", False),
        "save_predictions": config.get("output.save_predictions", False),
    }

def _load_and_preprocess_image(image_path: Path, tile_size: int, stride: int, drones: List[Drone]) -> tuple[Image.Image, list]:
    """Load the input image and preprocess it by tiling it into crops.
    
    Args:
        image_path: Path to the input image
        tile_size: Size of each tile
        stride: Stride for tiling
        drones: List of drone configurations
        
    Returns:
        tuple: (loaded_image, list_of_image_crops)
    """
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path).convert("RGB")
    
    cropper = GridCropper(
        cell_size=(tile_size, tile_size),
        stride=(stride, stride),
    )

    image_crops = cropper.generate_cells(
        image=image,
        metadata={
            "source_id": image_path.stem,
            "image_path": str(image_path),
        },
    )

    logger.info("Done cropping ({} tiles generated)", len(image_crops))
    return image, image_crops


def _setup_directories(config_params: dict, run_output_dir: Path) -> tuple[Path | None, Path | None]:
    """Setup output directories for corrupted tiles and predictions.
    
    Args:
        config_params: Configuration parameters
        run_output_dir: Base output directory
        
    Returns:
        tuple: (corrupted_dir, predictions_dir) - may be None if save flags are False
    """
    corrupted_dir = run_output_dir / "corrupted_tiles"
    predictions_dir = run_output_dir / "predictions"

    if config_params["save_corrupted_tiles"]:
        corrupted_dir.mkdir(parents=True, exist_ok=True)
    if config_params["save_predictions"]:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    return corrupted_dir if config_params["save_corrupted_tiles"] else None, \
           predictions_dir if config_params["save_predictions"] else None


def _process_all_drone_pipelines(config: ObservationPipelineConfig, config_params: dict, 
                                 run_output_dir: Path, drones: List[Drone],
                                 ground_truths: dict, image_crops: list) -> list[dict]:
    """Process the image through each drone's pipeline with corruptions and detection.
    
    Args:
        config: Pipeline configuration
        config_params: Parsed configuration parameters
        run_output_dir: Output directory for results
        drones: List of drone configurations
        ground_truths: Ground truth data by tile index
        image_crops: List of image tile crops
        
    Returns:
        list: All predictions from all drones
    """
    corrupted_dir, predictions_dir = _setup_directories(config_params, run_output_dir)

    all_predictions = []

    logger.info("Running corruptions and YOLO detection...")

    for drone_index, drone in enumerate(drones, start=1):
        drone_predictions = _process_single_drone_pipeline(
            drone=drone,
            drone_index=drone_index,
            drones=drones,
            config=config,
            config_params=config_params,
            image_crops=image_crops,
            ground_truths=ground_truths,
            corrupted_dir=corrupted_dir,
            predictions_dir=predictions_dir,
        )
        all_predictions.extend(drone_predictions)

    if config_params["save_predictions"] and predictions_dir is not None:
        all_predictions_path = predictions_dir / "all_predictions.json"
        with all_predictions_path.open("w") as f:
            json.dump(all_predictions, f, indent=2)
    
    return all_predictions


def _process_single_drone_pipeline(drone: Drone, drone_index: int, drones: List[Drone],
                                config: ObservationPipelineConfig, config_params: dict,
                                image_crops: list, ground_truths: dict,
                                corrupted_dir: Path | None, predictions_dir: Path | None) -> list:
    """Process a single drone's pipeline: load model, apply corruptions, run detection.
    
    Args:
        drone: Drone configuration
        drone_index: Index of current drone (for logging)
        drones: All drone configurations
        config: Pipeline configuration 
        config_params: Parsed configuration parameters
        image_crops: List of image tiles
        ground_truths: Ground truth data by tile index
        corrupted_dir: Directory to save corrupted tiles (or None)
        predictions_dir: Directory to save predictions (or None)
        
    Returns:
        list: Predictions from this drone
    """
    logger.info("Processing drone {}/{}: {}", drone_index, len(drones), drone.drone_id)

    detector = Yolo26Detector(drone.model, drone_id=drone.drone_id)
    model_path = detector.model_path

    drone_predictions = []
    drone_output_dir = None
    if corrupted_dir:
        drone_output_dir = corrupted_dir / drone.drone_id
        drone_output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(
        enumerate(image_crops),
        total=len(image_crops),
        desc=f"drone {drone_index}/{len(drones)} corrupt+detect",
        unit="tile",
    )

    base_seed = config_params["base_seed"]
    
    for tile_index, crop in progress_bar:
        tile_seed = hash((base_seed, drone.drone_id, tile_index)) % (2**32)

        corrupted = apply_corruptions(
            image=crop.image,
            corruption_config=drone.corruptions,
            seed=tile_seed,
        )

        if drone_output_dir:
            tile_filename = f"tile_{tile_index:05d}.png"
            corrupted.save(drone_output_dir / tile_filename)

        detections = detector.predict(corrupted)

        tile_ground_truth = ground_truths[tile_index]
        prediction = {
            "drone_id": drone.drone_id,
            "tile_index": tile_index,
            "model": str(model_path),
            "corruptions": drone.corruptions,
            "ground_truth": tile_ground_truth,
            "true_mine_count": tile_ground_truth["mine_count"],
            "true_mine_present": tile_ground_truth["mine_present"],
            "num_detections": len(detections),
            "detections": detections,
        }

        drone_predictions.append(prediction)
        del corrupted

    metrics = compute_observation_metrics(drone_predictions)

    logger.info("Results for {}", drone.drone_id)
    logger.info("  Model: {}", model_path)
    logger.info("  Tiles: {}", metrics["num_tiles"])
    logger.info("  Tiles with true mines: {}", metrics["tiles_with_true_mines"])
    logger.info("  Total true mines: {}", metrics["total_true_mines"])
    logger.info("  Tiles with detections: {}", metrics["tiles_with_detections"])
    logger.info("  Total detections: {}", metrics["total_detections"])
    logger.info("  Mean confidence: {:.4f}", metrics["mean_confidence"])

    if predictions_dir and config_params["save_predictions"]:
        prediction_path = predictions_dir / f"{drone.drone_id}_predictions.json"
        with prediction_path.open("w") as f:
            json.dump(drone_predictions, f, indent=2)

    del detector
    return drone_predictions

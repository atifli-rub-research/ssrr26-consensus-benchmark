from pathlib import Path
from typing import Any, List

from benchmark.observations.core.types import Drone



class ObservationPipelineConfig:
    """Configuration manager for the observation pipeline.
    
    Handles loading and accessing YAML configuration files for the mine detection
    consensus benchmark pipeline. Supports nested config values using dot notation.
    """
    
    def __init__(self):
        self.config = {}

    def load(self, path: str | Path) -> dict:
        """Load configuration from YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            dict: The loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If the config file does not exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        import yaml
        with path.open("r") as f:
            self.config = yaml.safe_load(f)

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Access nested config values using dot notation.

        Example:
            config.get("tiling.tile_size")
        config.get("experiment.seed", 42)
        
        Args:
            key: Dot-separated path to config value
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        current = self.config

        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default

            current = current[part]

        return current

    def get_raw(self) -> dict:
        return self.config

def parse_agents(
    config: ObservationPipelineConfig,
) -> List[Drone]:

    drones_dict = config.get("agents")

    drones = []

    for drone_dict in drones_dict:

        drone_id = drone_dict["id"]

        model = drone_dict["model"]

        corruptions = drone_dict["corruptions"]

        drones.append(
            Drone(
                drone_id,
                model,
                corruptions,
            )
        )

    return drones

from typing import Union, Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path

# --- Distribution Models (for Priors) ---

class DistributionConfig(BaseModel):
    type: str
    mean: Optional[float] = None
    sigma: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    
    # Allow extra fields for other distributions
    class Config:
        extra = "allow"

# --- Component Models ---

class NoiseConfig(BaseModel):
    type: str = "Gaussian"
    sigma: float

class ImageConfig(BaseModel):
    pixel_scale: float
    size_x: int
    size_y: int
    n_objects: int = 1 # Default to 1 for simple tests
    noise: NoiseConfig

class PSFConfig(BaseModel):
    type: str = "Gaussian"
    sigma: float
    beta: Optional[float] = 2.5 # For Moffat

class ShearComponentConfig(BaseModel):
    # Can be a fixed float or a distribution
    type: Optional[str] = None # If None, assume fixed value in parent or handled elsewhere
    mean: Optional[float] = 0.0
    sigma: Optional[float] = 0.05
    
    # To handle the case where it's just a float in YAML, we might need a custom validator
    # but for now let's assume structured input as per design doc

class ShearConfig(BaseModel):
    type: str = "G1G2"
    g1: Union[float, DistributionConfig]
    g2: Union[float, DistributionConfig]

class GalaxyConfig(BaseModel):
    type: str = "Exponential"  # Changed default from Sersic to Exponential
    n: Optional[Union[float, DistributionConfig]] = None  # Make optional for Exponential
    flux: Union[float, DistributionConfig]
    half_light_radius: Union[float, DistributionConfig] = Field(..., alias="half_light_radius")
    shear: ShearConfig

class InferenceConfig(BaseModel):
    warmup: int = 500
    samples: int = 1000
    chains: int = 1
    dense_mass: bool = False

class ShineConfig(BaseModel):
    image: ImageConfig
    psf: PSFConfig
    gal: GalaxyConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    data_path: Optional[str] = None
    output_path: str = "results"

class ConfigHandler:
    @staticmethod
    def load(path: str) -> ShineConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        # Basic validation and type conversion via Pydantic
        return ShineConfig(**data)

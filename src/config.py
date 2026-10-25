"""
Central Configuration for RDH Triple-Hybrid Predictor Framework.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional
import os


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    bossbase_dir: Path = Path(r"C:\Users\mails\Downloads\BOSSbase_1.01")
    uscsipi_dir: Path = Path(r"C:\Users\mails\Downloads\misc\misc")
    
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    splits_path: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.outputs_dir = self.project_root / "outputs"
        self.figures_dir = self.outputs_dir / "figures"
        self.tables_dir = self.outputs_dir / "tables"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.manifest_path = self.processed_data_dir / "dataset_manifest.json"
        self.splits_path = self.processed_data_dir / "splits.json"


@dataclass
class DatasetConfig:
    seed: int = 42
    train_size: int = 3500
    val_size: int = 500
    holdout_size: int = 6000
    target_size: Tuple[int, int] = (512, 512)
    grayscale: bool = True
    val_range: Tuple[int, int] = (0, 255)
    
    # USC-SIPI standard 512x512 benchmark files
    benchmark_test_files: List[str] = field(default_factory=lambda: [
        "4.2.03.tiff",   # Baboon / Mandrill
        "4.2.05.tiff",   # Airplane / F-16
        "4.2.06.tiff",   # Sailboat / Lake
        "4.2.07.tiff",   # Peppers
        "boat.512.tiff", # Boat
        "house.tiff",    # House
        "5.2.08.tiff",   # Couple
        "5.2.09.tiff",   # Aerial / Stream & Bridge
        "5.2.10.tiff",   # Girl
        "7.1.01.tiff",   # Grass texture
        "7.1.02.tiff",   # Fieldstone texture
        "7.1.03.tiff",   # Sand texture
        "7.1.04.tiff",   # Straw texture
        "7.1.05.tiff",   # Brick texture
        "7.1.06.tiff",   # Wood texture
        "7.1.07.tiff",   # Raffia texture
        "7.1.08.tiff",   # Water texture
        "7.1.09.tiff",   # Pigskin texture
        "7.1.10.tiff",   # Bark texture
    ])


@dataclass
class HardwareConfig:
    # RTX 2050 Laptop GPU has 4GB VRAM.
    # Keep batch size conservative to prevent CUDA OOM.
    batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0  # 0 is recommended on Windows to avoid spawn issues
    pin_memory: bool = True
    device: str = "cuda"  # fallback to 'cpu' if cuda is not available


@dataclass
class PredictorConfig:
    # Router thresholds: [0, T1) -> RP, [T1, T2) -> CNN, [T2, inf) -> Transformer
    initial_T1: float = 3.0
    initial_T2: float = 12.0
    window_size: int = 8  # Window restriction for lightweight Transformer


@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)


# Global default configuration instance
DEFAULT_CONFIG = ProjectConfig()


def get_config() -> ProjectConfig:
    return DEFAULT_CONFIG

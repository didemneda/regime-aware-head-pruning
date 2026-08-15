from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    tslib_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_dir / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.project_dir / "results"

    @property
    def config_dir(self) -> Path:
        return self.project_dir / "configs"

    def create_output_dirs(self) -> None:
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

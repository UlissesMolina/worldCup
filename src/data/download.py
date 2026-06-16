import shutil
from pathlib import Path

import kagglehub


MATCH_DATASET = "martj42/international-football-results-from-1872-to-present"


def download_match_data(output_dir: Path) -> Path:
    """Download match results CSV from Kaggle and copy to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached_path = Path(kagglehub.dataset_download(MATCH_DATASET))

    for csv_file in cached_path.glob("*.csv"):
        shutil.copy2(csv_file, output_dir / csv_file.name)

    return output_dir

from unittest.mock import patch, MagicMock
from pathlib import Path
from src.data.download import download_match_data


def test_download_match_data_calls_kagglehub(tmp_path):
    """Verify download_match_data uses kagglehub and copies files to output dir."""
    fake_kaggle_path = tmp_path / "kaggle_cache"
    fake_kaggle_path.mkdir()
    (fake_kaggle_path / "results.csv").write_text("date,home_team,away_team\n2024-01-01,Brazil,Germany\n")

    output_dir = tmp_path / "raw"

    with patch("src.data.download.kagglehub.dataset_download", return_value=str(fake_kaggle_path)):
        download_match_data(output_dir=output_dir)

    assert (output_dir / "results.csv").exists()
    content = (output_dir / "results.csv").read_text()
    assert "Brazil" in content

"""Download the AGMARKNET daily price dataset from Kaggle into data/raw/."""
import os
from pathlib import Path

from agromaster import config


class KaggleTokenMissingError(RuntimeError):
    """Raised when no Kaggle API credentials can be found."""


def kaggle_config_dir() -> Path:
    """Return the folder the kaggle client reads its credentials from."""
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".kaggle"


def has_kaggle_credentials() -> bool:
    """True if a kaggle.json/access_token file or credential env vars exist."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    folder = kaggle_config_dir()
    return (folder / "kaggle.json").is_file() or (folder / "access_token").is_file()


def is_downloaded(dest: Path) -> bool:
    """True if the destination already holds at least one CSV file."""
    return dest.is_dir() and any(dest.rglob("*.csv"))


def download_dataset(
    dataset: str | None = None,
    dest: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and unzip the Kaggle dataset into ``dest``.

    Skips the download when ``dest`` already contains CSV files, unless
    ``force`` is set. Raises ``KaggleTokenMissingError`` with setup steps
    when no Kaggle credentials are available.
    """
    dataset = dataset or config.KAGGLE_DATASET
    dest = dest or config.DATA_RAW
    if not dataset:
        raise ValueError("KAGGLE_DATASET is not set in agromaster.config")
    if is_downloaded(dest) and not force:
        print(f"Raw data already present in {dest}; skipping download.")
        return dest
    if not has_kaggle_credentials():
        raise KaggleTokenMissingError(
            "Kaggle API token not found. Create one at "
            "https://www.kaggle.com/settings/api and save it as "
            f"{kaggle_config_dir() / 'kaggle.json'} "
            "(or set KAGGLE_API_TOKEN)."
        )

    # Imported here because the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
    return dest


if __name__ == "__main__":
    download_dataset()

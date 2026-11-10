"""Tests for the Kaggle download helpers in agromaster.download."""
import sys
import types
from pathlib import Path

import pytest

from agromaster import config, download

CREDENTIAL_ENV_VARS = ["KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"]


@pytest.fixture
def no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the kaggle config dir at an empty folder and clear credential env vars."""
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    folder = tmp_path / "kaggle_cfg"
    folder.mkdir()
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(folder))
    return folder


class FakeKaggleApi:
    """Stand-in for KaggleApi that records calls and writes a tiny CSV."""

    calls: list[tuple[str, str]] = []

    def authenticate(self) -> None:
        pass

    def dataset_download_files(
        self, dataset: str, path: str, unzip: bool, quiet: bool
    ) -> None:
        FakeKaggleApi.calls.append((dataset, path))
        (Path(path) / "prices.csv").write_text("a,b\n1,2\n")


@pytest.fixture
def fake_kaggle(monkeypatch: pytest.MonkeyPatch) -> type[FakeKaggleApi]:
    """Install a fake ``kaggle.api.kaggle_api_extended`` module."""
    FakeKaggleApi.calls = []
    module = types.ModuleType("kaggle.api.kaggle_api_extended")
    module.KaggleApi = FakeKaggleApi
    monkeypatch.setitem(sys.modules, "kaggle", types.ModuleType("kaggle"))
    monkeypatch.setitem(sys.modules, "kaggle.api", types.ModuleType("kaggle.api"))
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", module)
    return FakeKaggleApi


def test_kaggle_dataset_is_set() -> None:
    owner, _, name = config.KAGGLE_DATASET.partition("/")
    assert owner and name


def test_kaggle_config_dir_uses_env(no_credentials: Path) -> None:
    assert download.kaggle_config_dir() == no_credentials


def test_kaggle_config_dir_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    assert download.kaggle_config_dir() == Path.home() / ".kaggle"


def test_has_credentials_false_when_nothing_set(no_credentials: Path) -> None:
    assert not download.has_kaggle_credentials()


@pytest.mark.parametrize("filename", ["kaggle.json", "access_token"])
def test_has_credentials_from_file(no_credentials: Path, filename: str) -> None:
    (no_credentials / filename).write_text("{}")
    assert download.has_kaggle_credentials()


def test_has_credentials_from_api_token_env(
    no_credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    assert download.has_kaggle_credentials()


def test_has_credentials_needs_both_username_and_key(
    no_credentials: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    assert not download.has_kaggle_credentials()
    monkeypatch.setenv("KAGGLE_KEY", "key")
    assert download.has_kaggle_credentials()


def test_is_downloaded(tmp_path: Path) -> None:
    dest = tmp_path / "raw"
    assert not download.is_downloaded(dest)
    dest.mkdir()
    (dest / ".gitkeep").write_text("")
    assert not download.is_downloaded(dest)
    (dest / "sub").mkdir()
    (dest / "sub" / "prices.csv").write_text("a\n1\n")
    assert download.is_downloaded(dest)


def test_download_skips_when_csv_present(
    tmp_path: Path, no_credentials: Path, fake_kaggle: type[FakeKaggleApi]
) -> None:
    (tmp_path / "prices.csv").write_text("a\n1\n")
    # No credentials: the skip must happen before the token check.
    assert download.download_dataset(dataset="owner/name", dest=tmp_path) == tmp_path
    assert fake_kaggle.calls == []


def test_download_raises_clear_error_without_token(
    tmp_path: Path, no_credentials: Path, fake_kaggle: type[FakeKaggleApi]
) -> None:
    with pytest.raises(download.KaggleTokenMissingError, match="kaggle.json"):
        download.download_dataset(dataset="owner/name", dest=tmp_path / "raw")
    assert fake_kaggle.calls == []
    assert not (tmp_path / "raw").exists()


def test_force_with_data_present_still_needs_token(
    tmp_path: Path, no_credentials: Path, fake_kaggle: type[FakeKaggleApi]
) -> None:
    (tmp_path / "prices.csv").write_text("a\n1\n")
    with pytest.raises(download.KaggleTokenMissingError):
        download.download_dataset(dataset="owner/name", dest=tmp_path, force=True)


def test_download_calls_kaggle_api_with_token(
    tmp_path: Path, no_credentials: Path, fake_kaggle: type[FakeKaggleApi]
) -> None:
    (no_credentials / "kaggle.json").write_text("{}")
    dest = tmp_path / "raw"
    assert download.download_dataset(dataset="owner/name", dest=dest) == dest
    assert fake_kaggle.calls == [("owner/name", str(dest))]
    assert download.is_downloaded(dest)


def test_empty_dataset_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "KAGGLE_DATASET", "")
    with pytest.raises(ValueError, match="KAGGLE_DATASET"):
        download.download_dataset(dest=tmp_path)

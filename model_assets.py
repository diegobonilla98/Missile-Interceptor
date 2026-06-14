from pathlib import Path
from urllib.request import urlretrieve


model_repo = "Boni98/MissileInterceptor"
checkpoint_filename = "best.pt"
xgboost_filename = "xgboost_guidance.joblib"
model_directory = Path("models")
checkpoint_url = f"https://huggingface.co/{model_repo}/resolve/main/{checkpoint_filename}"
xgboost_url = f"https://huggingface.co/{model_repo}/resolve/main/{xgboost_filename}"


def download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary_path = path.with_suffix(path.suffix + ".download")
        urlretrieve(url, temporary_path)
        temporary_path.replace(path)
    return path


def resolve_checkpoint_path(path: str | Path | None = None, auto_download: bool = True) -> Path:
    if path is not None:
        return Path(path)
    default_path = model_directory / checkpoint_filename
    if default_path.exists() or auto_download:
        return download_file(checkpoint_url, default_path) if auto_download else default_path
    return default_path


def resolve_xgboost_path(path: str | Path | None = None, auto_download: bool = True) -> Path:
    if path is not None:
        return Path(path)
    default_path = model_directory / xgboost_filename
    if default_path.exists() or auto_download:
        return download_file(xgboost_url, default_path) if auto_download else default_path
    return default_path

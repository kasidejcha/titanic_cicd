from classification_model.config.core import PACKAGE_ROOT


with open(PACKAGE_ROOT / "version.txt") as version_file:
    __version__ = version_file.read().strip()
import json
import zipfile
from pathlib import Path

import qt
import slicer
# NOTE: PyGithub is imported lazily inside the methods that use it, so this module -
# and therefore the whole extension - loads even before dependencies are installed, as the
# Slicer guidelines require. The Dependencies preflight installs PyGithub on demand.


class PythonDependencyChecker:
    """Downloads the upstream model weights and keeps them up to date.

    Package installation lives in the Dependencies module; this class only handles the weights
    published in the upstream GitHub release.
    """

    def __init__(self, repoPath=None, destWeightFolder=None):
        from .SegmentationWidget import SegmentationWidget
        self.destWeightFolder = Path(destWeightFolder or SegmentationWidget.nnUnetFolder())
        self.repo_path = repoPath or "capenaka/SlicerUpperAirwaySegmentator"
        self._updateChecked = False  # ask the GitHub API for a newer release once per session

    def downloadWeightsIfNeeded(self, progressCallback):
        if self.areWeightsMissing():
            return self.downloadWeights(progressCallback)

        if not self._updateChecked:
            self._updateChecked = True
            if self.areWeightsOutdated():
                answer = qt.QMessageBox.question(
                    None,
                    "New weights are available",
                    "New weights are available. Would you like to download them?",
                    qt.QMessageBox.Yes | qt.QMessageBox.No,
                )
                if answer == qt.QMessageBox.Yes:
                    return self.downloadWeights(progressCallback)
        return True

    def areWeightsMissing(self):
        return self.getDatasetPath() is None

    def getLatestReleaseUrl(self):
        from github import Github
        g = Github()
        repo = g.get_repo(self.repo_path)
        assets = [asset for release in repo.get_releases() for asset in release.get_assets()]
        return assets[0].browser_download_url

    def areWeightsOutdated(self):
        """True when a newer release exists. Never raises: offline or rate-limited means 'no'."""
        if not self.getWeightDownloadInfoPath().exists():
            return True
        try:
            return self.getLastDownloadedWeights() != self.getLatestReleaseUrl()
        except Exception:  # noqa: BLE001 - no network, API rate limit, PyGithub missing...
            return False

    def getDatasetPath(self):
        try:
            return next(self.destWeightFolder.rglob("dataset.json"))
        except StopIteration:
            return None

    def getWeightDownloadInfoPath(self):
        return self.destWeightFolder / "download_info.json"

    def getLastDownloadedWeights(self):
        if not self.getWeightDownloadInfoPath().exists():
            return None

        with open(self.getWeightDownloadInfoPath(), "r") as f:
            return json.loads(f.read()).get("download_url")

    def downloadWeights(self, progressCallback):
        import shutil
        import requests

        progressCallback("Downloading model weights...")
        if self.destWeightFolder.exists():
            shutil.rmtree(self.destWeightFolder)
        self.destWeightFolder.mkdir(parents=True, exist_ok=True)

        try:
            download_url = self.getLatestReleaseUrl()
            session = requests.Session()
            response = session.get(download_url, stream=True)
            response.raise_for_status()

            file_name = download_url.split("/")[-1]
            destZipPath = self.destWeightFolder / file_name
            with open(destZipPath, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)

            self.extractWeightsToWeightsFolder(destZipPath)
            self.writeDownloadInfoURL(download_url)
            return True
        except Exception:  # noqa
            import traceback
            slicer.util.errorDisplay(
                "Failed to download weights. Please retry or manually install them to proceed.\n"
                "To manually install the weights, please refer to the documentation here :\n"
                "https://github.com/capenaka/SlicerUpperAirwaySegmentator",
                detailedText=traceback.format_exc()
            )
            return False

    def extractWeightsToWeightsFolder(self, zipPath):
        with zipfile.ZipFile(zipPath, "r") as f:
            f.extractall(self.destWeightFolder)

    def writeDownloadInfoURL(self, download_url):
        with open(self.destWeightFolder / "download_info.json", "w") as f:
            f.write(json.dumps({"download_url": download_url}))

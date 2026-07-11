from pathlib import Path

import qt


def iconPath(icon_name) -> str:
    """Return the path to an icon file shipped in Resources/Icons."""
    return Path(__file__).parent.joinpath("..", "Resources", "Icons", icon_name).as_posix()


def icon(icon_name) -> "qt.QIcon":
    """Load a Resources/Icons file as a QIcon."""
    return qt.QIcon(iconPath(icon_name))

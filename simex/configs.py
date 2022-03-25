from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    corpora = root / 'corpora'


class Heatmap:
    width = 300
    height = 300

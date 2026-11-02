"""findex — навчальний пошуковик: інвертований індекс, BM25, мова запитів і CLI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("findex")
except PackageNotFoundError:  # запуск з сирців без встановлення
    __version__ = "0.0.0"

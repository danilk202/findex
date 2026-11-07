import sys
from typing import Any, cast


def peak_rss_mib() -> float | None:
    """Пік resident set size цього процесу, МіБ (на Windows — peak working set)."""
    if sys.platform != "win32":
        import resource  # лише Unix

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # macOS: байти
    try:
        import psutil  # Windows: peak working set = пік RSS

        info = cast(Any, psutil.Process().memory_info())
        return float(info.peak_wset) / 2**20
    except (ImportError, AttributeError):
        pass
    return _win_peak_wset_mib()


def _win_peak_wset_mib() -> float | None:
    """Без psutil (напр. під 3.13t на Windows): GetProcessMemoryInfo через ctypes."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    windll = cast(Any, ctypes).windll
    handle = windll.kernel32.GetCurrentProcess()
    if not windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
        return None
    return c.PeakWorkingSetSize / 2**20

"""Minimal compatibility shim for CTM training imports.

The upstream CTM training code imports ``nvidia_smi`` unconditionally, even
when GPU usage logging is disabled. The runtime environment used for these
experiments does not provide that optional Python package. This shim keeps the
import path available for no-GAN CTM training; VRAM limits are monitored by the
launcher via the system ``nvidia-smi`` command instead.
"""


class _Utilization:
    gpu = 0
    memory = 0


class _MemoryInfo:
    total = 0
    free = 0
    used = 0


def nvmlInit():
    return None


def nvmlDeviceGetCount():
    return 0


def nvmlDeviceGetHandleByIndex(index):
    return index


def nvmlDeviceGetUtilizationRates(handle):
    return _Utilization()


def nvmlDeviceGetMemoryInfo(handle):
    return _MemoryInfo()

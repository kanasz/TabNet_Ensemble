"""Background sampler of memory / CPU / GPU use of this process and all its children.

The SOTA runners start upstream code in subprocesses (train.py, main.py workers) and
GA-HESO's MeanShift uses joblib worker processes, so in-process counters (tracemalloc,
torch.cuda.max_memory_allocated) miss most of the work. The sampler therefore walks the
whole process tree every `interval` seconds.

Reported:
  peak_rss_mb          max over samples of the summed RSS of the process tree
  cpu_seconds          user+system CPU of this process + last seen CPU of every
                       descendant (children living < interval may be missed)
  peak_gpu_mb          max summed GPU memory of the tree's processes (pynvml; None
                       where the driver does not report per-process memory, e.g.
                       Windows WDDM)
  gpu_util_mean/max    utilisation (%) of the GPUs the tree is using, while sampled
"""
import os
import threading
import time

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:  # pynvml missing, no driver, no GPU
    _NVML = False


def _cpu(times):
    return times.user + times.system


class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.peak_rss = 0
        self.peak_gpu = None
        self.child_cpu = {}  # pid -> last seen cpu seconds
        self.gpu_util = []
        self.samples = 0

    # ---------------------------------------------------------------- sampling
    def _tree(self):
        try:
            return [self.proc] + self.proc.children(recursive=True)
        except psutil.Error:
            return [self.proc]

    def _sample_gpu(self, pids):
        if not _NVML:
            return
        used, util, any_reported = 0, [], False
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except pynvml.NVMLError:
                continue
            mine = [p for p in procs if p.pid in pids]
            if not mine:
                continue
            for p in mine:
                if p.usedGpuMemory is not None:
                    used += p.usedGpuMemory
                    any_reported = True
            try:
                util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except pynvml.NVMLError:
                pass
        if any_reported:
            self.peak_gpu = max(self.peak_gpu or 0, used)
        if util:
            self.gpu_util.append(sum(util) / len(util))

    def sample(self):
        tree = self._tree()
        rss = 0
        for p in tree:
            try:
                rss += p.memory_info().rss
                if p.pid != self.proc.pid:
                    self.child_cpu[p.pid] = _cpu(p.cpu_times())
            except psutil.Error:
                pass
        self.peak_rss = max(self.peak_rss, rss)
        self._sample_gpu({p.pid for p in tree})
        self.samples += 1

    def _loop(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------- API
    def __enter__(self):
        self._cpu0 = _cpu(self.proc.cpu_times())
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.sample()
        self.own_cpu = _cpu(self.proc.cpu_times()) - self._cpu0
        return False

    def result(self):
        return {
            'peak_rss_mb': round(self.peak_rss / 2 ** 20, 1),
            'cpu_seconds': round(self.own_cpu + sum(self.child_cpu.values()), 2),
            'cpu_seconds_own_process': round(self.own_cpu, 2),
            'cpu_seconds_children': round(sum(self.child_cpu.values()), 2),
            'peak_gpu_mb': None if self.peak_gpu is None else round(self.peak_gpu / 2 ** 20, 1),
            'gpu_util_mean': round(sum(self.gpu_util) / len(self.gpu_util), 1) if self.gpu_util else None,
            'gpu_util_max': max(self.gpu_util) if self.gpu_util else None,
            'monitor_samples': self.samples,
            'monitor_interval_s': self.interval,
        }

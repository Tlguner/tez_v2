# resource_monitor.py
import psutil
import threading
import time
import os

class ResourceMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_usages = []
        self.ram_usages_mb = []
        self.is_running = False
        self.thread = None

    def _monitor(self):
        # Initialize CPU percent measurement
        self.process.cpu_percent(interval=None)
        while self.is_running:
            try:
                # CPU percent for this process (can be >100% on multi-core systems)
                cpu = self.process.cpu_percent(interval=None)
                # RAM RSS in MB
                ram = self.process.memory_info().rss / (1024 * 1024)
                self.cpu_usages.append(cpu)
                self.ram_usages_mb.append(ram)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.cpu_usages = []
        self.ram_usages_mb = []
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if not self.cpu_usages:
            self.cpu_usages = [0.0]
        if not self.ram_usages_mb:
            self.ram_usages_mb = [0.0]
            
        return {
            "avg_cpu_percent": round(sum(self.cpu_usages) / len(self.cpu_usages), 2),
            "max_cpu_percent": round(max(self.cpu_usages), 2),
            "avg_ram_mb": round(sum(self.ram_usages_mb) / len(self.ram_usages_mb), 2),
            "max_ram_mb": round(max(self.ram_usages_mb), 2)
        }

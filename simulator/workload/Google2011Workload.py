"""Long-lived Google tasks indexed by global source time; no modulo repetition."""
from pathlib import Path
import numpy as np
from .Workload import Workload
from simulator.container.IPSModels.IPSM import IPSM


class GlobalResource:
    def __init__(self, series):
        self.series=series

    def allocContainer(self, container):self.container=container

    def current(self):
        t=max(0,self.container.env.interval-1)
        if t>=len(self.series):raise IndexError('Google source exhausted; trace repetition is forbidden')
        return self.series[t]


class GlobalCPU(GlobalResource, IPSM):
    def __init__(self, series):
        GlobalResource.__init__(self,series)
        self.SLA=1e12
        self.totalInstructions=1e30
        self.completedInstructions=0.

    def getIPS(self):return float(self.current()[0])
    def getMaxIPS(self):return max(1860.,self.getIPS())


class GlobalRAM(GlobalResource):
    def ram(self):return float(self.current()[1]),1.,1.


class GlobalDisk(GlobalResource):
    def disk(self):return float(self.current()[2]),1.,1.


class Google2011Workload(Workload):
    def __init__(self, trace_path):
        super().__init__()
        with np.load(Path(trace_path)) as data:
            self.series=data['resource_series'];self.source_ids=data['source_ids']
        if self.series.ndim!=3 or self.series.shape[1:]!=(16,3):raise ValueError('Expected 16 source tasks and three resources')
        if not np.isfinite(self.series).all() or (self.series<0).any():raise ValueError('Invalid source resources')
        self.possible_indices=self.source_ids.tolist()

    def generateNewContainers(self, interval):
        if not self.createdContainers:
            for task in range(16):
                values=self.series[:,task]
                self.createdContainers.append((task,interval,GlobalCPU(values),GlobalRAM(values),GlobalDisk(values)))
                self.deployedContainers.append(False)
            self.creation_id=16
        return self.getUndeployedContainers()

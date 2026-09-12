import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scheduler.GOBI import GOBIScheduler

plt.style.use(['science'])
plt.rcParams["text.usetex"] = False

class Stats():
	feats_per_host = 3  # 3 (cpu, ram, disk) or 7 (version2 layout); see saveHostInfo

	def __init__(self, WorkloadModel, Datacenter, Scheduler):
		self.env = None
		self.workload = WorkloadModel
		self.datacenter = Datacenter
		self.scheduler = Scheduler
		# Per-interval bookkeeping history.  ``None`` (the default) keeps every
		# interval, exactly as before.  A positive value keeps only the most
		# recent N intervals, which is all any consumer reads: the recovery
		# models use ``metrics[-1]`` and the online dataset loader uses
		# ``hostinfo[r-1]`` for the last window only, while ``time_series`` and
		# ``schedule_series`` are unaffected and still grow exactly as before.
		# Protocol-022 problem P22-18: with full history a 3400-interval stream
		# spent ~1.4 GB on bookkeeping that nothing reads, which alone exceeded
		# the registered resource guard.
		self.history_limit = None
		# Optional bound on the *series* row history.  ``None`` keeps every
		# interval (previous behaviour).  A positive value retains only the most
		# recent N rows, which is all the online path reads: the recovery models
		# slice ``time_series[-LATEST_WINDOW_SIZE:]`` and the online dataset
		# loader slices ``[-LATEST_WINDOW_SIZE:]`` as well.  Protocol-022 problem
		# P22-20: after the np.append and bookkeeping fixes the remaining
		# per-interval growth was exactly this history, which the collector feeds
		# to a no-op recovery model and never reads again.
		self.series_tail = None

	def recordHistory(self, name, value):
		"""Append to a bookkeeping list, honouring ``history_limit``."""
		series = getattr(self, name)
		series.append(value)
		limit = getattr(self, 'history_limit', None)
		if limit is not None and len(series) > limit:
			del series[:len(series) - limit]
		return series

	def setEnvironment(self, env):
		self.env = env
		self.time_series = np.zeros((1, self.feats_per_host * self.env.hostlimit))
		self.schedule_series = np.zeros((1, self.env.containerlimit, self.env.hostlimit))
		self.simulated_scheduler = GOBIScheduler('energy_latency_'+str(self.datacenter.num_hosts))
		self.simulated_scheduler.env = self.env
		self.initStats()

	def initStats(self):	
		self.hostinfo = []
		self.workloadinfo = []
		self.activecontainerinfo = []
		self.allcontainerinfo = []
		self.metrics = []
		self.schedulerinfo = []

	def saveHostInfo(self):
		hostinfo = dict()
		hostinfo['interval'] = self.env.interval
		hostinfo['cpu'] = [host.getCPU() for host in self.env.hostlist]
		hostinfo['numcontainers'] = [len(self.env.getContainersOfHost(i)) for i,host in enumerate(self.env.hostlist)]
		hostinfo['power'] = [host.getPower() for host in self.env.hostlist]
		hostinfo['baseips'] = [host.getBaseIPS() for host in self.env.hostlist]
		hostinfo['ipsavailable'] = [host.getIPSAvailable() for host in self.env.hostlist]
		hostinfo['ipscap'] = [host.ipsCap for host in self.env.hostlist]
		hostinfo['apparentips'] = [host.getApparentIPS() for host in self.env.hostlist]
		hostinfo['ram'] = [host.getCurrentRAM() for host in self.env.hostlist]
		hostinfo['ramavailable'] = [host.getRAMAvailable() for host in self.env.hostlist]
		hostinfo['disk'] = [host.getCurrentDisk() for host in self.env.hostlist]
		hostinfo['diskavailable'] = [host.getDiskAvailable() for host in self.env.hostlist]
		cpulist, ramlist, disklist = hostinfo['cpu'], [i[0] for i in hostinfo['ram']], [i[0] for i in hostinfo['disk']]
		if self.feats_per_host == 7:
			# version2 layout per host: [cpu, ram, ram_r, ram_w, disk, disk_r, disk_w]
			# rw 列录制真实容器读写需求（read/write 分量）：在线语义与训练数据
			# （rw = 每 host 容器数派生）一致，模型离线/在线看到的通道分布一致。
			ramrlist = [i[1] for i in hostinfo['ram']]
			ramwlist = [i[2] for i in hostinfo['ram']]
			diskrlist = [i[1] for i in hostinfo['disk']]
			diskwlist = [i[2] for i in hostinfo['disk']]
			datapoint = np.concatenate([[cpulist[i], ramlist[i], ramrlist[i], ramwlist[i],
			                             disklist[i], diskrlist[i], diskwlist[i]]
			                            for i in range(len(cpulist))]).reshape(1, -1)
		else:
			datapoint = np.concatenate([[cpulist[i], ramlist[i], disklist[i]]
			                            for i in range(len(cpulist))]).reshape(1, -1)
		self.appendSeries('time_series', datapoint)
		schedule_point = np.array([self.env.scheduler.result_cache])
		self.appendSeries('schedule_series', schedule_point)
		self.recordHistory('hostinfo', hostinfo)

	def appendSeries(self, name, datapoint):
		"""Append one interval to a series with amortised capacity growth.

		The original form was ``series = np.append(series, datapoint, axis=0)``
		on every interval, which reallocates and copies the whole series each
		time: for 3400 intervals that is quadratic in time and an unbounded
		per-step RSS increase, and it aborts the registered 3.0 GiB resource
		guard on long streams (Protocol-022 problem P22-17).

		The public attribute keeps the EXACT contract consumers rely on
		(``len(stats.time_series)`` counts appended rows, ``series[-k:]`` is the
		last k appended rows, ``series[0]`` is the initial zero row): the
		attribute is a view trimmed to the used length, while the allocation
		behind it grows geometrically in a private buffer.  Values, order and
		dtype are unchanged -- only the reserved capacity differs.
		"""
		buffer_name = '_buffer_' + name
		used_name = '_used_' + name
		buffer = getattr(self, buffer_name, None)
		if buffer is None:
			buffer = getattr(self, name)
			setattr(self, buffer_name, buffer)
			setattr(self, used_name, buffer.shape[0])
		used = getattr(self, used_name)
		rows = datapoint.shape[0]
		if used + rows > buffer.shape[0]:
			new_capacity = max(buffer.shape[0] * 2, used + rows, 8)
			grown = np.zeros((new_capacity,) + buffer.shape[1:], dtype=buffer.dtype)
			grown[:used] = buffer[:used]
			buffer = grown
			setattr(self, buffer_name, buffer)
		buffer[used:used + rows] = datapoint
		used += rows
		setattr(self, used_name, used)
		tail = getattr(self, 'series_tail', None)
		if tail is not None and used > tail:
			# Keep only the most recent ``tail`` rows.  The retained arithmetic
			# and the exposed contract are unchanged: ``len()`` still reports the
			# rows kept, and every consumer of the online path reads the tail.
			kept = buffer[used - tail:used].copy()
			setattr(self, buffer_name, kept)
			setattr(self, used_name, kept.shape[0])
			setattr(self, name, kept)
			return
		setattr(self, name, buffer[:used])       # trimmed view: len() is honest

	def seriesLength(self, name):
		"""Number of appended rows in a series (its historical length)."""
		return getattr(self, '_used_' + name, getattr(self, name).shape[0])

	def saveWorkloadInfo(self, deployed, migrations):
		workloadinfo = dict()
		workloadinfo['interval'] = self.env.interval
		workloadinfo['totalcontainers'] = len(self.workload.createdContainers)
		if self.workloadinfo != []:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers'] - self.workloadinfo[-1]['totalcontainers'] 
		else:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers']
		workloadinfo['deployed'] = len(deployed)
		workloadinfo['migrations'] = len(migrations)
		workloadinfo['inqueue'] = len(self.workload.getUndeployedContainers())
		self.recordHistory('workloadinfo', workloadinfo)

	def saveContainerInfo(self):
		containerinfo = dict()
		containerinfo['interval'] = self.env.interval
		containerinfo['activecontainers'] = self.env.getNumActiveContainers()
		containerinfo['ips'] = [(c.getBaseIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['ram'] = [(c.getRAM() if c else 0) for c in self.env.containerlist]
		containerinfo['disk'] = [(c.getDisk() if c else 0) for c in self.env.containerlist]
		containerinfo['creationids'] = [(c.creationID if c else -1) for c in self.env.containerlist]
		containerinfo['hostalloc'] = [(c.getHostID() if c else -1) for c in self.env.containerlist]
		containerinfo['active'] = [(c.active if c else False) for c in self.env.containerlist]
		self.recordHistory('activecontainerinfo', containerinfo)

	def saveAllContainerInfo(self):
		containerinfo = dict()
		allCreatedContainers = [self.env.getContainerByCID(cid) for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['interval'] = self.env.interval
		if self.datacenter.__class__.__name__ == 'Datacenter':
			containerinfo['application'] = [self.env.getContainerByCID(cid).application for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['ips'] = [(c.getBaseIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['create'] = [(c.createAt) for c in allCreatedContainers]
		containerinfo['start'] = [(c.startAt) for c in allCreatedContainers]
		containerinfo['destroy'] = [(c.destroyAt) for c in allCreatedContainers]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['ram'] = [(c.getRAM() if c.active else 0) for c in allCreatedContainers]
		containerinfo['disk'] = [(c.getDisk() if c.active else 0) for c in allCreatedContainers]
		containerinfo['hostalloc'] = [(c.getHostID() if c.active else -1) for c in allCreatedContainers]
		containerinfo['active'] = [(c.active) for c in allCreatedContainers]
		self.recordHistory('allcontainerinfo', containerinfo)

	def saveMetrics(self, destroyed, migrations):
		metrics = dict()
		metrics['interval'] = self.env.interval
		metrics['numdestroyed'] = len(destroyed)
		metrics['nummigrations'] = len(migrations)
		metrics['energy'] = [host.getPower()*self.env.intervaltime for host in self.env.hostlist]
		metrics['energytotalinterval'] = np.sum(metrics['energy'])
		metrics['energypercontainerinterval'] = np.sum(metrics['energy'])/self.env.getNumActiveContainers()
		metrics['responsetime'] = [c.totalExecTime + c.totalMigrationTime for c in destroyed]
		metrics['avgresponsetime'] = np.average(metrics['responsetime']) if len(destroyed) > 0 else 0
		metrics['migrationtime'] = [c.totalMigrationTime for c in destroyed]
		metrics['avgmigrationtime'] = np.average(metrics['migrationtime']) if len(destroyed) > 0 else 0
		metrics['slaviolations'] = len(np.where([c.destroyAt > c.sla for c in destroyed]))
		metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed) if len(destroyed) > 0 else 0
		metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]
		# metrics['energytotalinterval_pred'], metrics['avgresponsetime_pred'] = self.runSimulationGOBI()
		self.recordHistory('metrics', metrics)

	def saveSchedulerInfo(self, selectedcontainers, decision, schedulingtime):
		schedulerinfo = dict()
		schedulerinfo['interval'] = self.env.interval
		schedulerinfo['selection'] = selectedcontainers
		schedulerinfo['decision'] = decision
		schedulerinfo['schedule'] = [(c.id, c.getHostID()) if c else (None, None) for c in self.env.containerlist]
		schedulerinfo['schedulingtime'] = schedulingtime
		if self.datacenter.__class__.__name__ == 'Datacenter':
			schedulerinfo['migrationTime'] = self.env.intervalAllocTimings[-1]
		self.recordHistory('schedulerinfo', schedulerinfo)

	def saveStats(self, deployed, migrations, destroyed, selectedcontainers, decision, schedulingtime):	
		self.saveHostInfo()
		self.saveWorkloadInfo(deployed, migrations)
		self.saveContainerInfo()
		self.saveAllContainerInfo()
		self.saveMetrics(destroyed, migrations)
		self.saveSchedulerInfo(selectedcontainers, decision, schedulingtime)

	def runSimulationGOBI(self):
		host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
		for i in range(len(self.env.hostlist)):
			host_alloc.append([])
		for c in self.env.containerlist:
			if c and c.getHostID() != -1: 
				host_alloc[c.getHostID()].append(c.id) 
				container_alloc[c.id] = c.getHostID()
		selected = self.simulated_scheduler.selection()
		decision = self.simulated_scheduler.filter_placement(self.simulated_scheduler.placement(selected))
		for cid, hid in decision:
			if self.env.getPlacementPossible(cid, hid) and container_alloc[cid] != -1:
				host_alloc[container_alloc[cid]].remove(cid)
				host_alloc[hid].append(cid)
		energytotalinterval_pred = 0
		for hid, cids in enumerate(host_alloc):
			ips = 0
			for cid in cids: ips += self.env.containerlist[cid].getApparentIPS()
			energytotalinterval_pred += self.env.hostlist[hid].getPowerFromIPS(ips)
		return energytotalinterval_pred*self.env.intervaltime, max(0, np.mean([metric_d['avgresponsetime'] for metric_d in self.metrics[-5:]]))

	def runSimulation(self, schedule_data):
		host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
		for i in range(len(self.env.hostlist)): host_alloc.append([])
		for c in self.env.containerlist:
			if c and c.getHostID() != -1: 
				host_alloc[c.getHostID()].append(c.id) 
				container_alloc[c.id] = c.getHostID()
		decision = []
		for cid in np.concatenate(host_alloc):
			cid = int(cid)
			one_hot = schedule_data[cid].tolist()
			new_host = one_hot.index(max(one_hot))
			if container_alloc[cid] != new_host: decision.append((cid, new_host))
		decision = self.simulated_scheduler.filter_placement(decision)
		for cid, hid in decision:
			if self.env.getPlacementPossible(cid, hid) and container_alloc[cid] != -1:
				host_alloc[container_alloc[cid]].remove(cid)
				host_alloc[hid].append(cid)
		energytotalinterval_pred = 0
		for hid, cids in enumerate(host_alloc):
			ips = 0
			for cid in cids: ips += self.env.containerlist[cid].getApparentIPS()
			energytotalinterval_pred += self.env.hostlist[hid].getPowerFromIPS(ips)
		return energytotalinterval_pred*self.env.intervaltime, max(0, np.mean([metric_d['avgresponsetime'] for metric_d in self.metrics[-5:]]))

	########################################################################################################

	def generateGraphsWithInterval(self, dirname, listinfo, obj, metric, metric2=None):
		fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True,figsize=(4, 0.5*len(listinfo[0][metric])))
		title = obj + '_' + metric + '_with_interval' 
		totalIntervals = len(listinfo)
		x = list(range(totalIntervals))
		metric_with_interval = []; metric2_with_interval = []
		ylimit = 0; ylimit2 = 0
		for hostID in range(len(listinfo[0][metric])):
			metric_with_interval.append([listinfo[interval][metric][hostID] for interval in range(totalIntervals)])
			ylimit = max(ylimit, max(metric_with_interval[-1]))
			if metric2:
				metric2_with_interval.append([listinfo[interval][metric2][hostID] for interval in range(totalIntervals)])
				ylimit2 = max(ylimit2, max(metric2_with_interval[-1]))
		for hostID in range(len(listinfo[0][metric])):
			axes[hostID].set_ylim(0, max(ylimit, ylimit2))
			axes[hostID].plot(x, metric_with_interval[hostID])
			if metric2:
				axes[hostID].plot(x, metric2_with_interval[hostID])
			axes[hostID].set_ylabel(obj[0].capitalize()+" "+str(hostID))
			axes[hostID].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + title + '.pdf')

	def generateMetricsWithInterval(self, dirname):
		fig, axes = plt.subplots(9, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.metrics)))
		res = {}
		for i,metric in enumerate(['numdestroyed', 'nummigrations', 'energytotalinterval', 'avgresponsetime',\
			 'avgmigrationtime', 'slaviolations', 'slaviolationspercentage', 'waittime', 'energypercontainerinterval']):
			metric_with_interval = [self.metrics[i][metric] for i in range(len(self.metrics))] if metric != 'waittime' else \
				[sum(self.metrics[i][metric]) for i in range(len(self.metrics))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric, fontsize=5)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
			res[metric] = sum(metric_with_interval)
			print("Summation ", metric, " = ", res[metric])
		print('Average energy (sum energy interval / sum numdestroyed) = ', res['energytotalinterval']/res['numdestroyed'])
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Metrics' + '.pdf')

	def generateWorkloadWithInterval(self, dirname):
		fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.workloadinfo)))
		for i,metric in enumerate(['totalcontainers', 'newcontainers', 'deployed', 'migrations', 'inqueue']):
			metric_with_interval = [self.workloadinfo[i][metric] for i in range(len(self.workloadinfo))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Workload' + '.pdf')

	########################################################################################################

	def generateCompleteDataset(self, dirname, data, name):
		title = name + '_with_interval' 
		metric_with_interval = []
		headers = list(data[0].keys())
		for datum in data:
			metric_with_interval.append([datum[value] for value in datum.keys()])
		df = pd.DataFrame(metric_with_interval, columns=headers)
		df.to_csv(dirname + '/' + title + '.csv', index=False)

	def generateTimeSeriesDataset(self, dirname):
		title = 'time_series'
		np.save(f'{dirname}/time_series.npy', self.time_series)
		np.save(f'{dirname}/schedule_series.npy', self.schedule_series)
		headers = np.concatenate([[f'cpu_{i}', f'ram_{i}', f'disk_{i}'] for i in range(len(self.env.hostlist))])
		df = pd.DataFrame(self.time_series, columns=headers)
		df.to_csv(dirname + '/' + title + '.csv', index=False)

	def generateDatasetWithInterval(self, dirname, metric, objfunc, metric2=None, objfunc2=None):
		title = metric + '_' + (metric2 + '_' if metric2 else "") + (objfunc + '_' if objfunc else "") + (objfunc2 + '_' if objfunc2 else "") + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []; metric2_with_interval = [] # metric1 is of host and metric2 is of containers
		host_alloc_with_interval = []; objfunc2_with_interval = []
		objfunc_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.hostinfo[interval][metric][hostID] for hostID in range(len(self.hostinfo[0][metric]))])
			host_alloc_with_interval.append([self.activecontainerinfo[interval]['hostalloc'][cID] for cID in range(len(self.activecontainerinfo[0]['hostalloc']))])
			objfunc_with_interval.append(self.metrics[interval+1][objfunc])
			if metric2:
				metric2_with_interval.append(self.activecontainerinfo[interval][metric2])
			if objfunc2:
				objfunc2_with_interval.append(self.metrics[interval+1][objfunc2])
		df = pd.DataFrame(metric_with_interval)
		if metric2: df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
		if objfunc2: df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateDatasetWithInterval2(self, dirname, metric, metric2, metric3, metric4, objfunc, objfunc2):
		title = metric + '_' + metric2 + '_'  + metric3 + '_'  + metric4 + '_'  +objfunc + '_' + objfunc2 + '_' + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []; metric2_with_interval = [] 
		metric3_with_interval = []; metric4_with_interval = []
		host_alloc_with_interval = []; objfunc2_with_interval = []
		objfunc_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.hostinfo[interval][metric][hostID] for hostID in range(len(self.hostinfo[0][metric]))])
			host_alloc_with_interval.append([self.activecontainerinfo[interval]['hostalloc'][cID] for cID in range(len(self.activecontainerinfo[0]['hostalloc']))])
			objfunc_with_interval.append(self.metrics[interval+1][objfunc])
			metric2_with_interval.append(self.activecontainerinfo[interval][metric2])
			metric3_with_interval.append(self.metrics[interval][metric3])
			metric4_with_interval.append(self.metrics[interval][metric4])
			objfunc2_with_interval.append(self.metrics[interval+1][objfunc2])
		df = pd.DataFrame(metric_with_interval)
		df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(metric3_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(metric4_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateGraphs(self, dirname):
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'cpu')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'numcontainers')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'power')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'baseips', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'ipscap', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.activecontainerinfo, 'container', 'ips', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.activecontainerinfo, 'container', 'hostalloc')
		# self.generateMetricsWithInterval(dirname)
		# self.generateWorkloadWithInterval(dirname)
		self.generateTimeSeriesDataset(dirname)
		self.generateMetricsWithInterval(dirname)

	def report_metrics(self, model, train_time_data, env_name='simulator'):
		from metrics.evaluation import FaultMetricsEvaluator
		evaluator = FaultMetricsEvaluator(self, model, train_time_data)
		return evaluator.run(env_name)

	def generateDatasets(self, dirname):
		# self.generateDatasetWithInterval(dirname, 'cpu', objfunc='energytotalinterval')
		self.generateDatasetWithInterval(dirname, 'cpu', metric2='apparentips', objfunc='energytotalinterval', objfunc2='avgresponsetime')
		# self.generateDatasetWithInterval2(dirname, 'cpu', 'apparentips', 'energytotalinterval_pred', 'avgresponsetime_pred', objfunc='energytotalinterval', objfunc2='avgresponsetime')
		
	def generateCompleteDatasets(self, dirname):
		self.generateCompleteDataset(dirname, self.hostinfo, 'hostinfo')
		self.generateCompleteDataset(dirname, self.workloadinfo, 'workloadinfo')
		self.generateCompleteDataset(dirname, self.metrics, 'metrics')
		self.generateCompleteDataset(dirname, self.activecontainerinfo, 'activecontainerinfo')
		self.generateCompleteDataset(dirname, self.allcontainerinfo, 'allcontainerinfo')
		self.generateCompleteDataset(dirname, self.schedulerinfo, 'schedulerinfo')
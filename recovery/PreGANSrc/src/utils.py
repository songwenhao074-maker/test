import os
import torch
import numpy as np
from .constants import *
from .models import *

# WINDOW_CURRENT: window [i-w+1 .. i] (INCLUDES the current row i), like the
# schedule window.  This gives the model the same current-step information
# the threshold rule uses (rule baseline F1 ~0.82); the swap collision-pair
# design is not preserved under this windowing (positive/negative windows
# differ), so GRAPH_AUX pairing naturally goes empty.
WINDOW_CURRENT = int(os.environ.get('WINDOW_CURRENT', '0'))

def convert_to_windows(data, model):
	data = torch.tensor(data).double()
	windows = []; w_size = model.n_window
	if WINDOW_CURRENT:
		for i, g in enumerate(data):
			lo = max(0, i - (w_size - 1))
			w = data[lo:i + 1]
			if w.shape[0] < w_size:
				w = torch.cat([data[0].repeat(w_size - w.shape[0], 1), w], dim=0)
			windows.append(w)
		return torch.stack(windows)
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def convert_schedule_to_windows(schedule, model):
	"""Window the schedule the same way as convert_to_windows, but the
	window ENDS at row i (the row whose fault is being predicted).

	The schedule at row i is known to the scheduler at decision time
	(past decisions + the current placement), so the schedule-aware graph
	path legitimately sees rows [i-w+1 .. i].  Front-padding uses row 0,
	exactly like convert_to_windows.
	"""
	schedule = torch.as_tensor(schedule).double()
	if schedule.dim() == 2:
		schedule = schedule.unsqueeze(0)
	w_size = model.n_window
	windows = []
	for i in range(schedule.shape[0]):
		lo = max(0, i - (w_size - 1))
		w = schedule[lo:i + 1]
		if w.shape[0] < w_size:
			w = torch.cat([schedule[0].repeat(w_size - w.shape[0], 1, 1), w], dim=0)
		windows.append(w)
	return torch.stack(windows)

def schedule_window_from_online_stats(stats, current_schedule, w_size=3):
	"""Build the same (past 2 rows + current row) schedule window online.

	stats.schedule_series holds one (n_containers, n_hosts) row per saved
	interval (the scheduler placement decided at that interval); the current
	decision is passed separately.  Front-padding mirrors the offline
	convert_schedule_to_windows convention.
	"""
	hist = np.asarray(stats.schedule_series[-2:], dtype=np.float64)
	current = np.asarray(current_schedule, dtype=np.float64)
	if hist.ndim == 2:
		hist = hist[np.newaxis, :, :]
	if hist.shape[0] == 2:
		window = np.concatenate([hist, current[np.newaxis, :, :]], axis=0)
	elif hist.shape[0] == 1:
		window = np.concatenate([hist, current[np.newaxis, :, :], current[np.newaxis, :, :]], axis=0)
	else:
		window = np.repeat(current[np.newaxis, :, :], w_size, axis=0)
	return torch.as_tensor(window).double()

def form_test_dataset(data):
	anomaly_per_dim = data > np.percentile(data, PERCENTILES, axis=0)
	feats_per_host = 7 if data.shape[1] % 7 == 0 else 3
	anomaly_which_dim, anomaly_any_dim = [], []
	for i in range(0, data.shape[1], feats_per_host):
		if feats_per_host == 7:
			# version2 layout [cpu, ram, ram_r, ram_w, disk, disk_r, disk_w]:
			# fold to 3 resource groups so class labels stay in {0,1,2} and
			# match the 3-prototype convention (cpu / ram / disk).
			grouped = np.stack([
				data[:, i],
				np.max(data[:, i + 1:i + 4], axis=1),
				np.max(data[:, i + 4:i + 7], axis=1),
			], axis=1)
			anomaly_which_dim.append(np.argmax(grouped, axis=1))
			anomaly_any_dim.append(np.logical_or.reduce(
				anomaly_per_dim[:, i:i + feats_per_host], axis=1))
		else:
			anomaly_which_dim.append(np.argmax(data[:, i:i + 3] + 0, axis=1))
			anomaly_any_dim.append(np.logical_or.reduce(
				anomaly_per_dim[:, i:i + 3], axis=1))
	anomaly_any_dim = np.stack(anomaly_any_dim, axis=1)
	anomaly_which_dim = np.stack(anomaly_which_dim, axis=1)
	return anomaly_any_dim + 0, anomaly_which_dim

def load_npyfile(folder, fname):
	path = os.path.join(folder, fname)
	if not os.path.exists(path):
		raise Exception('Data not found ' + path)
	return np.load(path)

def load_dataset(folder, model):
	time_data = load_npyfile(folder, data_filename)
	time_data = normalize_time_data(time_data) # Normalize data
	train_schedule_data = torch.tensor(load_npyfile(folder, schedule_filename)).double()
	train_time_data = convert_to_windows(time_data, model)
	anomaly_data, class_data = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data

def load_on_the_fly_dataset(model, folder, stats):
	train_time_data = load_npyfile(folder, data_filename)
	time_data = stats.time_series[-LATEST_WINDOW_SIZE:]
	time_data = normalize_test_time_data(time_data, train_time_data)
	raw_schedule = stats.schedule_series[-LATEST_WINDOW_SIZE:]
	train_schedule_data = convert_schedule_to_windows(raw_schedule, model)
	train_time_data = convert_to_windows(time_data, model)
	if os.environ.get('ONLINE_LABEL_MODE', '') == 'overload':
		# QoS-overload labels straight from the simulator: a host is
		# anomalous when its total base demand exceeds capacity (cpu
		# first, then ram).  No percentile rule — matches the offline
		# labels_overload_class convention (class 1 -> index 0).
		# Alignment: time_series row 0 is the init zeros row, row r>=1 was
		# recorded together with hostinfo[r-1]; labels are built per
		# time_series row so windows and labels always have equal length
		# (short series pad the head with normal rows).
		n_hosts = time_data.shape[1] // 7
		W = time_data.shape[0]
		n_ts = len(stats.time_series)
		hi = stats.hostinfo
		anomaly_data = np.zeros((W, n_hosts))
		class_data = np.zeros((W, n_hosts), dtype=int)
		for r in range(1, min(n_ts, len(hi) + 1)):
			hinfo = hi[r - 1]
			base = np.asarray(hinfo['baseips'], dtype=float)
			cap = np.asarray(hinfo['ipscap'], dtype=float)
			ram = np.asarray([x[0] for x in hinfo['ram']], dtype=float)
			ramcap = ram + np.asarray([x[0] for x in hinfo['ramavailable']],
									  dtype=float)
			w = W - (n_ts - r)          # time_data row for time_series row r
			if w < 0:
				continue
			for h in range(min(n_hosts, len(base))):
				if base[h] > cap[h]:
					anomaly_data[w, h] = 1
					class_data[w, h] = 0
				elif ram[h] > ramcap[h]:
					anomaly_data[w, h] = 1
					class_data[w, h] = 1
		# numpy convention (form_test_dataset returns numpy arrays; backprop
		# indexes them per-host and uses scalar ints for prototype[]).
		return train_time_data, train_schedule_data, anomaly_data, class_data
	anomaly_data, class_data = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data

def save_model(folder, fname, model, optimizer, epoch, accuracy_list):
	path = os.path.join(folder, fname)
	# if 'Att' in model.name: print(model.prototype)
	if 'G' in model.name or 'D' in model.name: model.prototype = {}
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_prototypes': model.prototype,
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, path)

def load_model(folder, fname, modelname):
	import recovery.PreGANSrc.src.models
	import recovery.PreGANSrc.src.models_v2
	path = os.path.join(folder, fname)
	model_class = getattr(recovery.PreGANSrc.src.models, modelname, None)
	if model_class is None:
		model_class = getattr(recovery.PreGANSrc.src.models_v2, modelname)
	model = model_class().double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	if os.path.exists(path):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(path, weights_only=False)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.prototype = checkpoint['model_prototypes']
		for p in model.prototype: p.requires_grad = False
		# if hasattr(model, 'loss_weights'): print(model.prototype)
		try:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		except (ValueError, RuntimeError) as e:
			# optimizer param-group mismatch (e.g. multi-LR training) —
			# evaluation does not need the optimizer state.
			print(f'{color.FAIL}optimizer state skipped: {e}{color.ENDC}')
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, epoch, accuracy_list

def load_gan(folder, gfname, dfname, gmodelname, dmodelname):
	gmodel, gopt, epoch, accuracy_list = load_model(folder, gfname, gmodelname)
	dmodel, dopt, _, _ = load_model(folder, dfname, dmodelname)
	return gmodel, dmodel, gopt, dopt, epoch, accuracy_list

def save_gan(folder, gfname, dfname, gmodel, dmodel, gopt, dopt, epoch, accuracy_list):
	save_model(folder, gfname, gmodel, gopt, epoch, accuracy_list)
	save_model(folder, dfname, dmodel, dopt, 0, [])

# Misc
def normalize_time_data(time_data):
	return time_data / (np.max(time_data, axis = 0) + 1e-8) 

def normalize_test_time_data(time_data, train_time_data):
	return (time_data / (np.max(train_time_data, axis = 0) + 1e-8))

def run_simulation(stats, schedule_data):
    e, r = stats.runSimulation(schedule_data)
    score = Coeff_Energy * e + Coeff_Latency * r
    return score

def get_classes(embeddings, model):
	class_list = []
	for e in embeddings:
		if (e == 0).all().item():
			class_list.append(-1); continue
		distances = np.array([(torch.mean((e - p)**2)).item() for p in model.prototype])
		class_list.append(np.argmin(distances))
	return class_list

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

class color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
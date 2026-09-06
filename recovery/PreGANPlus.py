import sys
sys.path.append('recovery/PreGANSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *

class PreGANPlusRecovery(Recovery):
    predictor_name = 'Transformer'
    # Per-method GAN identity: each recovery method gets its own GAN
    # checkpoint (method prefix) and its own per-seed copy (seed suffix),
    # so online tuning of one method/seed never contaminates another.
    gan_key = 'PreGANPlus'

    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.model_name = f'{self.predictor_name}_{hosts}'
        # gen_name/disc_name are MODEL CLASS names ('Gen_16'/'Disc_16',
        # used by load_model's getattr and the plotter); the per-method
        # checkpoint FILE prefix is handled by gan_ckpt_name().
        self.gen_name = f'Gen_{hosts}'
        self.disc_name = f'Disc_{hosts}'
        self.hosts = hosts
        # _env_override lets subclasses (e.g. FTMoEChainRecovery) point the
        # data/checkpoint namespace at a different trace without touching
        # the shared training/env wiring.
        self.env_name = getattr(self, '_env_override', None) or (
            'simulator' if env == '' else 'framework')
        self.training = training
        self.save_gan = True
        self.load_models()

    def gan_ckpt_name(self, model_name):
        """Per-method, per-seed GAN checkpoint file name.  The method prefix
        (self.gan_key) isolates each recovery method's online-tuned GAN;
        the seed suffix (FTMoE_CKPT_SEED) isolates per-seed runs."""
        seed = os.environ.get('FTMoE_CKPT_SEED', '')
        stem = f'{self.env_name}_{self.gan_key}_{model_name}'
        return f'{stem}.ckpt' if not seed else f'{stem}_s{seed}.ckpt'

    def load_models(self):
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model if not trained (offline training same as PreGAN)
        if self.epoch == -1: self.train_model()
        # Reduce lr of encoder
        # self.model.lr /= 5
        # Load generator and discriminator (per-method, per-seed files)
        self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
            load_gan(model_plus_folder, self.gan_ckpt_name(self.gen_name), self.gan_ckpt_name(self.disc_name), self.gen_name, self.disc_name)
        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        # GAN is always tuned
        self.ganloss = nn.BCELoss()
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def tune_model(self):
        # tune for a single epoch
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
        anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None)
        tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        self.accuracy_list.append((loss, factor, anomaly_score, class_score))

    def train_gan(self, embedding, schedule_data):
        # Train discriminator
        self.disc.zero_grad()
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data.detach())
        new_score, orig_score = run_simulation(self.env.stats, new_schedule_data), run_simulation(self.env.stats, schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) if new_score <= orig_score else torch.tensor([1, 0], dtype=torch.double)
        disc_loss = self.ganloss(probs, true_probs.detach().clone())
        disc_loss.backward(); self.dopt.step()
        # Train generator
        self.gen.zero_grad()
        probs = self.disc(schedule_data, new_schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) # to enforce new schedule is better than original schedule
        gen_loss = self.ganloss(probs, true_probs)
        gen_loss.backward(); self.gopt.step()
        # Append to accuracy list and save model
        if self.save_gan:            
            self.epoch += 1; self.accuracy_list.append((gen_loss.item(), disc_loss.item()))
            print(f'{color.HEADER}Epoch {self.epoch},\tGLoss = {gen_loss.item()},\tDLoss = {disc_loss.item()}{color.ENDC}')
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)
            save_gan(model_plus_folder, self.gan_ckpt_name(self.gen_name), self.gan_ckpt_name(self.disc_name), \
                    self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data)
        self.gan_plotter.new_better(probs[1] >= probs[0])
        if probs[0] > probs[1]: # original better
            return original_decision
        # Form new decision
        host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)): host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                host_alloc[c.getHostID()].append(c.id) 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision); hosts_from = [0] * self.hosts
        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            # NOTE: pick the target host from the GAN-generated schedule
            # (new_schedule_data), NOT from the GOBI schedule passed in —
            # the old code read schedule_data here, so the GAN output never
            # reached the decision and recovery degenerated to GOBI.
            one_hot = new_schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            if container_alloc[cid] != new_host:
                decision_dict[cid] = new_host
                hosts_from[container_alloc[cid]] = 1
        self.gan_plotter.plot_test(hosts_from)
        return list(decision_dict.items())

    def run_encoder(self, schedule_data):
        # Get latest data from Stat.  NOTE: do NOT pre-truncate to the window
        # length here — convert_to_windows() pads short inputs with repeated
        # row 0, so truncating first corrupts the final window with pad rows
        # (the pre-2026-08 detection-misalignment bug).  Converting the full
        # series makes the last window exactly the last n_window rows.
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        time_data = convert_to_windows(time_data, self.model)[-1]
        # The graph path expects the same schedule window as offline
        # training: the last two recorded decisions plus the decision made
        # for the current interval.
        schedule_data = schedule_window_from_online_stats(
            self.env.stats, schedule_data, self.model.n_window)
        return self.model(time_data, schedule_data)

    def run_model(self, time_series, original_decision):
        # Run encoder
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        anomaly, prototype = self.run_encoder(schedule_data)
        # If no anomaly predicted, return original decision 
        for a in anomaly:
            prediction = torch.argmax(a).item() 
            if prediction == 1: 
                self.gan_plotter.update_anomaly_detected(1)
                break
        else:
            self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        # Pass through GAN
        self.train_gan(embedding, schedule_data)
        # Tune Model
        self.tune_model()
        return self.recover_decision(embedding, schedule_data, original_decision)

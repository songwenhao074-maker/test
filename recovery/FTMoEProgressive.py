import os
import torch
import torch.nn as nn
from tqdm import tqdm

from .PreGANPlus import PreGANPlusRecovery
from .PreGANSrc.src.constants import data_folder, model_plus_folder, num_epochs, data_filename
from .PreGANSrc.src.plotter import Model_Plotter, GAN_Plotter
from .PreGANSrc.src.train import backprop, accuracy
from .PreGANSrc.src.utils import load_on_the_fly_dataset, save_model, load_model, load_gan, load_npyfile, load_dataset


class FTMoEProgressiveRecovery(PreGANPlusRecovery):
    """Progressive FT-MoE recovery: PreGAN+ training/inference pipeline with
    the FT-MoE innovations added one at a time.

    - predictor_name selects the model variant:
        'FTMoE_v0_16' : exact PreGAN+ baseline (Transformer_16), sanity anchor
        'FTMoE_v1b_16' : + per-host MoE branch (zero-init logit residual)
        'FTMoE_v2c2_16' : + soft adaptive expert gating (softgate)
        'FTMoE_v3b2_16' : + schedule-aware graph interaction residual
        'FTMoE_v4c1_16' : + inter-expert attention inside the MoE
    - Everything else (GAN scheduler, GAN tuning, data loading, checkpointing,
      online fine-tuning cadence) is inherited unchanged from PreGANPlusRecovery.

    The variant is selected at runtime via the FTMoeE_MODEL environment
    variable (default: v4c1, the best 6-seed-average model).
    """

    # Variant selected at runtime via the FTMoE_MODEL environment variable
    # (default: v4c1, the accepted 6-seed-verified model).  NOTE: the name
    # must NOT carry the host suffix — PreGANPlusRecovery.__init__ appends
    # f'_{hosts}' to build the checkpoint name, e.g. 'FTMoE_v4c1' -> 'FTMoE_v4c1_16'.
    predictor_name = os.environ.get('FTMoE_MODEL', 'FTMoE_v4c1')
    # Per-method GAN identity: FT-MoE keeps its own GAN checkpoint, separate
    # from PreGAN+ and PreGAN, so online tuning never leaks across methods.
    gan_key = 'FTMoE'
    online_adaptation_interval = 10

    def __init__(self, hosts, env, training=False):
        self._online_tuning_steps = 0
        super().__init__(hosts, env, training)

    def load_models(self):
        # Reuse the exact PreGAN+ loading routine; the checkpoint name is
        # derived from predictor_name so each variant keeps its own file.
        # An optional per-seed checkpoint is selected via FTMoE_CKPT_SEED
        # (matches train_progressive.py's SEED=N naming, e.g. _s1).
        seed_suffix = os.environ.get('FTMoE_CKPT_SEED', '')
        ckpt_name = f'{self.env_name}_{self.model_name}.ckpt' if not seed_suffix \
            else f'{self.env_name}_{self.model_name}_s{seed_suffix}.ckpt'
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_plus_folder, ckpt_name, self.model_name)
        if self.epoch == -1:
            self.train_model()
        self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
            load_gan(model_plus_folder, self.gan_ckpt_name(self.gen_name), self.gan_ckpt_name(self.disc_name), self.gen_name, self.disc_name)
        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        self.ganloss = nn.BCELoss()
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):
        # Offline training: identical to PreGANPlusRecovery.train_model.
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch + 1, self.epoch + num_epochs + 1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def tune_model(self):
        # Online tuning: same one-epoch fine-tune as PreGAN+, plus the MoE
        # capacity adaptation (remove/add experts) at the configured cadence
        # for variants that expose it.
        folder = os.path.join(data_folder, self.env_name)
        self.model.begin_routing_window() if hasattr(self.model, 'begin_routing_window') else None
        self.model.set_online_tuning(True) if hasattr(self.model, 'set_online_tuning') else None
        train_time_data, train_schedule_data, anomaly_data, class_data = \
            load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
        anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None)
        tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        self.accuracy_list.append((loss, factor, anomaly_score, class_score))
        self._online_tuning_steps += 1
        changed = False
        if self._online_tuning_steps % self.online_adaptation_interval == 0 and hasattr(self.model, 'adapt_experts'):
            changed = self.model.adapt_experts()
        if hasattr(self.model, 'set_online_tuning'):
            self.model.set_online_tuning(False)
        if changed:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model.lr, weight_decay=1e-5)
        return loss

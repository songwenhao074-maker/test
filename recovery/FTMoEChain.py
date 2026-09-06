"""Final-chain v2 model recovery for QoS replay experiments.

Loads the final ablation-chain checkpoints
(recovery/PreGANSrc/checkpoints_final_chain/simulator_FTMoE_<variant>_16_v2_s<seed>_snap85.ckpt)
so the QoS experiment uses the SAME trained models as the paper's ablation
chain (v0 < v1b < v2c2 < v3b2 < v4c1, seeds {1,2,6}).

Everything else (GAN scheduler, online fine-tuning cadence, data loading,
checkpointing) is inherited unchanged from PreGANPlusRecovery.  Online
tuning (tune_model) updates the in-memory encoder only — it never saves
the encoder checkpoint, so the final-chain checkpoints are never
overwritten; the GAN is the only online-saved state and lives in
checkpointsplus/ with its own FTMoE-prefixed names.

Variant selected via FTMoE_MODEL (default v4c1); seed via FTMoE_CKPT_SEED
(default 1).  The _v2 class names are resolved by load_model from
models_v2.py.
"""
import os
import torch
import torch.nn as nn

from .FTMoEProgressive import FTMoEProgressiveRecovery
from .PreGANSrc.src.constants import model_plus_folder
from .PreGANSrc.src.plotter import GAN_Plotter
from .PreGANSrc.src.utils import load_model, load_gan, load_npyfile
from .PreGANSrc.src.constants import data_folder, data_filename
from .PreGANSrc.src.utils import get_classes, run_simulation

# Final ablation chain: one trained model per (variant, seed).
CHAIN_CKPT_DIR = os.environ.get(
    'CHAIN_CKPT_DIR',
    'recovery/PreGANSrc/checkpoints_final_chain')
CHAIN_SEEDS = [1, 2, 6]


class FTMoEChainRecovery(FTMoEProgressiveRecovery):
    """Recovery wrapping a final-chain v2 checkpoint (snap85)."""

    online_adaptation_interval = 10

    def __init__(self, hosts, env, training=False):
        self._variant = os.environ.get('FTMoE_MODEL', 'v4c1')
        if not self._variant.startswith('FTMoE_'):
            self._variant = 'FTMoE_' + self._variant
        self._seed = int(os.environ.get('FTMoE_CKPT_SEED', '1'))
        if self._seed not in CHAIN_SEEDS:
            self._seed = CHAIN_SEEDS[0]
        # The v2 models are trained on the 112-column qos trace; env_name
        # must point at data/qos so online fine-tuning data has 112 columns.
        self._env_override = 'qos'
        super().__init__(hosts, env, training)

    @property
    def predictor_name(self):
        return self._variant + '_16_v2'

    def run_model(self, time_series, original_decision):
        """Same as PreGANPlusRecovery.run_model, plus anomaly-detection
        counting and threshold calibration (DETECT_TAU env).

        DETECT_TAU: instead of argmax (P1 > P0), trigger when the max
        per-host P1 logit exceeds tau.  This lets experiments align the
        trigger rate across variants (v0/v1b trigger every step at argmax,
        v3b2/v4c1 trigger less) so the GAN online-training advantage is
        not dominated by raw trigger frequency."""
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        anomaly, prototype = self.run_encoder(schedule_data)
        tau = float(os.environ.get('DETECT_TAU', '0'))  # 0 = argmax
        # per-host P1 logit (anomaly: [n_hosts, 1, 2])
        p1 = anomaly[:, 0, 1].detach()
        self.last_detected = 0
        if tau > 0:
            if p1.max().item() > tau:
                self.last_detected = 1
        else:
            for a in anomaly:
                if torch.argmax(a).item() == 1:
                    self.last_detected = 1
                    break
        if not self.last_detected:
            self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        self.gan_plotter.update_anomaly_detected(1)
        embedding = [torch.zeros_like(p) if p1[i].item() <= tau else p
                     for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        self.train_gan(embedding, schedule_data)
        if os.environ.get('QOS_DEBUG', '0') == '1':
            with torch.no_grad():
                new_schedule = self.gen(embedding, schedule_data)
            new_score = run_simulation(self.env.stats, new_schedule)
            orig_score = run_simulation(self.env.stats, schedule_data)
            print(f'[qos-debug] trigger: new_score={new_score:.2f} '
                  f'orig_score={orig_score:.2f} '
                  f'better={new_score < orig_score}', flush=True)
        # ONLINE_TUNE=0: skip encoder fine-tuning.  For overload-trained
        # chains the online labels are action-dependent (the model's own
        # migrations eliminate overload, so post-action labels are all
        # zero and fine-tuning trains the detector to predict nothing).
        # The offline-trained model already matches the replay protocol.
        if os.environ.get('ONLINE_TUNE', '1') != '0':
            self.tune_model()
        return self.recover_decision(embedding, schedule_data, original_decision)

    def load_models(self):
        # __init__ normalized self._seed to one of CHAIN_SEEDS; prefer an
        # explicit env override, else fall back to the normalized seed.
        seed_suffix = os.environ.get('FTMoE_CKPT_SEED', str(self._seed))
        # Parent __init__ builds model_name = predictor_name + '_16'; strip
        # the suffix to get the real v2 class name (FTMoE_v4c1_16_v2).
        cls_name = self.model_name.removesuffix('_16')
        ckpt_name = f'simulator_{cls_name}_s{seed_suffix}_snap85.ckpt'
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(CHAIN_CKPT_DIR, ckpt_name, cls_name)
        if self.epoch == -1:
            raise RuntimeError(
                f'final-chain ckpt not found in {CHAIN_CKPT_DIR}: '
                f'{ckpt_name} — model would train from scratch; aborting')
        self.model.eval()
        self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
            load_gan(model_plus_folder,
                     self.gan_ckpt_name(self.gen_name),
                     self.gan_ckpt_name(self.disc_name),
                     self.gen_name, self.disc_name)
        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name,
                                       self.disc_name, self.training)
        self.ganloss = nn.BCELoss()
        self.train_time_data = load_npyfile(
            os.path.join(data_folder, self.env_name), data_filename)

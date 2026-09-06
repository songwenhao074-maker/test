import os
import torch
from tqdm import tqdm

from .PreGANPlus import PreGANPlusRecovery
from .PreGANSrc.src.constants import data_folder, model_plus_folder, num_epochs
from .PreGANSrc.src.plotter import Model_Plotter
from .PreGANSrc.src.train import backprop, accuracy
from .PreGANSrc.src.utils import load_on_the_fly_dataset, save_model


class FTMoERecovery(PreGANPlusRecovery):
    """FT-MoE v2 recovery path using PreGAN+'s existing GAN decision generator."""

    predictor_name = 'FTMoEv2'
    online_adaptation_interval = 10

    def __init__(self, hosts, env, training=False):
        self._online_tuning_steps = 0
        super().__init__(hosts, env, training)

    def tune_model(self):
        """Fine-tune only the MoE, then adapt its capacity from routing data."""
        folder = os.path.join(data_folder, self.env_name)
        self.model.begin_routing_window()
        self.model.set_online_tuning(True)
        train_time_data, train_schedule_data, anomaly_data, class_data = \
            load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(
            self.epoch, self.model, train_time_data, train_schedule_data,
            anomaly_data, class_data, self.optimizer
        )
        anomaly_score, class_score = accuracy(
            self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None
        )
        tqdm.write(
            f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}'
        )
        self.accuracy_list.append((loss, factor, anomaly_score, class_score))
        self._online_tuning_steps += 1
        changed = False
        if self._online_tuning_steps % self.online_adaptation_interval == 0:
            changed = self.model.adapt_experts()

        # Restore the full parameter set for ordinary inference/checkpointing.
        self.model.set_online_tuning(False)
        if changed:
            # Added or removed experts introduce new parameters; rebuilding the
            # optimizer prevents stale parameter references after a topology update.
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model.lr, weight_decay=1e-5)
        return loss

    def train_model(self):
        # Keep the same offline routine as PreGAN+, but store FT-MoE checkpoints
        # under its own model name so Transformer checkpoints remain untouched.
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        from .PreGANSrc.src.utils import load_dataset
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch + 1, self.epoch + num_epochs + 1), position=0):
            loss, factor = backprop(
                self.epoch, self.model, train_time_data, train_schedule_data,
                anomaly_data, class_data, self.optimizer
            )
            anomaly_score, class_score = accuracy(
                self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.model_plotter
            )
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(
                model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt',
                self.model, self.optimizer, self.epoch, self.accuracy_list
            )

import numpy as np
import torch

from recovery.PreGANSrc.src.utils import load_dataset
from recovery.PreGANSrc.src.constants import data_folder


class FaultMetricsEvaluator:
    """Post-simulation evaluator that computes all paper-reported metrics."""

    def __init__(self, stats, model, train_time_data):
        self.stats = stats
        self.model = model
        self.train_time_data = train_time_data

    def evaluate(self, env_name='simulator'):
        metrics = {}
        if self.model is not None and self.train_time_data is not None:
            metrics.update(self._evaluate_fault_detection_and_classification(env_name))
        metrics.update(self._evaluate_qos())
        return metrics

    # ------------------------------------------------------------
    # Fault Detection & Classification
    # ------------------------------------------------------------
    def _evaluate_fault_detection_and_classification(self, env_name):
        import os
        folder = os.path.join(data_folder, env_name)

        # Score against ground-truth labels (generated from offline data via percentiles)
        train_time_data, train_schedule_data, anomaly_labels, class_labels = \
            load_dataset(folder, self.model)

        all_anomaly_preds = []
        all_proto_embs = []

        self.model.eval()
        with torch.no_grad():
            for i in range(len(train_time_data)):
                tw = train_time_data[i]   # (n_window, n_feats)
                sched = train_schedule_data[i]  # (n_containers, n_hosts)
                anomaly_scores, prototypes = self.model(tw, sched)
                all_anomaly_preds.append(anomaly_scores)
                all_proto_embs.append(prototypes)

        # Stack into tensors. Model returns list of n_hosts tensors per timestep.
        # After the loop each list element is one timestep of n_hosts × 1 × 2 / PROTO_DIM.
        # We flatten the extra middle dimension (1) and build (T, H, 2) / (T, H, PROTO_DIM).
        T = len(all_anomaly_preds)
        H = len(all_anomaly_preds[0])

        anomaly_stack = []
        proto_stack = []
        for t in range(T):
            host_anom = []
            host_proto = []
            for h in range(H):
                a = all_anomaly_preds[t][h].reshape(2).cpu().numpy()
                p = all_proto_embs[t][h].cpu().numpy()
                host_anom.append(a)
                host_proto.append(p)
            anomaly_stack.append(np.stack(host_anom, axis=0))   # (H, 2)
            proto_stack.append(np.stack(host_proto, axis=0))     # (H, PROTO_DIM)

        anomaly_preds = np.stack(anomaly_stack, axis=0)  # (T, H, 2)
        proto_embs = np.stack(proto_stack, axis=0)        # (T, H, PROTO_DIM)

        # Detection metrics
        det = self._compute_detection_metrics(anomaly_preds, anomaly_labels)

        # Classification metrics (only on anomalous hosts)
        cls = self._compute_classification_metrics(
            proto_embs, class_labels, anomaly_labels
        )

        return {**det, **cls}

    def _compute_detection_metrics(self, anomaly_preds, anomaly_labels):
        T, H = anomaly_preds.shape[0], anomaly_preds.shape[1]
        pred_labels = np.argmax(anomaly_preds, axis=-1)  # (T, H)

        tp = np.sum((pred_labels == 1) & (anomaly_labels == 1))
        fp = np.sum((pred_labels == 1) & (anomaly_labels == 0))
        tn = np.sum((pred_labels == 0) & (anomaly_labels == 0))
        fn = np.sum((pred_labels == 0) & (anomaly_labels == 1))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def _compute_classification_metrics(self, proto_embs, class_labels, anomaly_labels):
        T, H = proto_embs.shape[0], proto_embs.shape[1]
        prototypes = [p.detach().cpu().numpy() for p in self.model.prototype]

        hr_correct = 0
        hr_total = 0
        ndcg_scores = []

        for t in range(T):
            for h in range(H):
                if anomaly_labels[t, h] == 0:
                    continue
                hr_total += 1
                emb = proto_embs[t, h]
                true_class = class_labels[t, h]

                distances = np.array([
                    np.mean((emb - p_np) ** 2) for p_np in prototypes
                ])
                pred_class = int(np.argmin(distances))

                if pred_class == true_class:
                    hr_correct += 1

                # NDCG: closer distance = higher relevance
                # Rank by increasing distance (closest → rank 1)
                ranks = np.argsort(distances)
                # relevance = 1 at the position of true_class only
                relevance = np.zeros(len(ranks))
                relevance[np.where(ranks == true_class)[0][0]] = 1.0
                dcg = sum(
                    relevance[i] / np.log2(i + 2)  # rank_i = i+1 → log2(rank_i + 1) = log2(i+2)
                    for i in range(len(relevance))
                )
                idcg = 1.0 / np.log2(2.0)  # best case: true class at rank 1
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_scores.append(ndcg)

        hr = hr_correct / max(hr_total, 1)
        ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

        return {
            'hr': hr,
            'ndcg': ndcg,
        }

    # ------------------------------------------------------------
    # QoS
    # ------------------------------------------------------------
    def _evaluate_qos(self):
        hostinfo = self.stats.hostinfo
        metrics = self.stats.metrics

        # CPU Utilization: average across timesteps then across hosts
        cpu_avgs = []
        for hi in hostinfo:
            cpu_avgs.append(np.mean(hi['cpu']))
        avg_cpu = np.mean(cpu_avgs) if cpu_avgs else 0.0

        # RAM Utilization: per-interval: ram_size / (ram_size + ram_avail_size) * 100
        ram_avgs = []
        for hi in hostinfo:
            # ram = list of (size, read, write) tuples; ramavailable = list of (size, read, write) tuples
            rams = np.array([float(r[0]) for r in hi['ram']])
            ram_avail = np.array([float(ra[0]) for ra in hi['ramavailable']])
            denom = rams + ram_avail
            valid = denom > 0
            if valid.sum() > 0:
                ram_avgs.append(np.mean(100.0 * rams[valid] / denom[valid]))
        avg_ram = np.mean(ram_avgs) if ram_avgs else 0.0

        # Energy: sum over all intervals / sum numdestroyed / 1000 → KWhr
        total_energy = sum(m['energytotalinterval'] for m in metrics)
        total_destroyed = max(sum(m['numdestroyed'] for m in metrics), 1)
        avg_energy = total_energy / total_destroyed / 1000.0

        # Response Time
        resp_times = [m['avgresponsetime'] for m in metrics if m['numdestroyed'] > 0]
        avg_resp_time = np.mean(resp_times) if resp_times else 0.0

        return {
            'avg_cpu_util': avg_cpu,
            'avg_ram_util': avg_ram,
            'avg_energy_kwhr': avg_energy,
            'avg_response_time': avg_resp_time,
        }

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    def report(self, metrics):
        header = '\033[1m\033[95m'
        bold = '\033[1m'
        endc = '\033[0m'

        print(f'\n{header}{"="*72}{endc}')
        print(f'{header}  FT-MoE Evaluation Report (Paper Metrics Summary){endc}')
        print(f'{header}{"="*72}{endc}')

        # Fault Detection
        if 'accuracy' in metrics:
            print(f'\n{bold}  Fault Detection:{endc}')
            print(f'    Accuracy : {metrics["accuracy"]:.4f}')
            print(f'    Precision: {metrics["precision"]:.4f}')
            print(f'    Recall   : {metrics["recall"]:.4f}')
            print(f'    F1 Score : {metrics["f1"]:.4f}')

        # Fault Classification
        if 'hr' in metrics:
            print(f'\n{bold}  Fault Classification:{endc}')
            print(f'    HR (HitRate): {metrics["hr"]:.4f}')
            print(f'    NDCG        : {metrics["ndcg"]:.4f}')

        # QoS
        print(f'\n{bold}  Quality of Service (QoS):{endc}')
        print(f'    Avg CPU Utilization   : {metrics.get("avg_cpu_util", 0):.2f} %')
        print(f'    Avg RAM Utilization   : {metrics.get("avg_ram_util", 0):.2f} %')
        print(f'    Avg Energy            : {metrics.get("avg_energy_kwhr", 0):.4f} KWhr')
        print(f'    Avg Response Time     : {metrics.get("avg_response_time", 0):.2f} s')

        print(f'\n{header}{"="*72}{endc}\n')

    def run(self, env_name='simulator'):
        metrics = self.evaluate(env_name)
        self.report(metrics)
        return metrics

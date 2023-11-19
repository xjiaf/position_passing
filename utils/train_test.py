import logging

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision
    )
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from utils.early_stopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, params, save_path=None, val_size=0.15, device=device):
        self.params = params
        self.val_size = val_size
        if save_path is None:
            self.save_path = (params['result_path'] / params[
                'dataset'] / params['model'])
        else:
            self.save_path = save_path

        # Load the graph
        logging.info(f"load graph from {params['dataset']}")
        dataset = JODIEDataset(params['data_path'], name=params['dataset'])
        self.data = dataset[0]
        self.data = self.data.to(device)

        # Initialize
        self.writer = SummaryWriter(log_dir=self.save_path / 'logs')
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.mean_loss = MeanMetric().to(device)
        self.metrics = MetricCollection(
            [BinaryAUROC(), BinaryF1Score(), BinaryPrecision()]).to(device)
        self.train_loader, self.val_loader, self.test_loader = \
            self.create_loader()

    def create_loader(self):
        logging.info('create loader')
        train_data, val_data, test_data = self.data.train_val_test_split(
            val_ratio=self.val_size, test_ratio=self.val_size)
        train_loader = TemporalDataLoader(
                            train_data,
                            batch_size=self.params['batch_size'],
                            neg_sampling_ratio=self.params[
                                'neg_sampling_ratio'],
                            )
        val_loader = TemporalDataLoader(
                            val_data,
                            batch_size=self.params['batch_size'],
                            neg_sampling_ratio=self.params[
                                'neg_sampling_ratio'],
                        )
        test_loader = TemporalDataLoader(
                            test_data,
                            batch_size=self.params['batch_size'],
                            neg_sampling_ratio=self.params[
                                'neg_sampling_ratio'],
                        )
        return train_loader, val_loader, test_loader

    def init_model(self):
        """Initialize the model"""
        logging.info(f"init model: {self.params['model']}")
        if self.params['model'] == 'tgnn':
            from models.tgnn import TGNN
            model = TGNN(self.data.num_nodes, self.data.msg.size(-1),
                         self.params['memory_dim'], self.params['time_dim'],
                         self.params['embedding_dim'],
                         dropout=self.params['dropout'],
                         size=self.params['neighbor_sample_size'])
        elif self.params['model'] == 'ptgnn':
            from models.ptgnn import PTGNN
            model = PTGNN(
                self.data.num_nodes, self.data.msg.size(-1),
                self.params['memory_dim'], self.params['time_dim'],
                self.params['pos_embedding_dim'],
                self.params['embedding_dim'],
                dropout=self.params['dropout'],
                step=self.params['step'],
                size=self.params['neighbor_sample_size'])
        return model

    def model_forward(self, batch):
        if self.params['model'] == 'tgnn':
            pos_out, neg_out = self.model(
                data=self.data, src=batch.src, dst=batch.dst,
                neg_dst=batch.neg_dst, n_id=batch.n_id, t=batch.t,
                msg=batch.msg)
        elif self.params['model'] == 'ptgnn':
            pos_out, neg_out = self.model(
                data=self.data, src=batch.src, dst=batch.dst,
                neg_dst=batch.neg_dst, n_id=batch.n_id, t=batch.t,
                msg=batch.msg)
        return pos_out, neg_out

    def train(self, epoch_num: int = 50, is_early_stopping: bool = True):
        # Initialize
        if is_early_stopping:
            early_stopping = EarlyStopping(patience=self.params[
                'patience'], verbose=True,
                path=self.save_path / 'checkpoint.pt')

        logging.info("---------start training----------")
        self.model = self.init_model().to(device)
        optimizer = optim.Adam([{'params': self.model.parameters()}],
                               lr=self.params['lr'],
                               weight_decay=self.params['weight_decay'])
        for self.epoch_idx in range(epoch_num):
            logging.info(
                "---------start epoch {0}----------".format(
                    self.epoch_idx + 1))
            self.model.train()  # set the model to train mode
            self.model.reset_state()  # reset the state of the model
            self.mean_loss.reset()
            self.metrics.reset()
            for batch_idx, batch in enumerate(self.train_loader):
                optimizer.zero_grad()
                # Move batch to GPU
                batch = batch.to(device)
                pos_out, neg_out = self.model_forward(batch)
                train_loss = self.criterion(pos_out, torch.ones_like(pos_out))
                train_loss += self.criterion(
                    neg_out, torch.zeros_like(neg_out))

                self.model.update_state(
                    batch.src, batch.dst, batch.t, batch.msg)

                train_loss.backward()
                nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=5, norm_type=2.0)
                optimizer.step()
                self.model.memory.detach()  # detach the memory
                # Calculate metrics
                y_pred, y_true = self.predict(pos_out, neg_out)
                self.metrics(y_pred, y_true)
                self.mean_loss(train_loss.item())

                if self.epoch_idx == 0:
                    if batch_idx % 20 == 0:
                        logging.info(
                            f'Batch {batch_idx} train'
                            f'loss: {train_loss.item()}')
                        logging.info(f'GPU usage: {cuda_usage()}')

            else:
                # Epoch finished and evaluate the model
                loss = self.mean_loss.compute()
                train_metrics = self.metrics.compute()
                self.log_metrics(loss, train_metrics, 'train')

                val_loss = self.test(mode='val')
                if is_early_stopping:
                    early_stopping(val_loss, self.model)

                    if early_stopping.early_stop:
                        logging.info("Early stopping")
                        break
        else:
            logging.info("---------all epochs finished----------")

        self.writer.close()
        logging.info("---------finish training----------")

    def predict(self, pos_out, neg_out):
        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().detach().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)
        return y_pred.squeeze(), y_true

    def test(self, mode: str = 'test'):
        # Initialize
        torch.manual_seed(2023)
        if mode == 'test':
            loader = self.test_loader
        elif mode == 'val':
            loader = self.val_loader
        else:
            raise ValueError('mode must be `test` or `val`')

        if not hasattr(self, 'model'):
            # Load the model if it is not loaded
            save_model_path = self.save_path / 'model.pt'
            if not save_model_path.exists():
                save_model_path = self.save_path / 'checkpoint.pt'
            else:
                raise ValueError('model not found')
            logging.info('load model from {0}'.format(save_model_path))
            self.model = self.init_model().to(device)
            self.model.load_state_dict(torch.load(save_model_path))

        logging.info(f"---------start {mode}ing----------")
        self.model.eval()  # set the model to eval mode
        self.mean_loss.reset()
        self.metrics.reset()
        with torch.no_grad():
            for _, batch in enumerate(loader):
                batch = batch.to(device)
                pos_out, neg_out = self.model_forward(batch)
                y_pred, y_true = self.predict(pos_out, neg_out)
                loss = self.criterion(pos_out, torch.ones_like(pos_out))
                loss += self.criterion(neg_out, torch.zeros_like(neg_out))
                self.model.update_state(batch.src, batch.dst,
                                        batch.t, batch.msg)
                self.mean_loss(loss.item())
                self.metrics(y_pred, y_true)
            else:
                loss = self.mean_loss.compute()
                metrics = self.metrics.compute()

        self.log_metrics(loss, metrics, mode)

        if mode == 'test':
            self.writer.close()
        else:
            return loss
        torch.cuda.empty_cache()
        logging.info("---------finish {0}ing----------".format(mode))

    def log_metrics(self, loss, metrics, mode):
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar('{0}/{1}'.format(
                metric_name, mode), metric_value.item(), self.epoch_idx + 1)
            logging.info('Epoch {0}, {1} {2}: {3}'.format(
                self.epoch_idx + 1, mode, metric_name, metric_value.item()))

        self.writer.add_scalar('Loss/{0}'.format(
            mode), loss.item(), self.epoch_idx + 1)
        logging.info('Epoch {0}, {1} loss:{2}'.format(
            self.epoch_idx + 1, mode, loss.item()))


def cuda_usage():
    # in MB
    u = (torch.cuda.mem_get_info()[1] -
         torch.cuda.mem_get_info()[0]) // (1024**2)
    return u

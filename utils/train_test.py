
import logging
from pathlib import Path
from functools import cached_property

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
    BinaryAUROC, BinaryF1Score, BinaryRecall)
from torch_geometric.sampler import NeighborSampler, NegativeSampling
from torch_geometric.loader import LinkLoader, LinkNeighborLoader
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from utils.early_stopping import EarlyStopping

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


# class Dataset:
#     def __init__(self, params):
#         """
#         准备用dataloader的train_mask, val_mask, test_mask来淘汰这个类
#         """
#         self.params = params
#         # Load graph

#         x_path = (params['processed_data_path'] / params[
#             'dataset'] / params['x_file'])
#         x = torch.load(x_path)
#         train_edge, test_edge = self.train_test_split()
#         self.train = TemporalGraph(x=x, edge=train_edge)
#         self.test = TemporalGraph(x=x, edge=test_edge)

#         # for induction

#     def train_test_split(self, test_size=0.2):
#         # 加载并排序数据
#         edge_path = Path(self.params['processed_data_path'],
#                          self.params['dataset'],
#                          self.params['edge_path'])
#         edge_index = torch.load(edge_path)
#         sorted_edge_index = edge_index[torch.argsort(edge_index[:, 2])]

#         # 划分训练和测试集
#         train_size = int((1-test_size) * sorted_edge_index.shape[0])
#         train_edges, test_edges = sorted_edge_index.split(
#             [train_size, len(sorted_edge_index) - train_size])

#         # 获取训练数据中的节点
#         train_nodes_set = set(train_edges[:, :2].flatten().tolist())

#         # 使用集合运算快速分类测试边
#         is_transductive = [src in train_nodes_set and dst in
#                            train_nodes_set for src, dst, _ in test_edges]
#         transductive_mask = torch.tensor(is_transductive)
#         inductive_mask = ~transductive_mask

#         transductive_edges = test_edges[transductive_mask]
#         inductive_edges = test_edges[inductive_mask]

#     def get_train_loader(self):
#         print("get dataloader:", self.params['loader'])
#         if self.params['model'] == 'dgnn':
#             if self.params['loader'] == 'LinkNeighborLoader':
#                 loader = LinkNeighborLoader(
#                     data=self.train,
#                     edge_label_index=self.train.edge_index,
#                     batch_size=self.params['batch_size'],
#                     num_neighbors=self.params[
#                         'num_neighbors'],
#                     neg_sampling=NegativeSampling(
#                         mode='triplet'),
#                     temporal_strategy='uniform',
#                     shuffle=self.params['shuffle'],
#                     neg_sampling_ratio=self.params[
#                         'neg_sampling_ratio'],
#                     num_workers=self.params['num_workers'],
#                     edge_label_time=self.train.edge_time,
#                     time_attr='edge_time')
#             elif self.params['loader'] == 'LinkLoader':
#                 loader = LinkLoader(
#                     data=self.train,
#                     shuffle=self.params['shuffle'],
#                     neg_sampling=NegativeSampling(
#                         mode='triplet'),
#                     batch_size=self.params['batch_size'],
#                     neg_sampling_ratio=self.params[
#                         'neg_sampling_ratio'],
#                     num_workers=self.params['num_workers'],
#                     edge_label_time=self.train.edge_time,
#                     time_attr='edge_time')
#         return loader

#     def get_test_loader(self):
#         pass


class Trainer:
    def __init__(self, params, save_path=None, val_size=0.15, device=device):
        self.params = params
        self.val_size = val_size
        if save_path is None:
            self.save_path = (params['result_path'] / params[
                'dataset'] / params['model'])
        else:
            self.save_path = save_path

        # Initialize
        self.writer = SummaryWriter(log_dir=self.save_path / 'logs')
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.mean_loss = MeanMetric().to(device)
        self.metrics = MetricCollection(
            [BinaryAUROC(), BinaryF1Score(), BinaryRecall()]).to(device)

    def create_loader(self):
        dataset = JODIEDataset(self.params['data_path'], name=self.params['dataset'])
        data = dataset[0]
        data = data.to(device)
        train_data, val_data, test_data = data.train_val_test_split(
            val_ratio=0.15, test_ratio=0.15)
        train_loader = TemporalDataLoader(
                            train_data,
                            batch_size=200,
                            neg_sampling_ratio=1.0,
                            )
        val_loader = TemporalDataLoader(
                            val_data,
                            batch_size=200,
                            neg_sampling_ratio=1.0,
                        )
        test_loader = TemporalDataLoader(
                            test_data,
                            batch_size=200,
                            neg_sampling_ratio=1.0,
                        )
        # neighbor_loader = LastNeighborLoader(
        #     data.num_nodes, size=10, device=device)
        return train_loader, val_loader, test_loader

    def init_model(self):
        """Initialize the model"""
        logging.info('init model: %s', self.params['model'])
        if self.params['model'] == 'dgdcn':
            from model.dgdcn import DGDCN
            self.sampler = NeighborSampler(
                data=self.graph,
                num_neighbors=self.params['num_neighbors'],
                temporal_strategy='uniform',
                time_attr='edge_time')
            model = DGDCN(
                self.params,
                graph_in_channels=self.graph.num_node_features,
                graph_out_channels=self.params['graph_emb_dim'],
                dense_dim=self.dataset.dense_feature_dim)
        elif self.params['model'] == 'dcnn':
            from model.dcnn import DCNN
            model = DCNN(
                self.params,
                dense_dim=self.dataset.dense_feature_dim,
                sparse_dim=self.dataset.sparse_feature_dim,
                sparse_emb_dim=self.params['sparse_emb_dim']
            )

        return model

    def model_forward(self, batch):
        if self.params['model'] == 'dgdcn':
            output = self.model(self.sampler,
                                self.graph,
                                batch['item_id'],
                                batch['item_time'],
                                batch['dense_features']
                                )
        elif self.params['model'] == 'dcnn':
            output = self.model(
                batch['dense_features'])  # , batch['sparse_features']
        return output

    def train(self, epoch_num=None, is_early_stopping=True, is_val=True):
        # Initialize
        if epoch_num is None:
            epoch_num = self.params['epoch_num']
        if is_early_stopping:
            early_stopping = EarlyStopping(patience=self.params[
                'patience'], verbose=True,
                path=self.save_path / 'checkpoint.pt')

        logging.info("---------start training----------")
        loader = self.train_loader
        self.model = self.init_model().to(device)
        optimizer = optim.Adam([{'params': self.model.parameters()}],
                               lr=self.params['lr'],
                               weight_decay=self.params['weight_decay'])
        for self.epoch_idx in range(epoch_num):
            logging.info(
                "---------start epoch {0}----------".format(
                    self.epoch_idx + 1))
            self.model.train()  # set the model to train mode
            self.mean_loss.reset()
            self.metrics.reset()
            for batch_idx, batch in enumerate(loader):
                # Move batch to GPU
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model_forward(batch)
                optimizer.zero_grad()
                self.metrics(
                    output.cpu().detach(), batch['label'].cpu().detach())
                train_loss = self.criterion(output, batch['label'])
                self.mean_loss(train_loss.item())
                train_loss.backward()
                nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=5, norm_type=2.0)
                optimizer.step()

                if self.epoch_idx == 0:
                    if batch_idx % 500 == 0:
                        logging.info(
                            f'Batch {batch_idx} train'
                            f'loss: {train_loss.item()}')
                del output, train_loss, batch
            else:
                # Epoch finished and evaluate the model
                train_loss = self.mean_loss.compute()
                train_metrics = self.metrics.compute()
                self.log_metrics(train_loss, train_metrics, 'train')
                if is_val:
                    val_loss = self.test(mode='val')
                if is_early_stopping:
                    if is_val:
                        early_stopping(val_loss, self.model)
                    else:
                        early_stopping(train_loss, self.model)
                    if early_stopping.early_stop:
                        logging.info("Early stopping")
                        break
            torch.cuda.empty_cache()
        else:
            logging.info("---------all epochs finished----------")

        self.writer.close()
        logging.info("---------finish training----------")

    def test(self, mode: str = 'test'):
        if mode == 'test':
            loader = self.test_loader
            self.epoch_num = -1
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
            logging.info('load model from {0}'.format(self.save_path))
            self.model = self.init_model().to(device)
            self.model.load_state_dict(torch.load(save_model_path))

        logging.info("---------start {0}ing----------".format(mode))
        self.model.eval()  # set the model to eval mode
        self.mean_loss.reset()
        self.metrics.reset()
        with torch.no_grad():
            for _, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model_forward(batch)
                loss = self.criterion(output, batch['label'])
                self.mean_loss(loss)
                self.metrics(output, batch['label'])  # record the metrics
            else:
                loss = self.mean_loss.compute()
                metrics = self.metrics.compute()

        self.log_metrics(loss, metrics, mode)

        if mode == 'test':
            self.writer.close()
        else:
            return loss

        del output, loss, batch
        torch.cuda.empty_cache()
        logging.info("---------finish {0}ing----------".format(mode))

    def log_metrics(self, loss, metrics, mode):
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar('{0}/{1}'.format(
                metric_name, mode), metric_value.item(), self.epoch_idx + 1)
            logging.info('Epoch {0}, {1} {2}:'.format(
                self.epoch_idx + 1, mode, metric_name), metric_value.item())

        self.writer.add_scalar('Loss/{0}'.format(
            mode), loss.item(), self.epoch_idx + 1)
        logging.info('Epoch {0}, {1} loss:'.format(
            self.epoch_idx + 1, mode), loss.item())

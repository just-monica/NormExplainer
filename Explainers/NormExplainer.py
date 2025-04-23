import logging
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
#from torch_geometric.utils import get_embeddings


class NodeNormExplainer(ExplainerAlgorithm):
    r"""
    adapted from PGExplainer
    pred node mask first, then use the node mask to predict edge mask (multiply or add)
    """
    def __init__(self, epochs: int, lr: float = 0.003, model: torch.nn.Module = None, config = None, device = None):
        super().__init__()
        self.coeffs = {
            'prediction': 1.0,
            'norm': 1.0,
            'node_reduction': 'mean',
            'node_size': 0.05, #
            'node_ent': 1.0, #
            'temp': [5.0, 2.0],
            'bias': 0.01,
            'min_size': 0.00,
        }
        self.model = model
        self.device = device
        self.epochs = epochs #number of epochs to train
        self.lr = lr
        self.config = config
        self.coeffs.update(self.config['coeffs'])
        print(f'coeffs: {self.coeffs}')
        self.loss_dict = {'prediction_loss': 0.0, 'node_size_loss': 0.0, 'node_entropy_loss': 0.0, 'norm_loss': 0.0}
        self.node_mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.node_mlp = self.node_mlp.to(self.device)
        '''
        edge_mask_type = self.explainer_config.edge_mask_type
        node_mask_type = self.explainer_config.node_mask_type
        '''
        edge_mask_type = self.config.get('edge_mask_type', None) #must set this in pyg framework
        node_mask_type = self.config.get('node_mask_type', None)
        self.use_edge_mask = self.config.get('use_edge_mask', True) #we may calculate the edge mask but not use it, set it to True to use it
        self.calculation_type = self.config.get('calculation_type', 'add') #multiply or add
        self.use_concrete = self.config.get('use_concrete', True) #use binary concrete to process the edge mask
        self.log_norm = self.config.get('log_norm', False) #use log to normalize the norm
        self.params = []
        if node_mask_type is not None:
            self.params.append({'params': self.node_mlp.parameters(), 'lr': self.config.get('node_lr', self.lr)})
        print(f'params: {self.params}')
        self.optimizer = torch.optim.Adam(self.params)
        self._curr_epoch = -1 # current epoch of training

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.node_mlp)
        self._curr_epoch = -1

    def train(
        self,
        epoch: int,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        *,
        edge_attr: Tensor,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features of a
                homogeneous graph.
            edge_index (torch.Tensor): The input edge indices of a homogeneous
                graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")
        self._curr_epoch += 1
        clear_masks(self.model)
        self.optimizer.zero_grad()
        #z = get_embeddings(model, x, edge_index, **kwargs)[-1]
        temperature = self._get_temperature(epoch)
        z = self.model.get_node_embedding(x=x, edge_index=edge_index, edge_attr = edge_attr)#get node embedding does not need batch
        node_logits = self.node_mlp(z).view(-1,1)
        #node_mask = self._concrete_sample(node_logits, temperature) # still need to use sigmoid to be projected to [0, 1]
        edge_mask, node_mask = self._calculate_mask(node_logits, edge_index, temp = temperature)

        if self.use_edge_mask:
            set_masks(self.model, edge_mask, edge_index, apply_sigmoid=False)

        h = x if self.explainer_config.node_mask_type is None else x * node_mask.sigmoid()
        y_hat, y = self.model(h, edge_index,batch = batch, edge_attr = edge_attr, **kwargs), target
        graph_embedding = self.model.get_graph_embedding(h, edge_index, batch = batch, edge_attr = edge_attr)
        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss, loss_dict = self._loss(y_hat, y, edge_mask, node_mask)
        norm_p = self.config.get('norm_p') if self.config.get('norm_p') is not None else float('inf')
        if self.log_norm is True:
            norm_loss = self.coeffs['norm'] * (torch.log(torch.norm(graph_embedding, p= self.config.get('norm_p', float('inf')),  dim=1).mean()))
        else:
            norm_loss  = self.coeffs['norm'] * (torch.norm(graph_embedding, p= self.config.get('norm_p', float('inf')),  dim=1).mean())
        loss = loss - norm_loss #we want to maximize the norm of the graph embedding
        loss_dict['norm_loss'] = norm_loss.item()
        self.loss_dict.update(loss_dict)
        loss.backward()
        self.optimizer.step()

        clear_masks(self.model)

        return float(loss), self.loss_dict

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_attr: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")
        '''# dont use safety check
        if self._curr_epoch < self.epochs - 1:  # Safety check:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
                             f"trained (got {self._curr_epoch + 1} epochs "
                             f"from {self.epochs} epochs). Please first train "
                             f"the underlying explainer model by running "
                             f"`explainer.algorithm.train(...)`.")
        '''
        '''dont use hard mask
        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))
        '''
        
        clear_masks(self.model)
        #z =  self.model.get_node_embedding(x, edge_index, edge_weight = None, edge_attr = edge_attr)
        z = self.model.get_node_embedding(x=x, edge_index=edge_index, edge_attr = edge_attr)
        node_logits = self.node_mlp(z).view(-1,1)

        node_mask = self._post_process_mask(node_logits, hard_mask = None,
                                            apply_sigmoid=True)#project node_logits to [0, 1]
        edge_mask, _ = self._calculate_mask(node_logits, edge_index, temp = -1.0)
        return Explanation(edge_mask=edge_mask, node_mask=node_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False
        '''
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False
        '''

        return True


    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        # transfer node embedding to edge embedding by concatenating
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        '''
        should use the sigmoid of the return value as the mask
        '''
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor, node_mask: Tensor) -> Tensor:
        loss_dict = {'prediction_loss': 0.0, 'edge_size_loss': 0.0, 'edge_entropy_loss': 0.0, 'node_size_loss': 0.0, 'node_entropy_loss': 0.0, 'norm_loss': 0.0}
        #prediction loss
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y) * self.coeffs['pred']
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y) * self.coeffs['pred']
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y) * self.coeffs['pred']
        loss_dict['prediction_loss'] = loss.item()
        # Regularization loss:
        '''
        if edge_mask is not None:
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            edge_mask_s = edge_mask.sigmoid()
            edge_size_loss = edge_reduce(edge_mask_s) * self.coeffs['edge_size']
            edge_mask_s = 0.99 * edge_mask_s + 0.005
            edge_mask_ent = - edge_mask_s * edge_mask_s.log() - (1 - edge_mask_s) * ((1 - edge_mask_s).log())
            edge_mask_ent_loss = edge_mask_ent.mean() * self.coeffs['edge_ent']
        else:
            edge_mask_ent_loss = 0.0
            edge_size_loss = 0.0
        loss_dict['edge_size_loss'] = edge_size_loss.item()
        loss_dict['edge_entropy_loss'] = edge_mask_ent_loss.item()
        '''
        if node_mask is not None:
            node_reduce = getattr(torch, self.coeffs['node_reduction'])
            node_mask_s = node_mask.sigmoid()
            node_size_loss = node_reduce(torch.abs(node_mask_s-self.coeffs['min_size'])) * self.coeffs['node_size']
            node_mask_s = 0.99 * node_mask_s + 0.005
            node_mask_ent = - node_mask_s * node_mask_s.log() - (1 - node_mask_s) * ((1 - node_mask_s).log())
            node_mask_ent_loss = node_mask_ent.mean() * self.coeffs['node_ent']
        else:
            node_mask_ent_loss = 0.0
        loss_dict['node_size_loss'] = node_size_loss.item()
        loss_dict['node_entropy_loss'] = node_mask_ent_loss.item()

        return loss +  node_size_loss + node_mask_ent_loss, loss_dict

    def _calculate_mask(self, node_logits: Tensor, edge_index: Tensor, temp):
        # node_logits is in shape [N, 1]
        if temp > 0:
            node_mask = self._concrete_sample(node_logits, temp) # still need to use sigmoid to be projected to [0, 1]
        else:
            node_mask = node_logits
        if self.calculation_type == 'add':
            src = node_logits[edge_index[0]]
            dst = node_logits[edge_index[1]]
            if self.use_concrete and temp > 0: #use temp > 0 to indicate that we need to use concrete (training)
                src_logits = self._concrete_sample(src, temp)
                dst_logits = self._concrete_sample(dst, temp)
                edge_mask = (src_logits.sigmoid() + dst_logits.sigmoid())/2.0
            else:
                edge_mask = (src.sigmoid() + dst.sigmoid())/2.0
            return edge_mask.view(-1), node_mask  #edge_mask is in [0,1], but node_mask still in R 
        elif self.calculation_type == 'multiply':
            src = node_logits[edge_index[0]]
            dst = node_logits[edge_index[1]]
            if self.use_concrete and temp > 0: #use temp > 0 to indicate that we need to use concrete (training)
                src = self._concrete_sample(src, temp)
                dst = self._concrete_sample(dst, temp)
            src = src.sigmoid()
            dst = dst.sigmoid()
            edge_mask = src * dst
            return edge_mask.view(-1), node_mask
        else:
            raise ValueError(f'calculation type {self.calculation_type} not supported')


class EdgeNormExplainer(ExplainerAlgorithm):
    r"""
    adapted from PGExplainer
    directly predict edge mask, no node mask
    """
    def __init__(self, epochs: int, lr: float = 0.003, model: torch.nn.Module = None, config = None, device = None):
        super().__init__()
        self.coeffs = {
            'prediction': 1.0,
            'norm': 1.0,
            'edge_reduction': 'mean',
            'edge_size': 0.05, #
            'edge_ent': 1.0, #
            'temp': [5.0, 1.0],
            'bias': 0.01,
            'min_size': 0.70,
        }
        self.model = model
        self.device = device
        self.epochs = epochs #number of epochs to train
        self.lr = lr
        self.config = config
        self.coeffs.update(self.config['coeffs'])
        print(f'coeffs: {self.coeffs}')
        self.loss_dict = {'prediction_loss': 0.0, 'edge_size_loss': 0.0, 'edge_entropy_loss': 0.0, 'norm_loss': 0.0}
        self.edge_mlp = Sequential(
            Linear(-1, 128),
            ReLU(),
            Linear(128, 1)
        )
        self.edge_mlp = self.edge_mlp.to(self.device)
        '''
        edge_mask_type = self.explainer_config.edge_mask_type
        node_mask_type = self.explainer_config.node_mask_type
        '''
        edge_mask_type = self.config.get('edge_mask_type', None) #must set this in pyg framework
        node_mask_type = self.config.get('node_mask_type', None)
        self.use_concrete = self.config.get('use_concrete', True) #use binary concrete to process the edge mask
        self.log_norm = self.config.get('log_norm', False) #use log to normalize the norm
        self.params = []
        if edge_mask_type is not None:
            self.params.append({'params': self.edge_mlp.parameters(), 'lr': self.config.get('edge_lr', self.lr), 'weight_decay': self.config.get('edge_wd', 0.0)})
        print(f'params: {self.params}')
        self.optimizer = torch.optim.Adam(self.params, lr = self.lr)
        self._curr_epoch = -1 # current epoch of training

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.node_mlp)
        self._curr_epoch = -1

    def train(
        self,
        epoch: int,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        *,
        edge_attr: Tensor,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features of a
                homogeneous graph.
            edge_index (torch.Tensor): The input edge indices of a homogeneous
                graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")
        self._curr_epoch += 1
        clear_masks(self.model)
        self.optimizer.zero_grad()
        #z = get_embeddings(model, x, edge_index, **kwargs)[-1]
        temperature = self._get_temperature(epoch)
        z = self.model.get_node_embedding(x=x, edge_index=edge_index, edge_attr = edge_attr)#get node embedding does not need batch
        edge_embedding = self._get_inputs(z, edge_index, index = None)
        edge_logits = self.edge_mlp(edge_embedding).view(-1)
        #node_mask = self._concrete_sample(node_logits, temperature) # still need to use sigmoid to be projected to [0, 1]
        edge_mask = self._calculate_mask(edge_logits, temp = temperature)


        set_masks(self.model, edge_mask, edge_index, apply_sigmoid=True)

        h = x 
        y_hat, y = self.model(h, edge_index,batch = batch, edge_attr = edge_attr, **kwargs), target
        graph_embedding = self.model.get_graph_embedding(h, edge_index, batch = batch, edge_attr = edge_attr)
        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss, loss_dict = self._loss(y_hat, y, edge_mask)
        if self.log_norm is True:
            norm_loss = self.coeffs['norm'] * (torch.log(torch.norm(graph_embedding, p= self.config.get('norm_p', float('inf')),  dim=1).mean()))
        else:
            norm_loss  = self.coeffs['norm'] * (torch.norm(graph_embedding, p= self.config.get('norm_p', float('inf')),  dim=1).mean())
        loss = loss - norm_loss #we want to maximize the norm of the graph embedding
        loss_dict['norm_loss'] = norm_loss.item()
        self.loss_dict.update(loss_dict)
        loss.backward()
        self.optimizer.step()

        clear_masks(self.model)

        return float(loss), self.loss_dict

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        edge_attr: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        
        clear_masks(self.model)
        #z =  self.model.get_node_embedding(x, edge_index, edge_weight = None, edge_attr = edge_attr)
        z = self.model.get_node_embedding(x=x, edge_index=edge_index, edge_attr = edge_attr)

        edge_embedding = self._get_inputs(z, edge_index, index = None)
        edge_logits = self.edge_mlp(edge_embedding).view(-1)
        edge_mask = self._calculate_mask(edge_logits, temp = -1.0)
        node_mask = torch.rand((x.size(0),)).to(self.device)#set a random node mask to avoid downstream error
        return Explanation(edge_mask=edge_mask, node_mask=node_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False
        '''
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False
        '''

        return True


    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        # transfer node embedding to edge embedding by concatenating
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        '''
        should use the sigmoid of the return value as the mask
        '''
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        loss_dict = {'prediction_loss': 0.0, 'edge_size_loss': 0.0, 'edge_entropy_loss': 0.0, 'norm_loss': 0.0}
        #prediction loss
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y) * self.coeffs['pred']
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y) * self.coeffs['pred']
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y) * self.coeffs['pred']
        loss_dict['prediction_loss'] = loss.item()
        # Regularization loss:
        
        if edge_mask is not None:
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            edge_mask_s = edge_mask.sigmoid()
            edge_mask_s = 0.99 * edge_mask_s + 0.005
            edge_size_loss = edge_reduce(torch.abs(edge_mask_s-self.coeffs['min_size'])) * self.coeffs['edge_size']
            edge_mask_ent = - edge_mask_s * edge_mask_s.log() - (1 - edge_mask_s) * ((1 - edge_mask_s).log())#is entropy loss necessary?
            edge_mask_ent_loss = edge_mask_ent.mean() * self.coeffs['edge_ent']
        else:
            edge_mask_ent_loss = 0.0
            edge_size_loss = 0.0
        loss_dict['edge_size_loss'] = edge_size_loss.item()
        loss_dict['edge_entropy_loss'] = edge_mask_ent_loss.item()


        return loss +  edge_size_loss + edge_mask_ent_loss, loss_dict

    def _calculate_mask(self, edge_logits: Tensor, temp):
        if temp > 0:
            edge_mask = self._concrete_sample(edge_logits, temp) # still need to use sigmoid to be projected to [0, 1]
        else:
            edge_mask = edge_logits
        return edge_mask.view(-1)
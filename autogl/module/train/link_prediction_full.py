from . import register_trainer, Evaluation
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import torch.nn.functional as F
from ..model import BaseAutoEncoderMaintainer, BaseAutoDecoderMaintainer, BaseAutoModel
from .evaluation import Auc, EVALUATE_DICT
from .base import EarlyStopping, BaseLinkPredictionTrainer
from typing import Union, Tuple
from copy import deepcopy
from torch_geometric.utils import negative_sampling
# from ...datasets.utils import negative_sampling
from ...utils import get_logger

from ...backend import DependentBackend

LOGGER = get_logger("link prediction trainer")


def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, list):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")

class _DummyLinkModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        if isinstance(encoder, BaseAutoModel):
            self.encoder = encoder.model
            self.decoder = None
        else:
            self.encoder = encoder.encoder
            self.decoder = None if decoder is None else decoder.decoder
    
    def encode(self, data):
        if isinstance(self.encoder, BaseAutoModel):
            return self.encoder.lp_encode(data)
        return self.encoder(data)
    
    def decode(self, features, data, pos_edges, neg_edges):
        if isinstance(self.encoder, BaseAutoModel) or self.decoder is None:
            return features
        return self.decoder(features, data, pos_edges, neg_edges)

@register_trainer("LinkPredictionFull")
class LinkPredictionTrainer(BaseLinkPredictionTrainer):
    """
    The link prediction trainer.

    Used to automatically train the link prediction problem.

    Parameters
    ----------
    model: ``BaseModel`` or ``str``
        The (name of) model used to train and predict.

    optimizer: ``Optimizer`` of ``str``
        The (name of) optimizer used to train and predict.

    lr: ``float``
        The learning rate of link prediction task.

    max_epoch: ``int``
        The max number of epochs in training.

    early_stopping_round: ``int``
        The round of early stop.

    device: ``torch.device`` or ``str``
        The device where model will be running on.

    init: ``bool``
        If True(False), the model will (not) be initialized.
    """

    space = None

    def __init__(
        self,
        model: Union[Tuple[BaseAutoEncoderMaintainer, BaseAutoDecoderMaintainer], BaseAutoEncoderMaintainer, BaseAutoModel, str] = None,
        num_features=None,
        optimizer=torch.optim.Adam,
        lr=1e-4,
        max_epoch=100,
        early_stopping_round=101,
        weight_decay=1e-4,
        device="auto",
        init=True,
        feval=[Auc],
        loss="binary_cross_entropy_with_logits",
        lr_scheduler_type=None,
        *args,
        **kwargs,
    ):
        if isinstance(model, Tuple):
            encoder, decoder = model
        elif isinstance(model, BaseAutoModel):
            encoder, decoder = model, None
        else:
            encoder, decoder = model, "lp-decoder"
        super().__init__(encoder, decoder, num_features, "auto", device, feval, loss)

        self.opt_received = optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam": self.optimizer = torch.optim.Adam
            elif optimizer.lower() == "sgd": self.optimizer = torch.optim.SGD
            else: raise ValueError("Currently not support optimizer {}".format(optimizer))
        elif isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError("Currently not support optimizer {}".format(optimizer))

        self.lr_scheduler_type = lr_scheduler_type

        self.lr = lr
        self.max_epoch = max_epoch
        self.early_stopping_round = early_stopping_round
        self.args = args
        self.kwargs = kwargs
        self.weight_decay = weight_decay

        self.early_stopping = EarlyStopping(
            patience=early_stopping_round, verbose=False
        )

        self.valid_result = None
        self.valid_result_prob = None
        self.valid_score = None

        self.pyg_dgl = DependentBackend.get_backend_name()

        self.hyper_parameter_space = [
            {
                "parameterName": "max_epoch",
                "type": "INTEGER",
                "maxValue": 500,
                "minValue": 10,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "early_stopping_round",
                "type": "INTEGER",
                "maxValue": 30,
                "minValue": 10,
                "scalingType": "LINEAR",
            },
            {
                "parameterName": "lr",
                "type": "DOUBLE",
                "maxValue": 1e-1,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
            {
                "parameterName": "weight_decay",
                "type": "DOUBLE",
                "maxValue": 1e-2,
                "minValue": 1e-4,
                "scalingType": "LOG",
            },
        ]

        self.hyper_parameters = {
            "max_epoch": self.max_epoch,
            "early_stopping_round": self.early_stopping_round,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

        if init is True:
            self.initialize()

    def _initialize(self):
        #  Initialize the auto model in trainer.
        self.encoder.initialize()
        if self.decoder is not None:
            self.decoder.initialize()

    def _compose_model(self):
        return _DummyLinkModel(self.encoder, self.decoder)

    @classmethod
    def get_task_name(cls):
        return "LinkPrediction"

    def _train_only_pyg(self, data, train_mask=None):

        model = self._compose_model()
        data = data.to(self.device)
        optimizer = self.optimizer(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(1, self.max_epoch):
            model.train()

            try:
                neg_edge_index = data.train_neg_edge_index
            except:
                neg_edge_index = negative_sampling(
                    edge_index=data.train_pos_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=data.train_pos_edge_index.size(1),
                )

            optimizer.zero_grad()
            link_logits = model.encode(data)
            link_logits = model.decode(link_logits, data, data.train_pos_edge_index, neg_edge_index)
            link_labels = self.get_link_labels(
                data.train_pos_edge_index, neg_edge_index
            )
            
            if hasattr(F, self.loss):
                loss = getattr(F, self.loss)(link_logits, link_labels)
            else:
                raise TypeError(
                    "PyTorch does not support loss type {}".format(self.loss)
                )

            loss.backward()
            optimizer.step()
            scheduler.step()

            if type(self.feval) is list:
                feval = self.feval[0]
            else:
                feval = self.feval
            val_loss = self.evaluate([data], mask="val", feval=feval)
            if feval.is_higher_better() is True:
                val_loss = -val_loss
            self.early_stopping(val_loss, model)
            if self.early_stopping.early_stop:
                LOGGER.debug("Early stopping at %d", epoch)
                break
        self.early_stopping.load_checkpoint(model)

    def _train_only_dgl(self, dataset):
        model = self._compose_model()
        train_graph = dataset['train'].to(self.device)
        train_pos_graph = dataset['train_pos'].to(self.device)
        train_neg_data = dataset['train_neg'].to(self.device)

        optimizer = self.optimizer(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        lr_scheduler_type = self.lr_scheduler_type
        if type(lr_scheduler_type) == str and lr_scheduler_type == "steplr":
            scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == "multisteplr":
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif type(lr_scheduler_type) == str and lr_scheduler_type == "exponentiallr":
            scheduler = ExponentialLR(optimizer, gamma=0.1)
        elif (
            type(lr_scheduler_type) == str and lr_scheduler_type == "reducelronplateau"
        ):
            scheduler = ReduceLROnPlateau(optimizer, "min")
        else:
            scheduler = None

        for epoch in range(1, self.max_epoch):
            model.train()

            optimizer.zero_grad()

            pos_edges, neg_edges = torch.stack(train_pos_graph.edges()), torch.stack(train_neg_data.edges())

            link_logits = model.encode(train_graph)
            link_logits = model.decode(link_logits, train_graph, pos_edges, neg_edges)
            link_labels = self.get_link_labels(pos_edges, neg_edges)
            if hasattr(F, self.loss):
                loss = getattr(F, self.loss)(link_logits, link_labels)
            else:
                raise TypeError(
                    "PyTorch does not support loss type {}".format(self.loss)
                )

            loss.backward()
            optimizer.step()
            if self.lr_scheduler_type:
                scheduler.step()

            if type(self.feval) is list:
                feval = self.feval[0]
            else:
                feval = self.feval
            val_loss = self._evaluate_dgl(dataset, mask="val", feval=feval)
            if feval.is_higher_better() is True:
                val_loss = -val_loss
            self.early_stopping(val_loss, model)
            if self.early_stopping.early_stop:
                LOGGER.debug("Early stopping at %d", epoch)
                break

        self.early_stopping.load_checkpoint(model)

    def _predict_only_pyg(self, data):
        data = data.to(self.device)
        model = self._compose_model()
        model.eval()
        with torch.no_grad():
            res = model.encode(data)
        return res

    def _predict_only_dgl(self, dataset):
        pos_data = dataset['train']
        model = self._compose_model()
        model.eval()
        with torch.no_grad():
            z = model.encode(pos_data)
        return z

    def train(self, dataset, keep_valid_result=True):
        """
        train on the given dataset

        Parameters
        ----------
        dataset: The link prediction dataset used to be trained.

        keep_valid_result: ``bool``
            If True(False), save the validation result after training.

        Returns
        -------
        self: ``autogl.train.LinkPredictionTrainer``
            A reference of current trainer.

        """
        if self.pyg_dgl == 'pyg':
            data = dataset[0]
            data.edge_index = data.train_pos_edge_index
            self._train_only_pyg(data)
            if keep_valid_result:
                self.valid_result = self._predict_only_pyg(data)
                self.valid_result_prob = self._predict_proba_pyg(dataset, "val")
                self.valid_score = self._evaluate_pyg(dataset, mask="val", feval=self.feval)
        elif self.pyg_dgl == 'dgl':
            self._train_only_dgl(dataset)
            if keep_valid_result:
                self.valid_result = self._predict_only_dgl(dataset)
                self.valid_result_prob = self._predict_proba_dgl(dataset, "val")
                self.valid_score = self._evaluate_dgl(dataset, mask="val", feval=self.feval)

    def predict(self, dataset, mask=None):
        """
        The function of predicting on the given dataset.

        Parameters
        ----------
        dataset: The link prediction dataset used to be predicted.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.
        
        .. Note:: Deprecated, this function will be removed in the future.

        Returns
        -------
        The prediction result of ``predict_proba``.
        """
        if self.pyg_dgl == 'pyg':
            return self._predict_proba_pyg(dataset, mask=mask, in_log_format=False)
        elif self.pyg_dgl == 'dgl':
            return self._predict_proba_dgl(dataset, mask=mask, in_log_format=False)

    def predict_proba(self, dataset, mask=None, in_log_format=False):
        if self.pyg_dgl == 'pyg':
            return self._predict_proba_pyg(dataset, mask, in_log_format)
        elif self.pyg_dgl == 'dgl':
            return self._predict_proba_dgl(dataset, mask, in_log_format)

    def _predict_proba_pyg(self, dataset, mask=None, in_log_format=False):
        data = dataset[0]
        data.edge_index = data.train_pos_edge_index
        data = data.to(self.device)
        try:
            if mask in ["train", "val", "test"]:
                pos_edge_index = data[f"{mask}_pos_edge_index"]
                neg_edge_index = data[f"{mask}_neg_edge_index"]
            else:
                pos_edge_index = data[f"test_pos_edge_index"]
                neg_edge_index = data[f"test_neg_edge_index"]
        except:
            pos_edge_index = data[f"test_edge_index"]
            neg_edge_index = torch.zeros(2, 0).to(self.device)

        model = self._compose_model()
        model.eval()
        with torch.no_grad():
            z = self._predict_only_pyg(data)
            link_logits = model.decode(z, data, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()

        return link_probs

    def _predict_proba_dgl(self, dataset, mask=None, in_log_format=False):
        train_graph = dataset['train']
        try:
            try:
                pos_graph = dataset[f'{mask}_pos']
                neg_graph = dataset[f'{mask}_neg']
            except:
                pos_graph = dataset[f'test_pos']
                neg_graph = dataset[f'test_neg']
        except:
            import dgl
            pos_graph = dataset[mask]
            neg_graph = dgl.graph([], num_nodes=0).to(self.device)

        model = self._compose_model()
        model.eval()
        with torch.no_grad():
            z = model.encode(train_graph)
            link_logits = model.decode(
                z, 
                train_graph,
                torch.stack(pos_graph.edges()), 
                torch.stack(neg_graph.edges())
            )
            link_probs = link_logits.sigmoid()

        return link_probs

    def get_valid_predict(self):
        return self.valid_result

    def get_valid_predict_proba(self):
        return self.valid_result_prob

    def get_valid_score(self, return_major=True):
        """
        The function of getting the valid score.

        Parameters
        ----------
        return_major: ``bool``.
            If True, the return only consists of the major result.
            If False, the return consists of the all results.

        Returns
        -------
        result: The valid score in training stage.
        """
        if isinstance(self.feval, list):
            if return_major:
                return self.valid_score[0], self.feval[0].is_higher_better()
            else:
                return self.valid_score, [f.is_higher_better() for f in self.feval]
        else:
            return self.valid_score, self.feval.is_higher_better()

    def get_name_with_hp(self):
        name = "-".join(
            [
                str(self.optimizer),
                str(self.lr),
                str(self.max_epoch),
                str(self.early_stopping_round),
                str(self.model),
                str(self.device),
            ]
        )
        name = (
            name
            + "|"
            + "-".join(
                [
                    str(x[0]) + "-" + str(x[1])
                    for x in self.model.get_hyper_parameter().items()
                ]
            )
        )
        return name

    def evaluate(self, dataset, mask=None, feval=None):
        """
        The function of training on the given dataset and keeping valid result.

        Parameters
        ----------
        dataset: The link prediction dataset used to be evaluated.

        mask: ``train``, ``val``, or ``test``.
            The dataset mask.

        feval: ``str``.
            The evaluation method used in this function.

        Returns
        -------
        res: The evaluation result on the given dataset.

        """
        if self.pyg_dgl == 'pyg':
            return self._evaluate_pyg(dataset, mask, feval)
        elif self.pyg_dgl == 'dgl':
            return self._evaluate_dgl(dataset,mask,feval)

    def _evaluate_pyg(self, dataset, mask=None, feval=None):
        data = dataset[0]
        data = data.to(self.device)
        if feval is None:
            feval = self.feval
        else:
            feval = get_feval(feval)

        if mask in ["train", "val", "test"]:
            pos_edge_index = data[f"{mask}_pos_edge_index"]
            neg_edge_index = data[f"{mask}_neg_edge_index"]
        else:
            pos_edge_index = data[f"test_pos_edge_index"]
            neg_edge_index = data[f"test_neg_edge_index"]

        model = self._compose_model()
        model.eval()
        with torch.no_grad():
            link_probs = self._predict_proba_pyg(dataset, mask)
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)

        if not isinstance(feval, list):
            feval = [feval]
            return_signle = True
        else:
            return_signle = False

        res = []
        for f in feval:
            res.append(f.evaluate(link_probs.cpu().numpy(), link_labels.cpu().numpy()))
        if return_signle:
            return res[0]
        return res


    def _evaluate_dgl(self, dataset, mask=None, feval=None):
        if feval is None:
            feval = self.feval
        else:
            feval = get_feval(feval)

        train_graph = dataset['train']
        try:
            pos_graph = dataset[f'{mask}_pos']
            neg_graph = dataset[f'{mask}_neg']
        except:
            pos_graph = dataset[f'test_pos']
            neg_graph = dataset[f'test_neg']

        model = self._compose_model()
        model.eval()
        with torch.no_grad():
            z = model.encode(train_graph)
            link_logits = model.decode(
                    z,
                    train_graph,
                    torch.stack(pos_graph.edges()),
                    torch.stack(neg_graph.edges())
                )
            link_probs = link_logits.sigmoid()
            link_labels = self.get_link_labels(
                torch.stack(pos_graph.edges()), torch.stack(neg_graph.edges())
            )
        if not isinstance(feval, list):
            feval = [feval]
            return_signle = True
        else:
            return_signle = False

        res = []
        for f in feval:
            res.append(f.evaluate(link_probs.cpu().numpy(), link_labels.cpu().numpy()))
        if return_signle:
            return res[0]
        return res

    def to(self, new_device):
        self.device = new_device
        if self.encoder is not None: self.encoder.to_device(self.device)
        if self.decoder is not None: self.decoder.to_device(self.device)


    def duplicate_from_hyper_parameter(self, hp: dict, model=None, restricted=True):
        """
        The function of duplicating a new instance from the given hyperparameter.

        Parameters
        ----------
        hp: ``dict``.
            The hyperparameter used in the new instance.

        model: The model used in the new instance of trainer.

        restricted: ``bool``.
            If False(True), the hyperparameter should (not) be updated from origin hyperparameter.

        Returns
        -------
        self: ``autogl.train.LinkPredictionTrainer``
            A new instance of trainer.

        """
        if isinstance(model, Tuple):
            encoder, decoder = model
        elif isinstance(model, BaseAutoModel):
            encoder, decoder = model, None
        elif isinstance(model, BaseAutoEncoderMaintainer):
            encoder, decoder = model, self.decoder
        elif model is None:
            encoder, decoder = self.encoder, self.decoder
        else:
            raise TypeError("Cannot parse model with type", type(model))
        
        trainer_hp = hp.get("trainer", {})
        encoder_hp = hp.get("encoder", {})
        decoder_hp = hp.get("decoder", {})

        if not restricted:
            origin_hp = deepcopy(self.hyper_parameters)
            origin_hp.update(trainer_hp)
            trainer_hp = origin_hp
        
        encoder = encoder.from_hyper_parameter(encoder_hp)
        decoder = decoder.from_hyper_parameter_and_encoder(decoder_hp, encoder)
        
        ret = self.__class__(
            model=(encoder, decoder),
            num_features=self.num_features,
            optimizer=self.optimizer,
            lr=trainer_hp["lr"],
            max_epoch=trainer_hp["max_epoch"],
            early_stopping_round=trainer_hp["early_stopping_round"],
            device=self.device,
            weight_decay=trainer_hp["weight_decay"],
            feval=self.feval,
            init=True,
            *self.args,
            **self.kwargs,
        )

        return ret

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=self.device)
        link_labels[: pos_edge_index.size(1)] = 1.0
        return link_labels

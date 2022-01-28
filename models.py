from pathlib import Path

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

from TransFG.models.modeling import VisionTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from global_params import data_source_path


class NDGModel(nn.Module):
    def __init__(self):
        super(NDGModel, self).__init__()

    @property
    def latent_layer(self) -> nn.Module:
        raise NotImplementedError

    @property
    def latent_to_pred(self) -> nn.Module:
        raise NotImplementedError

    def latent_to_bool(self, saved_output) -> torch.BoolTensor:
        raise NotImplementedError

    def save_hook(self, x):
        return x

    def bool_to_latent(self, nodes) -> torch.Tensor:
        raise NotImplementedError

    def new_params(self):
        return None


class ALLNLIModel(NDGModel):
    def __init__(self, pretrained="sentence_transformers"):
        super(ALLNLIModel, self).__init__()
        if pretrained == "distilroberta-base":
            raise NotImplementedError
            self.config = AutoConfig.from_pretrained(pretrained)
            self.config.num_labels = 3
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained, config=self.config)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        elif pretrained == "sentence_transformers":
            ce = CrossEncoder('distilroberta-base', num_labels=3)
            self.model = ce.model
            self.model.classifier = ModifiedRobertaClassificationHead(self.model.config)
            self.model.init_weights()
            self.tokenizer = ce.tokenizer
        else:
            raise NotImplementedError

    def forward(self, input):
        ret = self.model(**input, return_dict=True).logits
        return ret

    @property
    def latent_layer(self):
        # pre sigmoid
        return self.model.classifier.dense

    @property
    def latent_to_pred(self):
        p = self.model.classifier._latent_to_pred
        return p

    def latent_to_bool(self, saved_output):
        latent = saved_output > 0
        return latent

    def bool_to_latent(self, nodes):
        nodes = nodes.float()
        # latent = (nodes - 0.5) * 2
        return nodes

    def new_params(self):
        names = ['lm_head.dense.bias', 'lm_head.dense.weight', 'roberta.pooler.dense.bias',
                 'roberta.pooler.dense.weight', 'lm_head.bias', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight',
                 'lm_head.layer_norm.weight']
        params = []
        for name, p in self.model.named_parameters():
            if name in names:
                params.append(p)
        params += list(self.model.classifier.parameters())
        return params


class NLIModel(NDGModel):
    def __init__(self, pretrained="sentence-transformers/all-MiniLM-L6-v2"):
        super(NLIModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained)
        if pretrained == "bert-base-uncased":
            raise NotImplementedError
            self.hidden = 768
        elif pretrained == "sentence-transformers/all-MiniLM-L6-v2":
            self.hidden = 384
        else:
            raise NotImplementedError
        self.head = nn.ModuleList([nn.ReLU(), nn.Linear(self.hidden, 3)])

    def forward(self, input):
        pred = self.model(**input.data)
        pred = pred.last_hidden_state
        out = pred[:, 0, :]
        for l in self.head:
            out = l(out)
        return out


class CUB200Model(NDGModel):
    def __init__(self):
        super(CUB200Model, self).__init__()
        from TransFG.train import get_parser, setup
        args = get_parser().parse_args("")
        args.pretrained_dir = Path(str(data_source_path / "vit"))
        args.actv = 2
        head_hidden_size = None
        self.args, self.model = setup(args, head_hidden_size)
        self.model: VisionTransformer

    def forward(self, *input):
        ret = self.model(*input)
        return ret

    @property
    def latent_layer(self) -> nn.Module:
        return self.model.part_head[0]

    @property
    def latent_to_pred(self) -> nn.Module:
        # pre relu
        if self.args.actv:
            p = nn.Sequential(*self.model.part_head[1:])
        else:
            p = nn.Identity()

        return p

    def latent_to_bool(self, saved_output) -> torch.BoolTensor:
        return saved_output > 0

    def save_hook(self, x):
        return x

    def bool_to_latent(self, nodes) -> torch.Tensor:
        return nodes.float()

    def new_params(self):
        return self.model.part_head.parameters()


class CUB200ResNet(NDGModel):
    def __init__(self):
        super(CUB200ResNet, self).__init__()
        from TransFG.train import get_parser, setup
        args = get_parser().parse_args("")
        args.pretrained_dir = Path("/drive/data/vit")
        args.actv = 2
        head_hidden_size = None
        self.args, self.model = setup(args, head_hidden_size)
        self.model: VisionTransformer

    def forward(self, *input):
        ret = self.model(*input)
        return ret

    @property
    def latent_layer(self) -> nn.Module:
        return self.model.part_head[0]

    @property
    def latent_to_pred(self) -> nn.Module:
        # pre relu
        if self.args.actv:
            p = nn.Sequential(*self.model.part_head[1:])
        else:
            p = nn.Identity()

        return p

    def latent_to_bool(self, saved_output) -> torch.BoolTensor:
        return saved_output > 0

    def save_hook(self, x):
        return x

    def bool_to_latent(self, nodes) -> torch.Tensor:
        return nodes.float()

    def new_params(self):
        return self.model.part_head.parameters()


class MNISTM(NDGModel):
    def __init__(self, num_predicates=32):
        super(MNISTM, self).__init__()
        enc = MNISTSeq1(num_predicates)
        dec = MNISTSeq2(num_predicates)
        model = LatentModel(enc, dec, num_predicates)
        self.model = model

    def forward(self, image):
        ret = self.model(image)[0]
        return ret

    @property
    def latent_layer(self):
        return self.model.encoder

    @property
    def multi_latent(self):
        return [self.model.encoder.fc1, self.model.decoder.fc2]

    @property
    def latent_to_pred(self):
        return self.model.decoder

    def latent_to_bool(self, saved_output):
        latent = saved_output > 0.5
        return latent

    @property
    def multi_latent_to_bool(self):
        return [relu_latent_to_bool, relu_latent_to_bool]

    def bool_to_latent(self, nodes) -> torch.Tensor:
        return nodes.float()


def relu_hugging_slice(x):
    x = x[:, 0, :]
    x = x > 0
    return x


def multi_layer_hack(model):
    """
    To load old checkpoints
    """
    name = str(type(model))
    if "MNISTM" in name:
        return [model.model.encoder.fc1,
                model.model.decoder.fc2], [relu_latent_to_bool, relu_latent_to_bool]
    elif "Sentiment" in name:
        return [model.model.pre_classifier,
                model.model.distilbert.transformer.layer[5].ffn.lin1], [relu_latent_to_bool,
                                                                        relu_hugging_slice]
    elif "ALLNLIModel" in name:
        return [model.model.classifier.dense,
                model.model.roberta.encoder.layer[5].intermediate.dense], [relu_latent_to_bool, relu_hugging_slice]
    elif "CUB200" in name:
        return [model.model.part_head[0],
                model.model.transformer.encoder.part_layer.ffn.fc1], [relu_latent_to_bool, relu_hugging_slice]
    elif "Code" in name:
        return [model.model.classifier.dense,
                model.model.roberta.encoder.layer[11].intermediate.dense], [relu_latent_to_bool, relu_hugging_slice]
    else:
        raise NotImplementedError


def inter_layer_selection(model, dataset):
    if "MNIST" == dataset:
        return [model.encoder.
                    fc1, model.decoder.
                    fc2]
    elif "MNIST even" == dataset:
        return [model.
                    encoder.fc1, model.decoder.
                    fc2]
    elif "SST2" == dataset:
        return [model.pre_classifier,
                model.distilbert.transformer.layer[5].ffn.lin1]
    elif "AllNLI" == dataset:
        return [model.classifier.dense,
                model.roberta.encoder.layer[5].intermediate.dense]
    elif "CUB200" == dataset:
        return [model.part_head[0],
                model.transformer.encoder.part_layer.ffn.fc1]
    elif "Devign" == dataset:
        return [model.classifier.dense,
                model.roberta.encoder.layer[11].intermediate.dense]


def relu_latent_to_bool(saved_output):
    latent = saved_output > 0
    return latent


def sigmoid_latent_to_bool(saved_output):
    latent = saved_output > 0.5
    return latent


class MNISTParityM(MNISTM):
    def __init__(self, num_predicates=32):
        super(MNISTParityM, self).__init__()
        encoder = MNISTSeq1(num_predicates)
        decoder = MNISTSeq2(num_predicates, target=2)
        model = LatentModel(encoder, decoder, num_predicates)
        self.model = model


class CodeModel(NDGModel):
    def __init__(self):
        super(CodeModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")
        self.model.classifier = ModifiedRobertaClassificationHead(self.model.config)
        self.model.init_weights()

    def forward(self, input):
        o = self.model(**input)
        return o.logits

    @property
    def latent_layer(self):
        # pre sigmoid
        return self.model.classifier.dense

    @property
    def latent_to_pred(self):
        p = self.model.classifier._latent_to_pred
        return p

    def latent_to_bool(self, saved_output):
        latent = saved_output > 0
        return latent

    def bool_to_latent(self, nodes):
        nodes = nodes.float()
        # latent = (nodes - 0.5) * 2
        return nodes

    def new_params(self):
        names = ['lm_head.dense.bias', 'lm_head.dense.weight', 'roberta.pooler.dense.bias',
                 'roberta.pooler.dense.weight', 'lm_head.bias', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight',
                 'lm_head.layer_norm.weight']
        params = []
        for name, p in self.model.named_parameters():
            if name in names:
                params.append(p)
        params += list(self.model.classifier.parameters())
        return params


class Sentiment(NDGModel):
    def __init__(self, change_forward=False):
        super(Sentiment, self).__init__()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english")
        if change_forward:
            from transformers.models.distilbert.modeling_distilbert import SequenceClassifierOutput, MSELoss, \
                BCEWithLogitsLoss, CrossEntropyLoss
            def forward(
                    self,
                    input_ids=None,
                    attention_mask=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
            ):
                r"""
                labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                    Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                    config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                    If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                distilbert_output = self.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
                pooled_output = hidden_state[:, 0]  # (bs, dim)
                pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
                pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
                pooled_output = self.dropout(pooled_output)  # (bs, dim)
                logits = self.classifier(pooled_output)  # (bs, num_labels)

                loss = None
                if labels is not None:
                    if self.config.problem_type is None:
                        if self.num_labels == 1:
                            self.config.problem_type = "regression"
                        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                            self.config.problem_type = "single_label_classification"
                        else:
                            self.config.problem_type = "multi_label_classification"

                    if self.config.problem_type == "regression":
                        loss_fct = MSELoss()
                        if self.num_labels == 1:
                            loss = loss_fct(logits.squeeze(), labels.squeeze())
                        else:
                            loss = loss_fct(logits, labels)
                    elif self.config.problem_type == "single_label_classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    elif self.config.problem_type == "multi_label_classification":
                        loss_fct = BCEWithLogitsLoss()
                        loss = loss_fct(logits, labels)

                if not return_dict:
                    output = (logits,) + distilbert_output[1:]
                    return ((loss,) + output) if loss is not None else output

                return SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=distilbert_output.hidden_states,
                    attentions=distilbert_output.attentions,
                )

            self.model.forward = forward

    def forward(self, input):
        o = self.model(**input)
        return o.logits

    @property
    def latent_layer(self):
        # pre relu
        return self.model.pre_classifier

    @property
    def latent_to_pred(self):
        p = nn.Sequential(nn.ReLU(),
                          self.model.dropout,
                          self.model.classifier)
        return p

    def latent_to_bool(self, saved_output):
        latent = saved_output > 0
        return latent

    def bool_to_latent(self, nodes) -> torch.Tensor:
        return nodes.float()


class ModifiedRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self._latent_to_pred = nn.Sequential(nn.Sigmoid(),
                                             self.dropout,
                                             self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self._latent_to_pred(x)
        return x


class CallbackLatentHook:
    def __init__(self, model: NDGModel, layer=None, callback=None, latent_to_bool=None):
        self.model = model
        self.layer = layer or model.latent_layer
        self.callback = callback
        self.layer.register_forward_hook(self.save_output)
        self.layer.register_full_backward_hook(self.save_gradient)
        self.saved_output = None
        self.saved_gradient = None
        self.latent_to_bool = latent_to_bool or self.model.latent_to_bool

    def save_output(self, module, input, output):
        self.saved_output = self.model.save_hook(output)

    def save_gradient(self, module, grad_input, grad_output):
        """

        :param module:
        :param grad_input: input of the module
        :param grad_output: output of the module
        :return:
        """
        self.saved_gradient = grad_output

    def get_latent(self):
        if self.callback is None:
            return self.model.latent_to_bool(self.saved_output)
        else:
            ret = self.callback(self.saved_output)
            return ret

    def attribute(self, input, target):
        input.requires_grad = True
        model_output = self.model(input)
        loss = model_output[:, target]
        loss = loss.sum()
        loss.backward()
        return self.saved_output, self.saved_gradient, model_output

    def get_boo(self):
        latent = self.get_latent()
        boo = self.latent_to_bool(latent)
        boo = boo.long().cpu()
        return boo


class LatentModel(nn.Module):
    def __init__(self, encoder, decoder, num_predicates, discretize=False, only_pred=False):
        super(LatentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_predicates = num_predicates
        self.bridge = None
        self.discretize = discretize
        self.only_pred = only_pred

    def forward(self, x):
        nodes = self.encoder(x)
        if self.bridge is not None:
            nodes = self.bridge.deduce(nodes)
        if self.discretize:
            nodes[nodes > 0.5] = 1
            nodes[nodes <= 0.5] = 0
        pred = self.decoder(nodes)
        if not self.only_pred:
            return pred, nodes
        else:
            return pred


class MNISTSeq1(nn.Module):
    def __init__(self, num_predicates, relu=False):
        super(MNISTSeq1, self).__init__()
        self.relu = relu
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_predicates)
        self.hsm = torch.nn.Sigmoid()
        # self.ex = TrickFun()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.relu:
            output = torch.relu(x)
        else:
            # output = torch.sigmoid(self.ex.apply(x))
            output = torch.sigmoid(x)

        # if not self.training:
        #     output[output > 0.5] = 1
        #     output[output <= 0.5] = 0
        return output


class MNISTSeq2(nn.Module):
    def __init__(self, num_predicates, target=10, hidden=128):
        super(MNISTSeq2, self).__init__()
        self.num_predicates = num_predicates
        self.fc1 = nn.Linear(num_predicates, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, target)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class TrickFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 10

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

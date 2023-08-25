from transformers import GPT2Config, PreTrainedModel, LlamaConfig
import math
from torch import nn, Tensor
import torch
from transformers.utils import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2PreTrainedModel, GPT2Model, GPT2Attention, GPT2MLP
from typing import Optional, Tuple, Union
from collections import OrderedDict
import re
import tempfile
from tqdm import tqdm
from torch.distributed.pipeline.sync import Pipe
from torch.distributed import rpc
from datetime import datetime
from torch.nn import CrossEntropyLoss
from gpt2.structure_data import GenerateTraindata


def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=self.dtype)  # fp16 compatibility
    encoder_extended_attention_mask = (
        1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    return encoder_extended_attention_mask


def convert_head_mask_to_5d(self,  num_hidden_layers):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(
            0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        # We can specify head_mask for each layer
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    assert head_mask.dim(
    ) == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    # switch to float if need + fp16 compatibility
    head_mask = head_mask.to(dtype=self.dtype)
    return head_mask


def get_head_mask(
    head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
) -> Tensor:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (`int`):
            The number of hidden layers in the model.
        is_attention_chunked (`bool`, *optional*, defaults to `False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
        `[None]` for each layer.
    """
    if head_mask is not None:
        head_mask = convert_head_mask_to_5d(num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask


logger = logging.get_logger(__name__)

# embeding层


class PipeEmbedding(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.config = config
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(self, *data):
        input_ids, attention_mask = data
        past_key_values = None
        # attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.n_layer)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]

            attention_mask = attention_mask.to(dtype=next(
                self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * \
                torch.finfo(next(self.parameters()).dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_attention_mask = invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_attention_mask = None

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        # output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        hidden_atten_position_return = (
            hidden_states, attention_mask)
        return hidden_atten_position_return


class PipeDecodeBlock(nn.Module):
    def __init__(self, config: GPT2Config, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.config = config
        self.decoder_layer = GPT2Block(config=config, layer_idx=layer_index)

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]
        hidden_states, attention_mask = args

        _device = next(self.decoder_layer.parameters()).device

        layer_outputs = self.decoder_layer(
            hidden_states=hidden_states.to(_device),
            layer_past=None,
            attention_mask=attention_mask.to(_device),
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False)
        hidden_states = layer_outputs[0]
        hidden_atten_position_return = (
            hidden_states, attention_mask)
        return hidden_atten_position_return


class PipeLMHead(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, *args, **kwargs) -> torch.tensor:
        if len(args) == 1:
            args = args[0]
        hidden_states, attention_mask = args
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


class PipeGPT2ModelCasualLM:
    def __init__(self, config: GPT2Config,
                 trainedmodel: PreTrainedModel,
                 numgpus: int = None) -> None:
        self.base_config = config
        self.base_model = trainedmodel
        self.num_gpus = numgpus

    def create_pipe_model(self, gpu_status: bool = True):
        first_gpu = 0
        last_gpu = self.num_gpus-1
        pipe_model = nn.Sequential()
        embedding = PipeEmbedding(config=self.base_config).to(
            f'cuda:{first_gpu}' if gpu_status else 'cpu')
        pipe_model.add_module(
            name='embedding',
            module=embedding)

        for num in range(self.base_config.n_layer):
            pipe_model.add_module(
                name=f'layer{num}',
                module=PipeDecodeBlock(
                    config=self.base_config,
                    layer_index=num).to(f'cuda:{num//math.ceil(self.base_config.n_layer/self.num_gpus)}' if gpu_status else 'cpu')
            )
        lmhead = PipeLMHead(config=self.base_config).to(
            f'cuda:{last_gpu}' if gpu_status else 'cpu')

        # lmhead.lm_head.weight = embedding.wte.weight
        pipe_model.add_module(name="lmhead", module=lmhead)
        base_model_param = self.get_basemodel_params()
        pipe_model.load_state_dict(base_model_param, strict=False)
        return pipe_model

    def get_basemodel_params(self):
        base_model_param = OrderedDict(
            {self.transname(n): v for n, v in self.base_model.named_parameters()})
        return base_model_param

    def transname(self, name: str) -> str:
        if name.find("model.embed_tokens") != -1:
            return "emebdding.embed_tokens.weight"

        if name.find("model.layers") != -1:
            layer_index = re.findall(".([0-9][0-9]*).", name)[0]
            sub_name = re.findall(".[0-9][0-9]*.(.*)", name)[0]
            new_name = f"layer{layer_index}.decoder_layer.{sub_name}"
            return new_name

        if name.find("model.norm.weight") != -1:
            return "lmhead.norm.weight"

        if name.find("lm_head.weight") != -1:
            return "lmhead.lm_head.weight"
        return name


def CreatePipeModel(pipemodel: nn.Sequential, chunks: int = 2) -> nn.Sequential:
    pipemodel_pytorch = Pipe(pipemodel, chunks=chunks)
    return pipemodel_pytorch


# 初始化
def init_rpc():
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch
            # versions >= 1.8.1
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )


class PipeTrain():
    def __init__(self, model: nn.Sequential, config: LlamaConfig,) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1.0, gamma=0.95)
        self.get_total_params()
        self.total_loss = 0

    def train_mini_batchs(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        logits = self.model(*(input_ids, attention_mask))
        # logits = self.model(PipeEmbeddingInput(input_ids=input_ids))
        logits = logits.local_value().float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.show_loss(loss=loss)

    def train(self, total_epoch: int = 10000, dataset: GenerateTraindata = None):
        self.model.train()

        input_ids, labels, attention_mask = dataset.generate_data()

        for index in tqdm(range(total_epoch)):
            self.train_mini_batchs(
                input_ids=input_ids.to('cuda'),
                labels=labels.to('cuda'),
                attention_mask=attention_mask.to('cuda'))

    def show_loss(self, loss: torch.Tensor) -> None:
        loss = loss.item()
        self.total_loss += loss
        print(f"datetime:{datetime.now()}, loss:{loss:.3f}")

    def get_total_params(self):
        total_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
        print('Total parameters in model: {:,}'.format(total_params))


def train(numgpus: int = 0):
    init_rpc()
    # create pipe model
    base_config = GPT2Config(
        vocab_size=32000,
        n_embd=512,  # 4096,  #
        n_layer=16,  # 32,  #
        n_head=16,  # 32,  #
        # num_key_value_heads=None,
        activation_function="gelu_new",
        n_positions=2048,  # 2048,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5)

    base_model = GPT2Model(base_config)
    pgmc = PipeGPT2ModelCasualLM(
        config=base_config, trainedmodel=base_model, numgpus=numgpus)
    pipe_model = pgmc.create_pipe_model(
        gpu_status=True if numgpus > 0 else False)
    if numgpus > 0:
        pipe_model_pytorch = CreatePipeModel(
            pipemodel=pipe_model, chunks=numgpus)
    else:
        pipe_model_pytorch = CreatePipeModel(pipemodel=pipe_model, chunks=1)

    del base_model
    del pgmc

    generatedata = GenerateTraindata()
    pipetrain = PipeTrain(model=pipe_model_pytorch, config=base_config)
    pipetrain.train(total_epoch=100, dataset=generatedata)


if __name__ == "__main__":
    train(numgpus=2)

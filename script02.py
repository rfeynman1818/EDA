import torch.nn as nn
from transformers import SwinModel, SwinConfig, Swinv2Model, Swinv2Config
import logging
import os

logger = logging.getLogger(__name__)
__all__ = ['Swin']

class Swin(nn.Module):
    def __init__(self, model_name=None, use_v2=False, **kwargs):
        super(Swin, self).__init__()
        self.model = None
        self.model_name = model_name
        self.use_v2 = use_v2
        self.kwargs = kwargs

        if self.model_name:
            self._load_pretrained()
        else:
            self._init_from_config()

    def _load_pretrained(self):
        try:
            if self.use_v2:
                logger.info(f"Loading SwinV2 model: {self.model_name}")
                self.model = Swinv2Model.from_pretrained(
                    self.model_name, local_files_only=True
                )
            else:
                logger.info(f"Loading SwinV1 model: {self.model_name}")
                self.model = SwinModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
        except OSError as e:
            logger.warning(
                f"Could not load pretrained model '{self.model_name}'. "
                f"Error: {e}. Falling back to config initialization."
            )
            self._init_from_config()

    def _init_from_config(self):
        if self.use_v2:
            logger.info("Initializing SwinV2 model from scratch")
            configuration = Swinv2Config(**self.kwargs)
            self.model = Swinv2Model(configuration)
        else:
            logger.info("Initializing SwinV1 model from scratch")
            configuration = SwinConfig(**self.kwargs)
            self.model = SwinModel(configuration)

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        return x, outputs.hidden_states[1:4]

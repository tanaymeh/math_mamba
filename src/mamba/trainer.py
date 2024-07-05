import os
import wandb

import torch
from torch import nn

from transformers import Trainer
from transformers.training_args import OptimizerNames

from typing import Dict, Optional, Union, Any


class MambaTrainer(Trainer):
    def _wandb_log(self, **kwargs):
        for k, v in kwargs.items():
            wandb.log({k: v})

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        logits = model(input_ids).logits
        labels = input_ids.to(logits.device)

        # Get all logits except the last one
        shift_logits = logits[:, :-1, :].contiguous()

        # Get all labels except the first one
        labels = labels[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss()

        # Squeeze both shift logits and labels
        lm_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        #     kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        try:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        except:
            self.accelerator.backward(loss, **kwargs)

        # Log to W&B
        self._wandb_log(
            train_step_loss=loss.detach() / self.args.gradient_accumulation_steps
        )

        return loss.detach() / self.args.gradient_accumulation_steps

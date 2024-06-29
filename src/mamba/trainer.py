import os
import torch
from transformers import Trainer


class MambaTrainer(Trainer):
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

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)

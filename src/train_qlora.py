import click

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments

from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig

import warnings
warnings.simplefilter('ignore')

@click.command()
@click.option("--num_epochs", default=2, help="Number of epochs to train the model for")
@click.option("--lr", default=2e-4, help="Learning rate")
@click.option("--train_bs", default=2, help="Training batch size")
@click.option("--g_accum", default=2, help="Gradient accumulation steps")
@click.option("--max_seq_len", default=2048, help="Maximum context length of the model")
def train(num_epochs, lr, train_bs, g_accum, max_seq_len):
    model_id = "google/codegemma-7b"
    train_dataset = load_dataset("json", data_files="data/train_openmath_instruct_1.jsonl", split='train[:100000]')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'

    model, tokenizer = setup_chat_format(model, tokenizer)

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules='all-linear',
        task_type='CAUSAL_LM'
    )

    args = TrainingArguments(
        output_dir="models/OpenMath-CodeGemma-7b",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=g_accum,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=lr,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_len,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False
        }
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()

if __name__ == "__main__":
    train()
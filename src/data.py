import json
import click
from functools import partial
from datasets import load_dataset


@click.command()
@click.option(
    "--dataset",
    default="nvidia/OpenMathInstruct-1",
    help="Huggingface Dataset identifier",
)
@click.option(
    "--prompt",
    default="prompts/openmath_prompt.json",
    help="JSON file containing the system prompt.",
)
@click.option("--train_split", default=0.99, help="Ratio of samples for training set")
@click.option(
    "--savefile_suffix", default="openmath_data", help="Prefix of the saved data file"
)
def run(dataset, prompt, train_split, savefile_suffix):
    raw_ds = load_dataset(dataset).shuffle()
    names = ["question", "expected_answer"]

    # Read the prompt from the prompts file
    with open(prompt, "r") as fl:
        SYSTEM_MSG = json.load(fl)["prompt"]

    def create_conversation(sample):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": sample[names[0]]},
                {"role": "assistant", "content": sample[names[1]]},
            ]
        }

    # Load, shuffle, split and convert the training and validation dataset
    train_ds = (
        load_dataset(dataset, split=f"train[:{int(train_split*100)}%]")
        .shuffle()
        .map(
            create_conversation, remove_columns=raw_ds["train"].features, batched=False
        )
    )

    valid_ds = (
        load_dataset(dataset, split=f"train[{int(train_split*100)}%:]")
        .shuffle()
        .map(
            create_conversation, remove_columns=raw_ds["train"].features, batched=False
        )
    )

    print(
        f"[INFO] Saving {len(train_ds)} training samples and {len(valid_ds)} validation samples."
    )

    # Save as JSON files
    train_ds.to_json(f"data/train_{savefile_suffix}.jsonl", orient="records")
    valid_ds.to_json(f"data/valid_{savefile_suffix}.jsonl", orient="records")

    print(f"[INFO] Saved successfully to 'data/'")


if __name__ == "__main__":
    run()

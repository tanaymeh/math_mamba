import click
import json
from datasets import load_dataset


@click.command()
@click.option(
    "--dataset",
    default="nvidia/OpenMathInstruct-1",
    help="Huggingface Dataset identifier",
)
@click.option(
    "--config",
    default="None",
    help="Huggingface Dataset Config",
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
@click.option("--question_col", default="question", help="Name of the question column")
@click.option(
    "--answer_col", default="expected_answer", help="Name of the answer column"
)
def run(
    dataset, config, prompt, train_split, savefile_suffix, question_col, answer_col
):
    config = config if config != "None" else None
    raw_ds = load_dataset(dataset, name=config).shuffle()

    # Read the prompt from the prompts file
    with open(prompt, "r") as fl:
        SYSTEM_PROMPT = json.load(fl)["prompt"]

    # This function will return qna pair in a dict for each sample
    create_qna_pairs = lambda sample: {
        "question": sample[question_col],
        "answer": sample[answer_col],
    }

    # Apply this to the dataset and create train and validation datasets
    train_ds = (
        load_dataset(dataset, name=config, split=f"train[:{int(train_split*100)}%]")
        .shuffle()
        .map(create_qna_pairs, batched=False, remove_columns=raw_ds["train"].features)
    )

    valid_ds = (
        load_dataset(dataset, name=config, split=f"train[{int(train_split*100)}%:]")
        .shuffle()
        .map(create_qna_pairs, batched=False, remove_columns=raw_ds["train"].features)
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

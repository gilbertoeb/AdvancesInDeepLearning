from .base_llm import BaseLLM
from .data import Dataset, benchmark

from pathlib import Path
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import TrainingArguments, Trainer
import torch

# Github Copilot was used to help with the code completion.
def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    try:
        llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
        llm.model.eval()
        print(f"LoRA adapter loaded (SFT) from {model_path}")
    except Exception as e:
        print(f"Error loading LoRA adapter (SFT): {e}. Using base model.")
    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} <answer>{answer}</answer>{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(question: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    return {"question": question, "answer": f"{answer:.3f}"}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    llm = BaseLLM()
    train_dataset = Dataset("train")

    # Define LoRA configuration (you will update this based on the output)
    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # Get LoRA model
    model = get_peft_model(llm.model, lora_config)
    model.print_trainable_parameters()

    # Explicitly enable gradients for input embeddings of the LoRA model
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad_(True)

    if llm.device == "cuda":
        model.enable_input_require_grads()

    # Tokenize the dataset
    tokenized_train_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=3e-4,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True if llm.device == "cuda" else False,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the LoRA adapter
    adapter_path = Path(output_dir) / "adapter_model"
    trainer.save_model(adapter_path)

    # Move the final adapter to the required directory
    sft_model_path = Path(__file__).parent / "sft_model"
    sft_model_path.mkdir(exist_ok=True, parents=True)

    final_adapter_path = Path(output_dir) / "adapter_model"
    if final_adapter_path.exists():
        import shutil
        for item in final_adapter_path.iterdir():
            shutil.copy(item, sft_model_path)
        print(f"LoRA adapter saved to {sft_model_path}")
    else:
        print(f"Warning: Adapter model not found at {final_adapter_path}")

    test_model(sft_model_path)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    try:
        llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
        llm.model.eval()
        benchmark_result = benchmark(llm, testset, 100)
        print(f"(SFT) {benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
    except Exception as e:
        print(f"Error loading LoRA adapter for testing: {e}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
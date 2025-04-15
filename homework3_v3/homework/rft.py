from .base_llm import BaseLLM
from .data import Dataset, benchmark
from pathlib import Path
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import TrainingArguments, Trainer
import torch
import json

# Github Copilot was used to help with the code completion.
def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    try:
        llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
        llm.model.eval()
        print(f"LoRA adapter loaded (RFT) from {model_path}")
    except Exception as e:
        print(f"Error loading LoRA adapter (RFT): {e}. Using base model.")
    return llm


def tokenize_rft(tokenizer, question: str, correct_answer: float, reasoning: str):
    """
    Tokenize a RFT data element (question, correct_answer, reasoning).
    We train on the question and the reasoning. The label is the reasoning part.
    """
    full_text = f"{question} {reasoning}{tokenizer.eos_token}"
    question_len = len(tokenizer(question)["input_ids"])
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)

    input_ids = tokenized["input_ids"]
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if tokenized["attention_mask"][i] == 0:
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format a RFT example into a dictionary.
    """
    return {"question": question, "reasoning": reasoning, "correct_answer": f"{answer}"}


class TokenizedRFTDataset:
    def __init__(self, tokenizer, data: list, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize_rft(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    llm = BaseLLM()

    # Load the RFT dataset
    rft_data_path = Path(__file__).parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(f"RFT dataset not found at {rft_data_path}. Please run datagen.py first.")

    with open(rft_data_path, 'r') as f:
        rft_data = json.load(f)

    # Define LoRA configuration
    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        r=16,  # Increased LoRA rank for RFT
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Get LoRA model
    model = get_peft_model(llm.model, lora_config)
    model.print_trainable_parameters()

    # Explicitly enable gradients for input embeddings
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad_(True)

    if llm.device == "cuda":
        model.enable_input_require_grads()

    # Tokenize the RFT dataset
    tokenized_train_dataset = TokenizedRFTDataset(llm.tokenizer, rft_data, format_rft_example)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=1e-4,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        save_strategy="epoch",
        logging_steps=10,
        fp16=True if llm.device == "cuda" else False,
        max_grad_norm=1.0,
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

    # Move the final adapter to the rft_model directory
    rft_model_path = Path(__file__).parent / "rft_model"
    rft_model_path.mkdir(exist_ok=True, parents=True)

    final_adapter_path = Path(output_dir) / "adapter_model"
    if final_adapter_path.exists():
        import shutil
        for item in final_adapter_path.iterdir():
            shutil.copy(item, rft_model_path)
        print(f"LoRA adapter (RFT) saved to {rft_model_path}")
    else:
        print(f"Warning: Adapter model not found at {final_adapter_path}")

    test_model(rft_model_path)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    try:
        llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
        llm.model.eval()
        benchmark_result = benchmark(llm, testset, 100)
        print(f"(RFT) {benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
    except Exception as e:
        print(f"Error loading LoRA adapter (RFT) for testing: {e}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
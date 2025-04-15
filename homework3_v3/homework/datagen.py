from .cot import CoTModel
from .data import Dataset
import json
from tqdm import tqdm

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generates a dataset of question/correct_answer/reasoning tuples using a CoT model
    with rejection sampling.

    Args:
        output_json (str): Path to save the generated dataset in JSON format.
        oversample (int): Number of CoT completions to generate per question.
        temperature (float): Temperature to use for CoT model generation.
    """
    cot_model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct") # Use the 1.7B model
    train_dataset = Dataset("train")
    generated_data = []

    print(f"Generating RFT dataset with oversample={oversample}, temperature={temperature}")
    for question, correct_answer in tqdm(train_dataset.data, desc="Generating CoT completions"):
        if not isinstance(question, str) or not isinstance(correct_answer, (int, float)):
            continue  # Skip malformed data

        prompt = cot_model.format_prompt(question)
        outputs = cot_model.batched_generate([prompt] * oversample, temperature=temperature)

        best_completion = None
        for completion in outputs:
            try:
                extracted_answer = cot_model.parse_answer(completion)
                if abs(extracted_answer - correct_answer) < 1e-5:  # Check for numerical equality
                    best_completion = completion
                    break
            except ValueError:
                continue  # Ignore completions where answer parsing fails

        if best_completion:
            generated_data.append([question, correct_answer, best_completion])

    print(f"Generated {len(generated_data)} correct reasoning paths.")

    with open(output_json, 'w') as f:
        json.dump(generated_data, f, indent=4)

    print(f"RFT dataset saved to {output_json}")

if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)
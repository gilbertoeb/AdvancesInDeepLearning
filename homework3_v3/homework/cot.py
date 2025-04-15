from .base_llm import BaseLLM

# Github Copilot was used to help with the code completion.
class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that performs unit conversions accurately. Show your reasoning step-by-step, clearly stating the conversion factor and then the calculation needed to get the result. Enclose the final numerical answer within <answer> and </answer> tags. Be concise."},
            {"role": "user", "content": "Convert 60 inches to feet."},
            {"role": "assistant", "content": "Conversion factor: 1 foot = 12 inches. Calculation: To convert inches to feet, divide by 12. 60 / 12 = <answer>5</answer> feet."},
            {"role": "user", "content": "How does 4 years measure up in terms of week?"},
            {"role": "assistant", "content": "Conversion factor: 1 year ≈ 52 weeks. Calculation: To find the number of weeks, multiply by 52. 4 * 52 = <answer>208</answer> weeks."},
            {"role": "user", "content": "What is the measurement of 3 kg when converted into pound?"},
            {"role": "assistant", "content": "Conversion factor: 1 kilogram ≈ 2.20462 pounds. Calculation: To convert kg to pounds, multiply by 2.20462. 3 * 2.20462 = <answer>6.61386</answer> pounds."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt += "<|im_start|>assistant\n"
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    num_samples_to_inspect = 5

    for i in range(min(num_samples_to_inspect, len(testset.data))):
        item = testset.data[i]
        if len(item) >= 1 and isinstance(item[0], str):
            question = item[0]
            prompt = model.format_prompt(question)
            outputs = model.batched_generate([prompt], temperature=0.0)
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Generated Output: {outputs[0]}")
        else:
            print(f"Warning: Unexpected data format in testset at index {i}: {item}")

    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
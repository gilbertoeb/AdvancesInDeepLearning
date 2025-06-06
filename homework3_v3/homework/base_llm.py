from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Github Copilot was used to help with the code completion.
class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        # self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
        self.device = device
        self.tokenizer.padding_side = "left"  # Set padding side for correct generation

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs)
        # answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        outputs = self.model.generate(**inputs, **generation_kwargs)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
            ):
                batch_prompts = prompts[idx : idx + micro_batch_size]
                results.extend(self._batched_generate_internal(batch_prompts, num_return_sequences, temperature))
            return results

        return self._batched_generate_internal(prompts, num_return_sequences, temperature)

    def _batched_generate_internal(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        generation_kwargs = {
            "max_new_tokens": 100,  # Increased max_new_tokens
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences

        outputs = self.model.generate(**inputs, **generation_kwargs)

        # Only decode the generated tokens (after the input)
        generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        if num_return_sequences is None:
            return decoded_outputs
        else:
            # Reshape the flat list into a list of lists
            return [
                decoded_outputs[i * num_return_sequences : (i + 1) * num_return_sequences]
                for i in range(len(prompts))
            ]

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
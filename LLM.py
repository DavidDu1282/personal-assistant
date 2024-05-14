#%%
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

class LanguageModel:
    def __init__(self, model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        self.configure_peft_model()
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

        self.alpaca_prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{}\n### Input:\n{}\n### Response:\n{}''' 
        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN

    def configure_peft_model(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_alpha=16,
            lora_dropout=0,
            bias='none',
            use_gradient_checkpointing='unsloth',
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def generate_response(self, instruction, input='', max_new_tokens=500):
        prompt = self.format_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        text_streamer = TextStreamer(self.tokenizer)

        # Adjust generation settings to prevent looping
        generated = self.model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,  # Ensure EOS is respected
            do_sample=True,  # Enable sampling
            top_k=50,  # Use top_k sampling
            top_p=0.95,  # Use nucleus sampling
            temperature=0.9  # Slightly stochastic to avoid repetition
        )
        return generated

    def format_prompt(self, instruction, input):
        alpaca_formatted_prompt = self.alpaca_prompt.format(
                instruction, # instruction
                input, # input
                '', # output - leave this blank for generation!
            )
        return alpaca_formatted_prompt
#%%

if __name__ == '__main__':
    lm = LanguageModel('unsloth/Phi-3-mini-4k-instruct')
    response = lm.generate_response(instruction='can you give me the first 20 numbers of the fib sequence?', input = '')
    # print('Final Output:\n', response)
#%%
    # available models (anything unsloth supports): 
    # fourbit_models = [
    #     'unsloth/Phi-3-mini-4k-instruct', #super small model
    #     'unsloth/mistral-7b-bnb-4bit',
    #     'unsloth/mistral-7b-instruct-v0.2-bnb-4bit',
    #     'unsloth/llama-2-7b-bnb-4bit',
    #     'unsloth/yi-6b-bnb-4bit', #Chinese model!!!!
    #     'unsloth/gemma-7b-bnb-4bit',
    #     'unsloth/gemma-7b-it-bnb-4bit', # Instruct version of Gemma 7b
    #     'unsloth/gemma-2b-bnb-4bit',
    #     'unsloth/gemma-2b-it-bnb-4bit', # Instruct version of Gemma 2b
    #     'unsloth/llama-3-8b-bnb-4bit', # [NEW] 15 Trillion token Llama-3
    # ] # More models at https://huggingface.co/unsloth
# %%

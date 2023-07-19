

import sys
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from common import settings
from openai_api_protocol import (
    ChatCompletionRequest
)
from .base import BaseModel, ALL_CHAT_MODELS


class WazardLMModel(BaseModel):
    model_name = "wizardlm"

    def load_model(self):
        model_path = settings["llm_models"][self.model_name]["model_path"]
        load_8bit = False
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.tokenizer = tokenizer
        self.model = model

    def cal_messages(self, messages):
        prompts = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        for message in messages:
            prompts += "{}: ".format(message.role)
            prompts += "{} ".format(message.content)
        prompts += "assistant:"
        return prompts

    def chat(self, request: ChatCompletionRequest):
        prompts = self.cal_messages(request.messages)
        inputs = self.tokenizer(prompts, return_tensors="pt")
        device = "cuda"
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=request.max_length,
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        output = output[0].split("assistant:")[-1].strip()
        return output


ALL_CHAT_MODELS[WazardLMModel.model_name] = WazardLMModel

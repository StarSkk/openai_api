

import sys
import torch
import traceback
import transformers
from queue import Queue
from threading import Thread
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from common import settings
from openai_api_protocol import (
    ChatCompletionRequest
)
from .base import BaseModel, ALL_CHAT_MODELS


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


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
        self.tokenizer = tokenizer  # type: LlamaTokenizer
        self.model = model  # type: LlamaForCausalLM

    def cal_messages(self, messages):
        prompts = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        for message in messages:
            prompts += "{}: ".format(message.role)
            prompts += "{} ".format(message.content)
        prompts += "assistant:"
        return prompts

    def cal_generate_params(self, request):
        prompts = self.cal_messages(request.messages)
        inputs = self.tokenizer(prompts, return_tensors="pt")
        device = "cuda"
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
        )
        params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": request.max_length,
        }
        return params

    def chat(self, request: ChatCompletionRequest):
        generate_params = self.cal_generate_params(request)
        with torch.no_grad():
            generation_output = self.model.generate(**generate_params)
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=True)
        output = output[0].split("assistant:")[-1].strip()
        return output

    def stream_chat(self, request):
        generate_params = self.cal_generate_params(request)

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                self.model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        def new_gen():
            for output in generate_with_streaming(**generate_params):
                decoded_output = self.tokenizer.decode(output)

                if output[-1] in [self.tokenizer.eos_token_id]:
                    break

                yield decoded_output.split("assistant:")[-1].strip()
        return new_gen()


ALL_CHAT_MODELS[WazardLMModel.model_name] = WazardLMModel


from transformers import AutoTokenizer, AutoModel

from common import settings
from openai_api_protocol import (
    ChatCompletionRequest
)
from .base import BaseModel, ALL_CHAT_MODELS


class ChatGLMModel(BaseModel):
    model_name = "chatglm"

    def load_model(self):
        model_path = settings["llm_models"][self.model_name]["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def cal_messages(self, messages):
        query = messages[-1].content
        prev_messages = messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == "system":
            query = prev_messages.pop(0).content + query

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                    history.append([prev_messages[i].content, prev_messages[i+1].content])

        return query, history

    def chat(self, request: ChatCompletionRequest):
        query, history = self.cal_messages(request.messages)
        response, _ = self.model.chat(
            self.tokenizer,
            query,
            history=history,
            max_length=request.max_length,
            top_p=request.top_p,
            temperature=request.temperature)
        return response

    def stream_chat(self, request: ChatCompletionRequest):
        query, history = self.cal_messages(request.messages)
        generator = self.model.stream_chat(
            self.tokenizer,
            query,
            history,
            max_length=request.max_length,
            top_p=request.top_p,
            temperature=request.temperature)
        def new_gen():
            for new_response, _ in generator:
                yield new_response
        return new_gen()


ALL_CHAT_MODELS[ChatGLMModel.model_name] = ChatGLMModel



class BaseModel():
    model_name = "base"

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def load_model(self):
        pass

    def chat(self, requests):
        pass

    def stream_chat(self, request):
        pass


ALL_CHAT_MODELS = {}


import uvicorn
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from common import (
    app,
    logger,
    settings,
    default_port,
)
from llms.base import ALL_CHAT_MODELS
from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    DeltaMessage,
)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    logger.debug(str(request.messages).encode())
    if request.messages[-1].role != "user" or request.model != LLM_MODEL.model_name:
        raise HTTPException(status_code=400, detail="Invalid request")

    if request.stream:
        generate = predict(request)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = LLM_MODEL.chat(request)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )
    logger.debug("response: {}".format(response.encode()))

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(request: ChatCompletionRequest):
    response = "".encode()
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=LLM_MODEL.model_name, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    current_length = 0

    for new_response in LLM_MODEL.stream_chat(request):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        response += new_text.encode()
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=LLM_MODEL.model_name, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    logger.debug("response: {}".format(response))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=LLM_MODEL.model_name, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    llm_type = settings["llm_type"]
    port = default_port
    try:
        cls = ALL_CHAT_MODELS[llm_type]
        LLM_MODEL = cls()
        LLM_MODEL.load_model()
    except Exception as e:
        logger.error(e)

    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)

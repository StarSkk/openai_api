import openai

if __name__ == "__main__":
    openai.api_base = "http://localhost:8001/v1"
    openai.api_key = "none"

    model_name = "wizardlm"
    # model_name = "chatglm"

    if model_name in ["chatglm", "wizardlm"]:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个人工智能助理"},
                {"role": "user", "content": "你好"}
            ],
        )
        print(completion.choices[0].message.content)

    if model_name in ["chatglm", "wizardlm"]:
        message = ""
        for chunk in openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个人工智能助理"},
                {"role": "user", "content": "你好"}
            ],
            stream=True
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                print(chunk.choices[0].delta.content, end="", flush=True)

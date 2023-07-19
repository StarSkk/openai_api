import openai

if __name__ == "__main__":
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"

    test_wizardlm = False
    if not test_wizardlm:
        for chunk in openai.ChatCompletion.create(
            model="chatglm",
            messages=[
                {"role": "system", "content": "你是一个人工智能助理"},
                {"role": "user", "content": "你好"}
            ],
            stream=True
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                print(chunk.choices[0].delta.content, end="", flush=True)
    else:
        completion = openai.ChatCompletion.create(
            model="wizardlm",
            messages=[
                {"role": "system", "content": "你是一个人工智能助理"},
                {"role": "user", "content": "你好"}
            ],
        )
        print(completion.choices[0].message)



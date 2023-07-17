# OpenAI接口规范
通过对不同大模型进行封装，以提供统一输入输出的API供外部调用。
以[chatglm(2)-6b](#https://huggingface.co/THUDM/chatglm2-6b)为例分两部分来介绍[接口如何调用](#API调用示例)以及[如何封装代码](#API封装示例)。
## API调用示例
### 输入
python 示例
```
import openai
if __name__ == "__main__":
    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create(
        model="chatglm2-6b",
        messages=[
            {"role": "system", "content": "你是一个人工智能助理"},
            {"role": "user", "content": "你好"}
        ],
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True)
```
requests 示例
```
import json
import requests
if __name__ == "__main__":
    header = {"Content-Type": "application/json"}
    data = {
        "model": "chatglm2-6b",
        "messages": [
            {"role": "system", "content": "你是一个人工智能助理"},
            {"role": "user", "content": "你好"}
        ],
        "stream": True,
    }
    url = "http://localhost:8000/v1/chat/completions"
    response = requests.post(url, headers=header, data=json.dumps(data))

    for line in response.iter_lines():
        line_str = str(line, encoding='utf-8')
        if line_str.startswith("data:"):
            if line_str.startswith("data: [DONE]"):
                break
            line_json = json.loads(line_str[5:].strip())
            delta = line_json['choices'][0]['delta']
            if 'content' in delta:
                print(delta["content"], end="", flush=True)

```
Request参数说明
| 参数              | 类型      | 默认值   | 说明                        |
| ----------------- | --------- | -------- | --------------------------- |
| model             | string    | null     | 要使用的模型ID              |
| messages          | array     | null     | 至今为止对话的消息列表      |
| temperature       | number    | 0.95     | 样本温度, 取值0到2          |
| top_p             | number    | 0.7      | 核采样,取值0到1             |
| stream            | boolean   | false    | 支持流式返回                |
| max_tokens        | integer   | 2048     | 聊天中生成的最大tokens      |

### 输出
输出示例 (stream=False)
```
{
  "model": "chatglm2-6b",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好👋！我是人工智能助理 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1689579393
}
```

## API封装示例
python 示例
参考[chatglm_api.py](https://github.com/StarSkk/openai_api/blob/main/chatglm_api.py)

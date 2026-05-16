import base64
import mimetypes
import os

from openai import OpenAI

from models.base_model import BaseModel


def _get_api_key(config):
    api_key = getattr(config, "api_key", None)
    if api_key:
        return api_key

    api_key_env = getattr(config, "api_key_env", "SILICONFLOW_API_KEY")
    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key

    raise ValueError(
        f"SiliconFlow API key is missing. Set {api_key_env} or fill api_key in the model yaml."
    )


def _encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


class SiliconFlowBase(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = self._get_model_name()
        self.client = OpenAI(
            api_key=_get_api_key(self.config),
            base_url=self.config.base_url,
        )

    def _get_model_name(self):
        return getattr(self.config, "model", None) or self.config.model_id

    def create_ask_message(self, question):
        return {
            "role": "user",
            "content": question,
        }

    def create_ans_message(self, answer):
        return {
            "role": "assistant",
            "content": answer,
        }

    def _complete(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        return response.choices[0].message.content

    def predict(self, question, texts=None, images=None, history=None):
        messages = self.process_message(question, texts, images, history)
        result = self._complete(messages)
        messages.append(self.create_ans_message(result))
        return result, messages

    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str):
                return False
        return True


class SiliconFlowTextModel(SiliconFlowBase):
    def create_text_message(self, texts, question):
        prompt = ""
        for text in texts:
            prompt += text + "\n"
        return {
            "role": "user",
            "content": f"{prompt}\n{question}",
        }


class SiliconFlowVisionModel(SiliconFlowBase):
    def create_ask_message(self, question):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }

    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        return {
            "role": "user",
            "content": content,
        }

    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image(image_path)},
                }
            )
        content.append({"type": "text", "text": question})
        return {
            "role": "user",
            "content": content,
        }

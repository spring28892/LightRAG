import os
import google.generativeai as genai
from .base import BaseLLM
from lightrag.core.types import LLMOutput

# 從環境變數讀取 Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class GeminiLLM(BaseLLM):
    def __init__(self, model: str = "gemini-flash-latest", **kwargs):
        super().__init__()
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
        
        self.model_name = model
        self.client = None # Gemini SDK 不需要 client 物件
        genai.configure(api_key=GOOGLE_API_KEY)
        print(f"GeminiLLM initialized with model: {self.model_name}")

    def call(self, input: str, **kwargs) -> LLMOutput:
        """
        Main method to call the Gemini API.
        It needs to handle the response and format it into LLMOutput.
        """
        try:
            model = genai.GenerativeModel(self.model_name)
            # 檢查是否有 JSON 格式要求
            generation_config = {}
            if kwargs.get("json", False) or kwargs.get("response_format", {}).get("type") == "json_object":
                 generation_config["response_mime_type"] = "application/json"

            response = model.generate_content(input, generation_config=generation_config)
            
            output = LLMOutput(
                raw_output=response.text,
                # Gemini API 的 usage metadata 藏在比較深層的地方，這裡先簡化
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} 
            )
            return output
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # 可以加上 response.prompt_feedback 來查看是否被安全設定阻擋
            try:
                print(f"Prompt Feedback: {response.prompt_feedback}")
            except:
                pass
            return LLMOutput(raw_output="", error=str(e))
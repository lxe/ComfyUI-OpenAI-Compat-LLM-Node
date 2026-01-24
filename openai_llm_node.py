import requests
import json
import base64
import io
import numpy as np
from PIL import Image

class OpenAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello",
                    "placeholder": "Enter your prompt here..."
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "https://api.openai.com/v1/chat/completions",
                    "placeholder": "OpenAI-compatible endpoint URL"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Your API token"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "gpt-4-vision-preview",
                    "placeholder": "Model name"
                }),
                "max_tokens": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "image_detail": (["low", "high", "auto"], {
                    "default": "auto"
                }),
                 "compat_mode": ("BOOLEAN", {
                    "default": False
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optionally enter your system/developer prompt here."
                }),
           }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"
    
    def _encode_image_to_base64(self, image_tensor):
        """Convert ComfyUI image tensor to base64 encoded string"""
        try:
            # ComfyUI images are typically in format [batch, height, width, channels] with values 0-1
            if len(image_tensor.shape) == 4:
                # Take first image from batch
                image_array = image_tensor[0]
            else:
                image_array = image_tensor
            
            # Convert from 0-1 float to 0-255 uint8
            if image_array.max() <= 1.0:
#                image_array = (image_array * 255).astype(np.uint8)
                image_array = (image_array * 255).numpy().astype(np.uint8)
            
            # Convert numpy array to PIL Image
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image_array, 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA image
                pil_image = Image.fromarray(image_array, 'RGBA')
            else:
                # Grayscale or other format
                pil_image = Image.fromarray(image_array)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")
    
    def generate_text(self, prompt, endpoint, api_token, model="gpt-4-vision-preview", max_tokens=150, temperature=0.7, image=None, image_detail="auto", compat_mode=False, system_prompt=""):
        try:
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            # Construct message content
            if image is not None:
                # Encode image to base64
                image_data_url = self._encode_image_to_base64(image)
                
                # Create multimodal message with both text and image
                message_content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": image_detail
                        }
                    }
                ]
            else:
                # Text-only message
                message_content = prompt
            
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message_content}
                ],
                "temperature": temperature,
                "reasoning_effort": "none"
            }
            
            if compat_mode:
                data["max_tokens"] = max_tokens
            else:
                data["max_completion_tokens"] = max_tokens
            
            
            if system_prompt:
                if compat_mode:
                    data["messages"].insert(0, {
                        "role": "system",
                        "content": system_prompt
                    })
                else:
                    data["messages"].insert(0, {
                        "role": "developer",
                        "content": system_prompt
                    })
            
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return (content,)
            else:
                return ("Error: No response content found",)
                
        except requests.exceptions.RequestException as e:
            return (f"Request Error: {str(e)}",)
        except json.JSONDecodeError as e:
            return (f"JSON Error: {str(e)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)


import os
import sys

sys.path.append(
	os.path.dirname(os.path.abspath(__file__))
)

from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image

import torchvision.transforms as T
import torch

def caption_cogvlm(image, max_length):
	print("Captioning image with CogVLM")
	tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
	model = AutoModelForCausalLM.from_pretrained(
		'THUDM/cogvlm-chat-hf',
		torch_dtype=torch.bfloat16,
		low_cpu_mem_usage=True,
		trust_remote_code=True
	).to('cuda').eval()
	transform = T.ToPILImage()
	query = 'Describe the content of the image in a clear and concise sentence, focusing on the main objects, actions, and context, while maintaining natural language fluency'
	inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[transform(item) for item in image])
	gen_kwargs = {"max_length": max_length, "do_sample": False}
	with torch.no_grad():
		outputs = model.generate(**inputs, **gen_kwargs)
		outputs = outputs[:, inputs['input_ids'].shape[1]:]
	print("Finished captioning image with CogVLM")
	return tokenizer.decode(outputs[0])

class CogVLMCaption:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image" : ('IMAGE', {}),
				"max_length" : ('INT', {})
			}
		}
	CATEGORY = "caption"
	FUNCTION = "main"
	RETURN_TYPES = ("STRING", )

	def main(self, image, max_length):
		result = caption_cogvlm(image, max_length)
		return (result, )

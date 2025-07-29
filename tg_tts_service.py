import torch
import os
import io
import json
from typing import *
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from torchaudio._extension.utils import _init_dll_path
_init_dll_path() # I LOVE PYTORCH I LOVE PYTORCH I LOVE PYTORCH FUCKING TORCHAUDIO SUCKS ASS
import io
from pydub import AudioSegment, effects  
import json
from flask import Flask, request, send_file
from tqdm import tqdm
from indextts.infer import IndexTTS
import threading
print("I: Loading IndexTTS into memory...")
tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device="cuda:0", use_cuda_kernel=False)
tts_lock = threading.Lock()
print("Done loading.")
voice_name_mapping = {}
use_voice_name_mapping = True
with open("./voice_mapping.json", "r") as file:
	voice_name_mapping = json.load(file)
	if len(voice_name_mapping) == 0:
		use_voice_name_mapping = False

app = Flask(__name__)
letters_to_use = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
random_factor = 0.35
os.makedirs('samples', exist_ok=True)
trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))
latent_cache = {}
voice_name_mapping_reversed = {v: k for k, v in voice_name_mapping.items()}
global request_count

@app.route("/generate-tts")
def text_to_speech():
	with tts_lock:
		text = request.json.get("text", "")
		voice = request.json.get("voice", "")
		print(voice + " says, " + "\"" + text + "\"")
		if use_voice_name_mapping:
			voice = voice_name_mapping_reversed[voice]
		result = None
		speaker_id = "NO SPEAKER"
		actual_text_found = False
		with io.BytesIO() as data_bytes:
			for i, letter in enumerate(text):
				if letter in letters_to_use:
					actual_text_found = True
					break
			if not actual_text_found:
				stub_file = AudioSegment.empty()
				stub_file.set_frame_rate(24000)
				stub_file.export(data_bytes, format = "wav")
				result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
				return result
			with torch.inference_mode():
				final_letter = text[-1]
				acceptable_punctuation = [".", "?", "!"]
				if not final_letter in acceptable_punctuation:
					#print("Forgot punctuation, adding . ")
					text += ". "
				# Inference
				loaded_speaker = latent_cache[voice]
				tts.infer_tg(cached_voice=loaded_speaker, text=text, output_path=data_bytes)
				rawsound = AudioSegment.from_file(io.BytesIO(data_bytes.getvalue()), "wav")  
				normalizedsound = effects.normalize(rawsound)  
				normalizedsound.export(data_bytes, format="wav")
				result = send_file(io.BytesIO(data_bytes.getvalue()), mimetype="audio/wav")
		return result

@app.route("/tts-voices")
def voices_list():
	if use_voice_name_mapping:
		data = list(voice_name_mapping.values())
		data.sort()
		return json.dumps(data)
	
@app.route("/health-check")
def tts_health_check():
    return f"OK: 1", 200

if __name__ == "__main__":
	from waitress import serve
	print("Beginning voice caching")
	for k,v in tqdm(voice_name_mapping.items()):
		latent_cache[k] = tts.get_speaker_latents(k)
	print("Cached voices.")
	print("Warming model up...")
	with tts_lock:
		trash = io.BytesIO()
		tts.infer_tg(cached_voice=list(latent_cache.values())[0], text="The quick brown fox jumps over the lazy dog.", output_path=trash)
		del trash
	print("Serving TTS on :5003")
	serve(app, host="0.0.0.0", port=5003, backlog=32, channel_timeout=8)
	

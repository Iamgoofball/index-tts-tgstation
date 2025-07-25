import glob
import scipy
import scipy.signal as signal
import torch
import os
import io
import json
import gc
import random
import numpy as np
from typing import *
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence
import librosa
import numpy as np
import random
import torch
import numpy as np
import torch
from torchaudio._extension.utils import _init_dll_path
_init_dll_path() # I LOVE PYTORCH I LOVE PYTORCH I LOVE PYTORCH FUCKING TORCHAUDIO SUCKS ASS
import io
import librosa
from pydub import AudioSegment, effects  
import json
from flask import Flask, request, send_file, abort, make_response
import soundfile as sf
from tqdm import tqdm
from indextts.infer import IndexTTS

print("I: Loading IndexTTS into memory...")
tts = IndexTTS(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=True, device="cuda:0", use_cuda_kernel=False)
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

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, pipe_out=None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)

def audiosegment_to_librosawav(audiosegment):    
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr

blips_cache = {}

@app.route("/generate-tts-blips")
def text_to_speech_blips():
    global blips_cache
    text = request.json.get("text", "").upper()
    voice = request.json.get("voice", "")
    pitch_adjustment = request.json.get("pitch", "")
    print(voice + " (" + pitch_adjustment + ") blips, " + "\"" + text + "\"")
    if use_voice_name_mapping:
        voice = voice_name_mapping_reversed[voice]
    result = None
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
        with torch.no_grad():
            result_sound = AudioSegment.empty()
            if not os.path.exists('samples/' + voice) or len(os.listdir('samples/' + voice)) < 36:
                os.makedirs('samples/' + voice, exist_ok=True)
                loaded_speaker = tts.get_speaker_latents(voice)
                for i, value in enumerate(letters_to_use):
                    tts.infer_tg(cached_voice=loaded_speaker, text=value, output_path="samples/" + voice + "/" + value + ".wav")
                    blips_cache["samples/" + voice + "/" + value + ".wav"] = AudioSegment.from_file("samples/" + voice + "/" + value + ".wav")
            for i, letter in enumerate(text):
                if not letter.isalpha() or letter.isnumeric() or letter == " ":
                    continue
                if letter == ' ':
                    letter_sound = AudioSegment.empty()
                    new_sound = letter_sound._spawn(b'\x00' * (24000 // 3), overrides={'frame_rate': 24000})
                    new_sound = new_sound.set_frame_rate(24000)
                else:
                    if not i % 2 == 0:
                        continue # Skip every other letter
                    file_path = "samples/" + voice + "/" + letter + ".wav"

                    if not file_path in blips_cache:
                        continue

                    letter_sound = blips_cache[file_path]
                    stripped_sound = strip_silence(letter_sound)
                    raw = stripped_sound.raw_data[5000:-7500]
                    octaves = 1 + random.random() * random_factor
                    frame_rate = int(stripped_sound.frame_rate * (2.0 ** octaves))

                    new_sound = stripped_sound._spawn(raw, overrides={'frame_rate': frame_rate})
                    new_sound = new_sound.set_frame_rate(24000)
                    
                    result_sound += new_sound
            sf.write(data_bytes, librosa.effects.pitch_shift(audiosegment_to_librosawav(result_sound), sr=24000, n_steps=int(pitch_adjustment), bins_per_octave=48), 24000, format="wav")
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

@app.route("/pitch-available")
def pitch_available():
    return make_response("Pitch available", 200)

if __name__ == "__main__":
    from waitress import serve
    print("Beginning voice caching")
    for k,v in tqdm(voice_name_mapping.items()):
        latent_cache[k] = tts.get_speaker_latents(k)
        for letter in letters_to_use:
            file_path = "samples/" + k + "/" + letter + ".wav"

            if not os.path.exists(file_path):
                continue

            blips_cache[file_path] = AudioSegment.from_file(file_path)
    print("Cached voices.")
    print("Warming model up...")
    trash = io.BytesIO()
    tts.infer_tg(cached_voice=list(latent_cache.values())[0], text="The quick brown fox jumps over the lazy dog.", output_path=trash)
    del trash
    print("Serving TTS Blips on :5003")
    serve(app, host="0.0.0.0", port=5003, backlog=32, channel_timeout=8)
    

import os
import io
import gc
import subprocess
import requests
import re
import pysbd
import pydub
import random
import time
import numpy as np
import soundfile as sf
from blake3 import blake3
from flask import Flask, request, send_file, abort, make_response
from pydub.silence import detect_leading_silence
from stftpitchshift import StftPitchShift

trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))
tts_sample_rate = 24000
app = Flask(__name__)
segmenter = pysbd.Segmenter(language="en", clean=True)
radio_starts = ["./on1.wav", "./on2.wav"]
radio_ends = ["./off1.wav", "./off2.wav", "./off3.wav", "./off4.wav"]
authorization_token = os.getenv("TTS_AUTHORIZATION_TOKEN", "REPLACE_ME")
cached_messages = []
max_to_cache = 10
pitch_shifter = StftPitchShift(1024, 256, 24000)

def now() -> int:
    return time.time_ns() // 1_000_000


def hhmmss_to_seconds(string):
    new_time = 0
    separated_times = string.split(":")
    new_time = 60 * 60 * float(separated_times[0])
    new_time += 60 * float(separated_times[1])
    new_time += float(separated_times[2])
    return new_time

def audiosegment_to_librosawav(audiosegment):    
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr

def text_to_speech_handler(
    endpoint, voice, text, filter_complex, pitch, special_filters=[], 
):
    filter_complex = filter_complex.replace('"', "")
    data_bytes = io.BytesIO()
    final_audio = pydub.AudioSegment.empty()

    start_time = now()

    for sentence in segmenter.segment(text):
        sentence_audio = pydub.AudioSegment.empty()
        if endpoint == "http://haproxy:5003/generate-tts": # we dont cache blips for obvious reasons
            merged_text = voice + sentence
            hashed_message = blake3(merged_text.encode("utf-8")).hexdigest()
            if hashed_message in cached_messages and os.path.exists("./cache/" + hashed_message + "/"):
                cached_sentences = [f for f in os.listdir("./cache/" + hashed_message + "/") if os.path.isfile(os.path.join("./cache/" + hashed_message + "/", f))]
                if len(cached_sentences) >= max_to_cache:
                    sentence_audio = pydub.AudioSegment.from_file(os.path.join("./cache/" + hashed_message + "/", random.choice(cached_sentences)), "wav")
                else:
                    response = requests.get(
                        endpoint,
                        json={"text": sentence, "voice": voice},
                    )

                    if response.status_code != 200:
                        abort(response.status_code)
                    sentence_audio = pydub.AudioSegment.from_file(io.BytesIO(response.content), "wav")
                    sentence_audio.export("./cache/" + hashed_message + "/cached_" + str(len(cached_sentences)) + ".wav", format="wav")
            else:
                if not os.path.exists("./cache/" + hashed_message + "/"):
                    os.mkdir("./cache/" + hashed_message + "/")
                response = requests.get(
                    endpoint,
                    json={"text": sentence, "voice": voice},
                )

                if response.status_code != 200:
                    abort(response.status_code)
                sentence_audio = pydub.AudioSegment.from_file(io.BytesIO(response.content), "wav")
                sentence_audio.export("./cache/" + hashed_message + "/cached_0.wav", format="wav")
                cached_messages.append(hashed_message)
        else:
            response = requests.get(
                endpoint,
                json={"text": sentence, "voice": voice},
            )

            if response.status_code != 200:
                abort(response.status_code)
            sentence_audio = pydub.AudioSegment.from_file(io.BytesIO(response.content), "wav")
        sentence_silence = pydub.AudioSegment.silent(250, tts_sample_rate)
        sentence_audio += sentence_silence
        final_audio += sentence_audio
        # ""Goldman-Eisler (1968) determined that typical speakers paused for an average of 250 milliseconds (ms), with a range from 150 to 400 ms.""
        # (https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=10153&context=etd)
    if pitch != 0:
        pitch = 0.5 + (pitch - -12) * (1.5 - 0.5) / (12 - -12)
        numpy_audio = audiosegment_to_librosawav(final_audio)
        dtype = numpy_audio.dtype
        scale = np.finfo(dtype).max ** -1
        numpy_audio = numpy_audio.astype(np.float32) # use at least float32
        numpy_audio = numpy_audio * scale
        shifted_audio = pitch_shifter.shiftpitch(numpy_audio, pitch, normalization=True)
        dtype = np.float32
        scale = np.finfo(dtype).max
        shifted_audio = shifted_audio.clip(-1, +1) # preventively avoid clipping
        shifted_audio = (shifted_audio * scale).astype(dtype)
        sf.write(data_bytes, shifted_audio, 24000, format="wav")
        final_audio = pydub.AudioSegment.from_file(io.BytesIO(data_bytes.getvalue()), "wav")
    print(f"Total time to generate audio: {now() - start_time}")
    start_time = now()

    final_audio.export(data_bytes, format="wav")
    filter_complex = filter_complex.replace("%SAMPLE_RATE%", str(tts_sample_rate))
    ffmpeg_result = None
    if filter_complex != "":
        ffmpeg_result = subprocess.run(
            [
                "ffmpeg",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-filter_complex",
                filter_complex,
                "-c:a",
                "libvorbis",
                "-b:a",
                "64k",
                "-f",
                "ogg",
                "pipe:1",
            ],
            input=data_bytes.read(),
            capture_output=True,
        )
    else:
        if "silicon" in special_filters:
            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "wav",
                    "-i",
                    "pipe:0",
                    "-i",
                    "./SynthImpulse.wav",
                    "-i",
                    "./RoomImpulse.wav",
                    "-filter_complex",
                    "[0] aresample=44100 [re_1]; [re_1] apad=pad_dur=2 [in_1]; [in_1] asplit=2 [in_1_1] [in_1_2]; [in_1_1] [1] afir=dry=10:wet=10 [reverb_1]; [in_1_2] [reverb_1] amix=inputs=2:weights=8 1 [mix_1]; [mix_1] asplit=2 [mix_1_1] [mix_1_2]; [mix_1_1] [2] afir=dry=1:wet=1 [reverb_2]; [mix_1_2] [reverb_2] amix=inputs=2:weights=10 1 [mix_2]; [mix_2] equalizer=f=7710:t=q:w=0.6:g=-6,equalizer=f=33:t=q:w=0.44:g=-10 [out]; [out] alimiter=level_in=1:level_out=1:limit=0.5:attack=5:release=20:level=disabled",
                    "-c:a",
                    "libvorbis",
                    "-b:a",
                    "64k",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                input=data_bytes.read(),
                capture_output=True,
            )
        else:
            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "wav",
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "libvorbis",
                    "-b:a",
                    "64k",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                input=data_bytes.read(),
                capture_output=True,
            )

    ffmpeg_metadata_output = ffmpeg_result.stderr.decode()

    matched_length = re.search(r"time=([0-9:\\.]+)", ffmpeg_metadata_output)
    hh_mm_ss = matched_length.group(1)
    length = hhmmss_to_seconds(hh_mm_ss)

    print(f"ffmpeg result size: {len(ffmpeg_result.stdout)}")
    print(f"ffmpeg time: {now() - start_time}")

    start_time = now()

    export_audio = io.BytesIO(ffmpeg_result.stdout)
    if "radio" in special_filters:
        radio_audio = pydub.AudioSegment.from_file(random.choice(radio_starts), "wav")
        radio_audio += pydub.AudioSegment.from_file(
            io.BytesIO(ffmpeg_result.stdout), "ogg"
        )
        radio_audio += pydub.AudioSegment.from_file(random.choice(radio_ends), "wav")
        new_data_bytes = io.BytesIO()
        radio_audio.export(new_data_bytes, format="ogg")
        export_audio = io.BytesIO(new_data_bytes.getvalue())

    print(f"pydub time: {now() - start_time}")

    response = send_file(
        export_audio,
        as_attachment=True,
        download_name="identifier.ogg",
        mimetype="audio/ogg",
    )
    response.headers["audio-length"] = length
    return response


@app.route("/tts")
def text_to_speech_normal():
    if authorization_token != request.headers.get("Authorization", ""):
        abort(401)

    voice = request.args.get("voice", "")
    text = request.json.get("text", "")
    pitch = request.args.get("pitch", "")
    special_filters = request.args.get("special_filters", "")
    if pitch == "":
        pitch = "0"
    silicon = request.args.get("silicon", "")
    if silicon:
        special_filters = ["silicon"]

    filter_complex = request.args.get("filter", "")
    return text_to_speech_handler(
        "http://haproxy:5003/generate-tts",
        voice,
        text,
        filter_complex,
        int(pitch),
        special_filters
    )


@app.route("/tts-blips")
def text_to_speech_blips():
    if authorization_token != request.headers.get("Authorization", ""):
        abort(401)

    voice = request.args.get("voice", "")
    text = request.json.get("text", "")
    pitch = request.args.get("pitch", "")
    special_filters = request.args.get("special_filters", "")
    if pitch == "":
        pitch = "0"
    special_filters = special_filters.split("|")
    filter_complex = request.args.get("filter", "")
    return text_to_speech_handler(
        "http://blips:5003/generate-tts-blips",
        voice,
        text,
        filter_complex,
        int(pitch),
        special_filters
    )


@app.route("/tts-voices")
def voices_list():
    if authorization_token != request.headers.get("Authorization", ""):
        abort(401)

    response = requests.get(f"http://haproxy:5003/tts-voices")
    return response.content


@app.route("/health-check")
def tts_health_check():
    gc.collect()
    return "OK", 200

@app.route("/pitch-available")
def superpitch_available():
    if authorization_token != request.headers.get("Authorization", ""):
        abort(401)
    return make_response("Pitch available", 200)


if __name__ == "__main__":
    from waitress import serve

    print("Loading cached messages...")
    directories = [f for f in os.listdir("./cache/") if os.path.isdir(os.path.join("./cache/", f))]
    for directory in directories:
        cached_messages.append(directory)
    print("Loaded " + str(len(cached_messages)) + " messages.")
    serve(
        app,
        host="0.0.0.0",
        port=5002,
        backlog=32,
        channel_timeout=10,
    )

FROM docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /workspace

RUN apt update && \
  apt install -y \
  libsox-dev \
  ffmpeg \
  build-essential \
  cmake \
  libasound-dev \
  portaudio19-dev \
  libportaudio2 \
  libportaudiocpp0 \
  nvidia-cuda-toolkit \
  libvorbis-dev

RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY . .

RUN pip install -e . --no-build-isolation
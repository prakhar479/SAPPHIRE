FROM python:3.10-slim


# System deps for audio libs
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential libsndfile1 ffmpeg git pkg-config libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip
# Install core requirements. For heavy optional packages like essentia you may build separately.
RUN pip install -r requirements.txt || true


COPY . /app


ENTRYPOINT ["python", "music_multifeature_extract_v2.py"]
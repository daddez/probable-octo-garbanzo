# Usa un'immagine base ufficiale RunPod modernissima (PyTorch 2.9, CUDA 12.9)
# Questa versione supporta NATIVAMENTE le schede Blackwell (sm_120) senza dover fare downgrade/upgrade manuali
FROM runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2204

# Imposta variabili d'ambiente per evitare interazioni durante l'installazione
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Installa dipendenze di sistema (FFmpeg, ImageMagick, Node.js, e OpenCV/YOLO reqs)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    curl \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Fix policy di ImageMagick per permettere la lettura/scrittura dei PDF/Testi in MoviePy
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true
RUN sed -i 's/rights="none" pattern="LABEL"/rights="read|write" pattern="LABEL"/' /etc/ImageMagick-6/policy.xml || true
RUN sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@\*"/' /etc/ImageMagick-6/policy.xml || true

# Imposta la directory di lavoro
WORKDIR /app

# Copia i requisiti Python
COPY requirements.txt .

# Installa i pacchetti Python (PyTorch 2.9 è già integrato nell'immagine base!)
RUN pip install --no-cache-dir -r requirements.txt

# Installa il browser chromium necessario per il rendering CSS di PyCaps
RUN playwright install --with-deps chromium

# Configurazione del volume di rete (creiamo i mount point per sicurezza)
RUN mkdir -p /runpod-volume/barberpro/models/whisper \
    && mkdir -p /runpod-volume/barberpro/models/llm \
    && mkdir -p /runpod-volume/barberpro/models/yolo

# Copia l'intero worker (incluso handler.py e requisiti)
COPY . .

# Avvia l'handler serverless di RunPod
CMD ["python", "-u", "handler.py"]

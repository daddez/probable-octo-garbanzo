import runpod
import traceback
import sys
import faulthandler
import os
import time
from datetime import datetime

faulthandler.enable()

LOG_DIR = "/runpod-volume/barberpro/logs"
os.makedirs(LOG_DIR, exist_ok=True)
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000))
LOG_FILE = os.path.join(LOG_DIR, f"worker_{SESSION_ID}.log")

def write_log(msg: str):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}", flush=True)

write_log(f"Worker process started with args: {sys.argv}")

worker_operativo = False
messaggio_avaria = ""

import subprocess
import requests
import json
import numpy as np
import shutil
import base64
import urllib.request
import urllib.parse
import concurrent.futures
import random
import threading
import string

# ==========================================
# CONFIGURAZIONE PERCORSI E MODELLI
# ==========================================
NETWORK_VOL_DIR = "/runpod-volume/barberpro/models"
WHISPER_MODEL_DIR = os.path.join(NETWORK_VOL_DIR, "whisper")
LLM_DIR = os.path.join(NETWORK_VOL_DIR, "llm")
YOLO_MODEL_DIR = os.path.join(NETWORK_VOL_DIR, "yolo")
YOLO_PATH = os.path.join(YOLO_MODEL_DIR, "yolov8n.pt")

COMFYUI_DIR = "/runpod-volume/runpod-slim/ComfyUI"
VENV_PYTHON = "/runpod-volume/barberpro/venv/bin/python"
PYTHON_EXECUTABLE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
COMFYUI_PORT = "8188"
COMFYUI_URL = f"http://127.0.0.1:{COMFYUI_PORT}"

models_cache = {
    "whisper": None,
    "yolo": None,
    "llm": None
}

def load_ml_libraries():
    global VideoFileClip, concatenate_videoclips, CompositeVideoClip, ImageClip, vfx
    global WhisperModel, torch, pipeline, YOLO, Image
    if 'torch' not in globals():
        write_log("Loading heavy ML libraries (torch, transformers, moviepy, etc)...")
        from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, ImageClip, vfx
        from faster_whisper import WhisperModel
        import torch
        from transformers import pipeline
        from ultralytics import YOLO
        from PIL import Image
        
        globals()['VideoFileClip'] = VideoFileClip
        globals()['concatenate_videoclips'] = concatenate_videoclips
        globals()['CompositeVideoClip'] = CompositeVideoClip
        globals()['ImageClip'] = ImageClip
        globals()['vfx'] = vfx
        globals()['WhisperModel'] = WhisperModel
        globals()['torch'] = torch
        globals()['pipeline'] = pipeline
        globals()['YOLO'] = YOLO
        globals()['Image'] = Image
        write_log("Heavy ML libraries loaded successfully.")

def start_comfyui():
    global worker_operativo, messaggio_avaria
    write_log("Avvio procedura ComfyUI...")
    
    # Check volume con leggera tolleranza per attacco rete RunPod
    if not os.path.exists(COMFYUI_DIR):
        for _ in range(10):
            time.sleep(1)
            if os.path.exists(COMFYUI_DIR):
                break
                
    if not os.path.exists(COMFYUI_DIR):
        messaggio_avaria = f"Volume di rete assente in {COMFYUI_DIR}."
        write_log(messaggio_avaria)
        return False
        
    os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)
    os.makedirs(LLM_DIR, exist_ok=True)
    
    log_file_path = "/tmp/comfyui_startup.log"
    log_file = open(log_file_path, "w", encoding="utf-8")
    cmd = [PYTHON_EXECUTABLE, "main.py", "--listen", "127.0.0.1", "--port", COMFYUI_PORT]
    
    try:
        process = subprocess.Popen(cmd, cwd=COMFYUI_DIR, stdout=log_file, stderr=subprocess.STDOUT)
    except Exception as e:
        messaggio_avaria = f"Errore nell'avvio del processo ComfyUI: {str(e)}"
        write_log(messaggio_avaria)
        return False

    write_log("In attesa dell'avvio di ComfyUI locale (Timeout 120s)...")
    for _ in range(120):
        if process.poll() is not None:
            log_file.close()
            with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                error_log = f.read()
            messaggio_avaria = f"ComfyUI si è schiantato in fase di avvio.\n=== LOG ===\n{error_log}"
            write_log(messaggio_avaria)
            return False

        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=1)
            if response.status_code == 200:
                write_log("ComfyUI operativo e pronto a elaborare.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        
    messaggio_avaria = "TIMEOUT CRITICO: ComfyUI non ha risposto entro 120s."
    write_log(messaggio_avaria)
    return False

# ==========================================
# FUNZIONI DI SUPPORTO COMFYUI E API
# ==========================================
def get_flux_workflow(prompt):
    return {
      "6": {"inputs": {"text": prompt,"clip": ["38",0]},"class_type": "CLIPTextEncode"},
      "8": {"inputs": {"samples": ["13",0],"vae": ["10",0]},"class_type": "VAEDecode"},
      "9": {"inputs": {"filename_prefix": "ComfyUI","images": ["8",0]},"class_type": "SaveImage"},
      "10": {"inputs": {"vae_name": "full_encoder_small_decoder.safetensors"},"class_type": "VAELoader"},
      "12": {"inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors","weight_dtype": "default"},"class_type": "UNETLoader"},
      "13": {"inputs": {"noise": ["25",0],"guider": ["22",0],"sampler": ["16",0],"sigmas": ["48",0],"latent_image": ["47",0]},"class_type": "SamplerCustomAdvanced"},
      "16": {"inputs": {"sampler_name": "euler"},"class_type": "KSamplerSelect"},
      "22": {"inputs": {"model": ["12",0],"conditioning": ["26",0]},"class_type": "BasicGuider"},
      "25": {"inputs": {"noise_seed": random.randint(1, 10000000)},"class_type": "RandomNoise"},
      "26": {"inputs": {"guidance": 4,"conditioning": ["6",0]},"class_type": "FluxGuidance"},
      "38": {"inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors","type": "flux2","device": "default"},"class_type": "CLIPLoader"},
      "47": {"inputs": {"width": 720,"height": 1280,"batch_size": 1},"class_type": "EmptyFlux2LatentImage"},
      "48": {"inputs": {"steps": 20,"width": 720,"height": 1280},"class_type": "Flux2Scheduler"}
    }

def get_rmbg_workflow(filename, subject_prompt="character"):
    return {
      "2": {"inputs": {"image": filename}, "class_type": "LoadImage"},
      "77": {"inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}, "class_type": "CheckpointLoaderSimple"},
      "78": {"inputs": {"text": subject_prompt, "clip": ["77", 1]}, "class_type": "CLIPTextEncode"},
      "75": {
          "inputs": {
              "model": ["77", 0],
              "image": ["2", 0],
              "conditioning": ["78", 0],
              "threshold": 0.5,
              "refine_iterations": 2,
              "individual_masks": False,
              "positive_coords": "",
              "negative_coords": ""
          },
          "class_type": "SAM3_Detect"
      },
      "100": {"inputs": {"image": ["2", 0], "alpha": ["75", 0]}, "class_type": "JoinImageWithAlpha"},
      "3": {"inputs": {"filename_prefix": "ComfyUI_RMBG", "images": ["100", 0]}, "class_type": "SaveImage"}
    }

def get_wan2_i2v_workflow(image_filename, prompt):
    return {
      "97": {"inputs": {"image": image_filename},"class_type": "LoadImage"},
      "84": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"},"class_type": "CLIPLoader"},
      "90": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"},"class_type": "VAELoader"},
      "89": {"inputs": {"text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", "clip": ["84", 0]},"class_type": "CLIPTextEncode"},
      "93": {"inputs": {"text": prompt, "clip": ["84", 0]},"class_type": "CLIPTextEncode"},
      "98": {"inputs": {"width": 640, "height": 640, "length": 81, "positive": ["93", 0], "negative": ["89", 0], "vae": ["90", 0], "start_image": ["97", 0]},"class_type": "WanImageToVideo"},
      "96": {"inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"},"class_type": "UNETLoader"},
      "95": {"inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"},"class_type": "UNETLoader"},
      "104": {"inputs": {"shift": 5.0, "model": ["96", 0]},"class_type": "ModelSamplingSD3"},
      "103": {"inputs": {"shift": 5.0, "model": ["95", 0]},"class_type": "ModelSamplingSD3"},
      "86": {"inputs": {"add_noise": "enable", "noise_seed": random.randint(1, 10000000), "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 10, "return_with_leftover_noise": "enable", "model": ["104", 0], "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["98", 2]},"class_type": "KSamplerAdvanced"},
      "85": {"inputs": {"add_noise": "disable", "noise_seed": random.randint(1, 10000000), "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 10, "end_at_step": 20, "return_with_leftover_noise": "disable", "model": ["103", 0], "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["86", 0]},"class_type": "KSamplerAdvanced"},
      "87": {"inputs": {"samples": ["85", 0], "vae": ["90", 0]},"class_type": "VAEDecode"},
      "94": {"inputs": {"fps": 16, "images": ["87", 0]},"class_type": "CreateVideo"},
      "108": {"inputs": {"filename_prefix": "Wan2_i2v", "video": ["94", 0]},"class_type": "SaveVideo"}
    }

def upload_image_to_comfy(filepath):
    with open(filepath, "rb") as f:
        files = {"image": f}
        r = requests.post(f"{COMFYUI_URL}/upload/image", files=files).json()
        return r["name"]

def get_comfy_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{url_values}") as response:
        return response.read()

def subagent_comfyui_worker(scene, workspace_dir):
    prompt_base = scene.get("prompt", "")
    azione = scene.get("azione", "")
    
    if "engagement" in azione:
        prompt = prompt_base + ", highly engaging, surreal, viral, cinematic, masterpiece, 8k, highly detailed, visually stunning, rich vivid colors, emotional"
    elif "metaforica" in azione:
        prompt = prompt_base + ", metaphorical representation, artistic simile, evocative, conceptual art, beautiful lighting"
    else:
        prompt = prompt_base + ", realistic, technical documentary style, logical breakdown, clear, educational, factual, highly detailed photograph, professional"
        
    workflow = get_flux_workflow(prompt)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt_req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=10).json()
            if "error" in prompt_req:
                raise Exception(f"ComfyUI Error: {prompt_req['error']}")
                
            prompt_id = prompt_req['prompt_id']
            timeout_anomalia = 0
            while True:
                history_req = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
                if prompt_id in history_req:
                    outputs = history_req[prompt_id]['outputs']
                    for node_id in outputs:
                        if 'images' in outputs[node_id]:
                            img_info = outputs[node_id]['images'][0]
                            img_data = get_comfy_image(img_info['filename'], img_info['subfolder'], img_info['type'])
                            img_path = os.path.join(workspace_dir, f"asset_{prompt_id}.png")
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                            return img_path
                
                queue_req = requests.get(f"{COMFYUI_URL}/queue").json()
                pending = [p[1] for p in queue_req.get('queue_pending', [])]
                running = [p[1] for p in queue_req.get('queue_running', [])]
                if prompt_id not in pending and prompt_id not in running and prompt_id not in history_req:
                    timeout_anomalia += 1
                    if timeout_anomalia >= 5:
                        raise Exception("Job disappeared from ComfyUI queue.")
                else:
                    timeout_anomalia = 0
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2)
    return {"error": "Max retries exceeded"}

def subagent_comfyui_rmbg(image_path, workspace_dir, subject_prompt="character"):
    comfy_filename = upload_image_to_comfy(image_path)
    workflow = get_rmbg_workflow(comfy_filename, subject_prompt)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt_req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=10).json()
            if "error" in prompt_req:
                raise Exception(f"ComfyUI Error: {prompt_req['error']}")
                
            prompt_id = prompt_req['prompt_id']
            timeout_anomalia = 0
            while True:
                history_req = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
                if prompt_id in history_req:
                    outputs = history_req[prompt_id]['outputs']
                    for node_id in outputs:
                        if 'images' in outputs[node_id]:
                            img_info = outputs[node_id]['images'][0]
                            img_data = get_comfy_image(img_info['filename'], img_info['subfolder'], img_info['type'])
                            out_path = os.path.join(workspace_dir, f"rmbg_{prompt_id}.png")
                            with open(out_path, "wb") as f:
                                f.write(img_data)
                            return out_path
                
                queue_req = requests.get(f"{COMFYUI_URL}/queue").json()
                pending = [p[1] for p in queue_req.get('queue_pending', [])]
                running = [p[1] for p in queue_req.get('queue_running', [])]
                if prompt_id not in pending and prompt_id not in running and prompt_id not in history_req:
                    timeout_anomalia += 1
                    if timeout_anomalia >= 5:
                        raise Exception("Job disappeared from ComfyUI queue.")
                else:
                    timeout_anomalia = 0
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2)
    return {"error": "Max retries exceeded in RMBG"}

def subagent_comfyui_video(image_path, workspace_dir, prompt):
    comfy_filename = upload_image_to_comfy(image_path)
    workflow = get_wan2_i2v_workflow(comfy_filename, prompt)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt_req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=10).json()
            if "error" in prompt_req:
                raise Exception(f"ComfyUI Error: {prompt_req['error']}")
                
            prompt_id = prompt_req['prompt_id']
            timeout_anomalia = 0
            while True:
                history_req = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
                if prompt_id in history_req:
                    outputs = history_req[prompt_id]['outputs']
                    for node_id in outputs:
                        # Controllo se è un video o immagine
                        media_list = outputs[node_id].get('videos', outputs[node_id].get('gifs', outputs[node_id].get('images', [])))
                        if media_list:
                            vid_info = media_list[0]
                            vid_data = get_comfy_image(vid_info['filename'], vid_info['subfolder'], vid_info['type'])
                            out_ext = ".mp4" if 'videos' in outputs[node_id] else ".webm" # o fallback
                            out_path = os.path.join(workspace_dir, f"video_{prompt_id}{out_ext}")
                            with open(out_path, "wb") as f:
                                f.write(vid_data)
                            return out_path
                
                queue_req = requests.get(f"{COMFYUI_URL}/queue").json()
                pending = [p[1] for p in queue_req.get('queue_pending', [])]
                running = [p[1] for p in queue_req.get('queue_running', [])]
                if prompt_id not in pending and prompt_id not in running and prompt_id not in history_req:
                    timeout_anomalia += 1
                    if timeout_anomalia >= 15: # I2V richiede più tempo, tolleranza maggiore
                        raise Exception("Job disappeared from ComfyUI queue.")
                else:
                    timeout_anomalia = 0
                time.sleep(3)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2)
    return {"error": "Max retries exceeded in I2V"}

def subagent_video_chain(scena, asset_dir):
    # Genera immagine
    img_res = subagent_comfyui_worker(scena, asset_dir)
    if isinstance(img_res, dict) and "error" in img_res:
        return img_res
    
    # Se serve rimozione sfondo prima di animare
    if scena.get("rimuovi_sfondo"):
        soggetto = scena.get("soggetto_da_isolare", "character")
        rmbg_res = subagent_comfyui_rmbg(img_res, asset_dir, subject_prompt=soggetto)
        if isinstance(rmbg_res, dict) and "error" in rmbg_res:
            return rmbg_res
        img_res = rmbg_res
        
    # Anima l'immagine
    anim_prompt = "simple subtle animation, enriching original picture"
    if "metaforica" in scena.get("azione", ""):
        anim_prompt = "evocative animation, conceptual metaphor in motion, simple movements"
    
    vid_res = subagent_comfyui_video(img_res, asset_dir, anim_prompt)
    return vid_res

# ==========================================
# FUNZIONI VIDEO (FFMPEG & YOLO) E AI
# ==========================================
def taglia_silenzi_ffmpeg(input_file, output_file, threshold="-30dB", duration=0.5):
    cmd_detect = ["ffmpeg", "-i", input_file, "-af", f"silencedetect=noise={threshold}:d={duration}", "-f", "null", "-"]
    with open("/tmp/silencedetect.log", "w") as f:
        subprocess.run(cmd_detect, stderr=f)
    silenzi = []
    with open("/tmp/silencedetect.log", "r") as f:
        lines = f.readlines()
        start = None
        for line in lines:
            if "silence_start" in line:
                start = float(line.split("silence_start:")[1].strip())
            elif "silence_end" in line and start is not None:
                end = float(line.split("silence_end:")[1].split("|")[0].strip())
                silenzi.append({"start": start, "end": end})
                start = None
                
    video = VideoFileClip(input_file)
    durata_totale = video.duration
    segmenti_da_tenere = []
    ultimo_tempo = 0.0
    for s in silenzi:
        if s['start'] > ultimo_tempo:
            segmenti_da_tenere.append((ultimo_tempo, s['start']))
        ultimo_tempo = s['end']
    if ultimo_tempo < durata_totale:
        segmenti_da_tenere.append((ultimo_tempo, durata_totale))
        
    fps = video.fps or 25
    durata_sicura = max(0.0, durata_totale - 2.0 / fps)
    segmenti_da_tenere = [(inizio, min(fine, durata_sicura)) for inizio, fine in segmenti_da_tenere if inizio < durata_sicura and min(fine, durata_sicura) - inizio >= 0.1]

    if not segmenti_da_tenere:
        video.close()
        os.rename(input_file, output_file)
        return

    try:
        clip_estratte = [video.subclip(inizio, fine) for inizio, fine in segmenti_da_tenere]
        video_finale = concatenate_videoclips(clip_estratte)
        video_finale.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=video.fps, threads=4)
        video_finale.close()
    except OSError:
        video.close()
        os.rename(input_file, output_file)
        return
    video.close()

def applica_smart_auto_reframe(video_path, output_path, target_w=1080, target_h=1920):
    if models_cache["yolo"] is None:
        models_cache["yolo"] = YOLO(YOLO_PATH)
    model = models_cache["yolo"]
    clip = VideoFileClip(video_path)
    if clip.w == target_w and clip.h == target_h:
        clip.close()
        os.rename(video_path, output_path)
        return
    centers_x = []
    for t in range(0, int(clip.duration), 1):
        frame = clip.get_frame(t)
        results = model(frame, classes=[0], verbose=False)
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0].xyxy[0]
                centers_x.append(float((x1 + x2) / 2))
    avg_x = sum(centers_x) / len(centers_x) if centers_x else clip.w / 2
    new_w = int(clip.h * (9/16))
    new_h = clip.h
    x1 = int(avg_x - (new_w / 2))
    x2 = int(avg_x + (new_w / 2))
    if x1 < 0:
        x1 = 0
        x2 = new_w
    if x2 > clip.w:
        x2 = clip.w
        x1 = clip.w - new_w
    cropped_clip = clip.crop(x1=x1, y1=0, x2=x2, y2=new_h)
    resized_clip = cropped_clip.resize((target_w, target_h))
    resized_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=clip.fps, threads=4)
    clip.close()
    cropped_clip.close()
    resized_clip.close()

def trascrizione_whisper(video_path):
    if models_cache["whisper"] is None:
        models_cache["whisper"] = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=WHISPER_MODEL_DIR)
    segments, info = models_cache["whisper"].transcribe(video_path, word_timestamps=True, language="it")
    testo_completo = ""
    words = []
    for s in segments:
        testo_completo += s.text + " "
        if getattr(s, 'words', None):
            for w in s.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return testo_completo.strip(), words

import string
def find_scene_timestamps(scena, words_list, start_search_idx=0):
    p_in = scena.get("parola_inizio", "").strip().lower().strip(string.punctuation)
    p_fi = scena.get("parola_fine", "").strip().lower().strip(string.punctuation)
    
    start_t = None
    end_t = None
    end_idx = start_search_idx
    
    if p_in and p_fi and words_list:
        for i in range(start_search_idx, len(words_list)):
            clean_w = words_list[i]["word"].lower().strip(string.punctuation)
            if clean_w == p_in:
                start_t = words_list[i]["start"]
                end_idx = i
                break
                
        if start_t is not None:
            for i in range(end_idx, len(words_list)):
                clean_w = words_list[i]["word"].lower().strip(string.punctuation)
                if clean_w == p_fi:
                    end_t = words_list[i]["end"]
                    end_idx = i
                    break
                    
    if start_t is None or end_t is None or end_t <= start_t:
        return None, None, start_search_idx
    return start_t, end_t, end_idx + 1

def chiama_regista_locale(testo_narrativo):
    if models_cache["llm"] is None:
        pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct", model_kwargs={"torch_dtype": torch.float16, "cache_dir": LLM_DIR}, device_map="auto")
        models_cache["llm"] = pipe
    pipe = models_cache["llm"]
    
    system_prompt = """
    Sei il Regista. Ricevi la trascrizione del video.
    Dividi la trascrizione in Scene (blocchi).
    Per ogni scena, decidi se inserire un'immagine o un video per aumentare l'attenzione o spiegare un concetto.
    
    Puoi scegliere tra queste AZIONI:
    - "immagine_engagement" o "video_engagement": colore ricco, surreale, virale. Suscita emozione e serve a tenere l'utente incollato allo schermo.
    - "immagine_comprensione_metaforica" o "video_comprensione_metaforica": traduci un concetto in una similitudine o metafora visiva.
    - "immagine_comprensione_tecnica" o "video_comprensione_tecnica": dividi un concetto in parti semplici e letterali.
    - "null": se la scena non ha bisogno di nulla.
    
    Invece di scrivere il prompt esatto, fornisci una "idea_grezza" in italiano su cosa vorresti far vedere. Passerai poi la palla allo sceneggiatore.
    
    Puoi decidere se lo sfondo dell'asset debba essere rimosso. Imposta "rimuovi_sfondo" a true se l'asset deve essere scontornato, false altrimenti. Specifica il "soggetto" in "soggetto_da_isolare".
    
    Restituisci ESCLUSIVAMENTE e rigorosamente JSON valido:
    [
      {
        "testo_scena": "testo della trascrizione per questo blocco",
        "parola_inizio": "prima parola in italiano di questa scena",
        "parola_fine": "ultima parola in italiano di questa scena",
        "azione": "immagine_engagement" | "video_engagement" | "video_comprensione_metaforica" | "video_comprensione_tecnica" | "null",
        "idea_grezza": "Spiega brevemente in italiano l'idea visiva",
        "rimuovi_sfondo": true | false,
        "soggetto_da_isolare": "soggetto breve in inglese"
      }
    ]
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Trascrizione: {testo_narrativo}"}]
    outputs = pipe(messages, max_new_tokens=1024, do_sample=False)
    content = outputs[0]["generated_text"][-1]["content"].strip()
    if content.startswith("```json"): content = content[7:]
    if content.endswith("```"): content = content[:-3]
    try:
        data = json.loads(content.strip())
        if isinstance(data, dict):
             for key in data.keys():
                  if isinstance(data[key], list):
                       data = data[key]
                       break
        # LLM tenuto in cache per lo sceneggiatore!
        return data
    except Exception:
        return []

def invia_webhook_errore(url, msg):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        requests.post(url, json={"status": "error", "message": msg}, headers=headers)
    except:
        pass

llm_lock = threading.Lock()

def subagent_sceneggiatore(scena, testo_narrativo):
    with llm_lock:
        if models_cache["llm"] is None:
            pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct", model_kwargs={"torch_dtype": torch.float16, "cache_dir": LLM_DIR}, device_map="auto")
            models_cache["llm"] = pipe
        pipe = models_cache["llm"]
        
        azione = scena.get("azione", "")
        idea = scena.get("idea_grezza", "")
        testo = scena.get("testo_scena", "")
        
        system_prompt = f"""
        Sei uno Sceneggiatore-Filologo esperto di prompt design.
        Hai ricevuto un'idea grezza dal Regista per una scena, e l'intera trascrizione per capire il contesto.
        Trascrizione completa: {testo_narrativo}
        
        Azione richiesta dal regista: {azione}
        Testo specifico della scena: {testo}
        Idea grezza del regista: {idea}
        
        Devi trasformare questa idea in un brief visivo super dettagliato IN INGLESE per generare foto e video con AI.
        Rispondi con UNICO E SOLO prompt testuale continuo in inglese, senza titoli, spiegazioni o ritorni a capo superflui.
        
        Il prompt deve definire con estrema precisione:
        1. Soggetto e Azione (Chi/cosa è, cosa fa, espressione, interazioni)
        2. Ambientazione e Contesto (Dove si svolge, atmosfera generale, luci)
        3. Specifiche Tecniche (Stile visivo es. fotografia commerciale, pittura, obiettivo macro, reflex 35mm)
        4. Dinamiche Cinematografiche (SOLO se è un video: come si muove la camera, es. Panoramica, dolly in, camera fissa)
        
        Restituisci SOLO il prompt in inglese.
        """
        messages = [{"role": "system", "content": system_prompt}]
        outputs = pipe(messages, max_new_tokens=512, do_sample=False)
        prompt_enh = outputs[0]["generated_text"][-1]["content"].strip()
        torch.cuda.empty_cache()
        return prompt_enh

def subagent_pulizia(modello_da_liberare=None, file_da_eliminare=None, pulizia_comfyui=False):
    import gc
    write_log(f"Subagent Pulizia: Modello={modello_da_liberare}, File={file_da_eliminare}, Comfy={pulizia_comfyui}")
    
    if modello_da_liberare and modello_da_liberare in models_cache:
        models_cache[modello_da_liberare] = None
        
    if 'torch' in globals():
        torch.cuda.empty_cache()
    gc.collect()

    if file_da_eliminare:
        if isinstance(file_da_eliminare, list):
            for f in file_da_eliminare:
                if os.path.exists(f):
                    try:
                        if os.path.isdir(f): shutil.rmtree(f, ignore_errors=True)
                        else: os.unlink(f)
                    except: pass
        elif os.path.exists(file_da_eliminare):
            try:
                if os.path.isdir(file_da_eliminare): shutil.rmtree(file_da_eliminare, ignore_errors=True)
                else: os.unlink(file_da_eliminare)
            except: pass

    if pulizia_comfyui:
        comfy_out = os.path.join(COMFYUI_DIR, "output")
        comfy_in = os.path.join(COMFYUI_DIR, "input")
        for folder in [comfy_out, comfy_in]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    try:
                        file_path = os.path.join(folder, f)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except: pass

def process_video_workflow(job_input, job_id="default"):
    workspace = f"/tmp/runpod_workspace_{job_id}"
    os.makedirs(workspace, exist_ok=True)
    raw_video = os.path.join(workspace, "raw.mp4")
    cut_video = os.path.join(workspace, "cut.mp4")
    framed_video = os.path.join(workspace, "framed.mp4")
    final_video = os.path.join(workspace, "final.mp4")
    
    r2_download_url = job_input.get("r2DownloadUrl")
    r2_upload_url = job_input.get("r2UploadUrl")
    r2_upload_key = job_input.get("r2UploadKey", "")
    r2_raw_key = job_input.get("r2RawKey", "")
    webhook_url = job_input.get("webhookUrl")
    video_url = job_input.get("videoUrl")
    r2_mode = bool(r2_download_url and r2_upload_url)
    
    if r2_mode:
        r = requests.get(r2_download_url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        with open(raw_video, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    else:
        r = requests.get(video_url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        with open(raw_video, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            
    taglia_silenzi_ffmpeg(raw_video, cut_video)
    subagent_pulizia(file_da_eliminare=raw_video)
    
    applica_smart_auto_reframe(cut_video, framed_video, target_w=1080, target_h=1920)
    subagent_pulizia(modello_da_liberare="yolo", file_da_eliminare=cut_video)
    
    testo_narrativo, word_list = trascrizione_whisper(framed_video)
    subagent_pulizia(modello_da_liberare="whisper")
    
    scene_json = chiama_regista_locale(testo_narrativo)
    scene_elaborate = scene_json.copy()
    
    # Eseguo lo sceneggiatore IN SERIE per tutte le scene per isolare l'uso VRAM dell'LLM
    for scena in scene_elaborate:
        azione = scena.get("azione", "null")
        idea = scena.get("idea_grezza", "")
        if azione != "null" and idea:
            write_log(f"[Sceneggiatore] Analizzo l'idea '{idea}'...")
            scena["prompt"] = subagent_sceneggiatore(scena, testo_narrativo)
            write_log(f"[Sceneggiatore -> Regista] Prompt potenziato: {scena['prompt']}")
            
    # ORA LIBERO COMPLETAMENTE LLM PRIMA DI AVVIARE I THREAD DI COMFYUI
    subagent_pulizia(modello_da_liberare="llm")
    
    # Esecuzione Subagents Generativi in parallelo
    asset_dir = os.path.join(workspace, "assets")
    os.makedirs(asset_dir, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures_map = {}
        for idx, scena in enumerate(scene_elaborate):
            azione = scena.get("azione", "null")
            if azione != "null" and scena.get("prompt"):
                def task_generativo(s, d):
                    if s["azione"].startswith("video_"):
                        return subagent_video_chain(s, d)
                    else:
                        return subagent_comfyui_worker(s, d)
                        
                futures_map[executor.submit(task_generativo, scena, asset_dir)] = idx
            
        for future in concurrent.futures.as_completed(futures_map):
            idx = futures_map[future]
            res = future.result()
            if isinstance(res, dict) and "error" in res:
                err_msg = f"Avaria Subagent Generativo! Tentativi esauriti: {res['error']}"
                invia_webhook_errore(webhook_url, err_msg)
                print(err_msg + " -> Vado in standby morbido.")
                shutil.rmtree(workspace, ignore_errors=True)
                return {"status": "error_handled_standby", "message": "Avaria gestita con standby per evitare riavvii di RunPod."}
            else:
                scena = scene_elaborate[idx]
                azione = scena.get("azione", "")
                
                # Se era un'immagine pura e serviva rimozione sfondo
                if azione.startswith("immagine_") and scena.get("rimuovi_sfondo"):
                    soggetto = scena.get("soggetto_da_isolare", "character")
                    print(f"Rimozione sfondo richiesta per l'asset {idx} sul soggetto '{soggetto}'...")
                    rmbg_res = subagent_comfyui_rmbg(res, asset_dir, subject_prompt=soggetto)
                    if isinstance(rmbg_res, dict) and "error" in rmbg_res:
                        err_msg = f"Avaria Subagent RMBG! Tentativi esauriti: {rmbg_res['error']}"
                        invia_webhook_errore(webhook_url, err_msg)
                        print(err_msg + " -> Vado in standby morbido.")
                        shutil.rmtree(workspace, ignore_errors=True)
                        return {"status": "error_handled_standby", "message": "Avaria RMBG gestita con standby."}
                    else:
                        scene_elaborate[idx]["asset_path"] = rmbg_res
                else:
                    scene_elaborate[idx]["asset_path"] = res

    # Compositing Finale
    video_base = VideoFileClip(framed_video)
    fallback_durata = video_base.duration / max(1, len(scene_elaborate))
    elementi_da_sovrapporre = [video_base]
    
    start_search_idx = 0
    for idx, scena in enumerate(scene_elaborate):
        start_time, end_time, next_idx = find_scene_timestamps(scena, word_list, start_search_idx)
        if start_time is None or end_time is None:
            start_time = idx * fallback_durata
            durata_scena = fallback_durata
        else:
            durata_scena = end_time - start_time
            start_search_idx = next_idx
            
        if scena.get("asset_path") and os.path.exists(scena["asset_path"]):
            azione = scena.get("azione", "")
            is_video_file = scena["asset_path"].endswith(".mp4") or scena["asset_path"].endswith(".webm")
            
            try:
                if is_video_file or azione.startswith("video_"):
                    clip = VideoFileClip(scena["asset_path"])
                    # Se il video generato è più corto della durata richiesta dalla scena, lo mandiamo in loop
                    if clip.duration < durata_scena:
                        clip = clip.fx(vfx.loop, duration=durata_scena)
                    else:
                        clip = clip.set_duration(durata_scena)
                        
                    clip = (clip.set_start(start_time)
                            .resize(width=1000)
                            .set_position(("center", "center"))
                            .crossfadein(0.5)
                            .crossfadeout(0.5))
                else:
                    clip = (ImageClip(scena["asset_path"])
                            .set_duration(durata_scena)
                            .set_start(start_time)
                            .resize(width=1000)
                            .set_position(("center", "center"))
                            .crossfadein(0.5)
                            .crossfadeout(0.5))
                elementi_da_sovrapporre.append(clip)
            except Exception:
                pass # Fallback in caso di corruzione video
            
    clip_composta = CompositeVideoClip(elementi_da_sovrapporre, size=(1080, 1920))
    video_senza_testo = os.path.join(workspace, "composited_no_subs.mp4")
    clip_composta.write_videofile(video_senza_testo, codec="libx264", audio_codec="aac", fps=video_base.fps, threads=4)
    video_base.close()
    clip_composta.close()

    try:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        subprocess.run(["pycaps", "render", "--input", video_senza_testo, "--template", "hype", "--output", final_video], check=True)
    except Exception as e:
        shutil.copy2(video_senza_testo, final_video)
    
    if r2_mode:
        file_size = os.path.getsize(final_video)
        with open(final_video, 'rb') as f:
            requests.put(r2_upload_url, data=f, headers={"Content-Type": "video/mp4", "Content-Length": str(file_size)})
        requests.post(webhook_url, json={"status": "success", "s3KeyFinal": r2_upload_key, "s3KeyRaw": r2_raw_key}, headers={"User-Agent": "Mozilla/5.0"})
    else:
        with open(final_video, 'rb') as f:
            requests.post(webhook_url, files={'file': f}, headers={"User-Agent": "Mozilla/5.0"})
            
    # Pulizia globale finale: svuota VRAM, elimina workspace e pulisce cache ComfyUI
    subagent_pulizia(file_da_eliminare=workspace, pulizia_comfyui=True)
                    
    return {"status": "success", "message": "Reel generato con successo!", "r2_mode": r2_mode}

def handler(job):
    write_log(f"handler invoked with job: {job.get('id', 'default')}")
    try:
        load_ml_libraries()
        return process_video_workflow(job['input'], job.get("id", "default"))
    except Exception as e:
        stack = traceback.format_exc()
        messaggio_errore = f"=== INIZIO LOG INTEGRALE ===\n{stack}\n=== FINE LOG INTEGRALE ==="
        webhook_url = job.get("input", {}).get("webhookUrl")
        invia_webhook_errore(webhook_url, messaggio_errore)
        # INVECE di far fallire il worker restituendo {"error": str(e)} o esplodendo,
        # Ritorniamo uno status OK dal punto di vista di RunPod, e andiamo in "standby logico".
        # Questo blocca il riavvio in loop.
        return {"status": "error_handled_standby", "message": str(e), "stacktrace": stack}

def safe_handler(job):
    write_log(f"safe_handler invocato per job {job.get('id', 'unknown')}")
    if not worker_operativo:
        write_log("Rifiuto job per worker_operativo == False")
        invia_webhook_errore(job.get("input", {}).get("webhookUrl"), f"Worker in avaria al boot:\n{messaggio_avaria}")
        return {"status": "error_handled_standby", "message": "Worker in avaria al boot", "dettagli": messaggio_avaria}
    return handler(job)

if __name__ == "__main__":
    try:
        if start_comfyui():
            worker_operativo = True
            write_log("Init success, worker_operativo = True")
        else:
            worker_operativo = False
            write_log(f"Init failed, worker_operativo = False. Dettagli: {messaggio_avaria}")
    except Exception as e:
        messaggio_avaria = f"Avaria imprevista avvio Worker: {str(e)}\n{traceback.format_exc()}"
        write_log(f"Exception during init: {messaggio_avaria}")
        worker_operativo = False

    write_log("calling runpod.serverless.start")
    runpod.serverless.start({"handler": safe_handler})
    write_log("runpod.serverless.start returned (worker exiting)")

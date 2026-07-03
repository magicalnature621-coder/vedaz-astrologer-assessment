# Hosting a Fine-Tuned Qwen Model on a VPS with vLLM

This covers taking your fine-tuned Qwen2.5 / Qwen3 astrologer model and serving it in production on a rented VPS, with an OpenAI-compatible API endpoint.

## 1. Choose the VPS

vLLM needs a GPU for real throughput (CPU inference works but is very slow for a chat product).

| Model size | Min GPU VRAM | Example GPU | Example provider |
|---|---|---|---|
| Qwen2.5/3 - 7B (LoRA merged, fp16) | 16-20 GB | RTX 4090, A10, L4 | RunPod, Lambda, Vast.ai, Hetzner GPU |
| Qwen2.5/3 - 7B (AWQ/GPTQ 4-bit) | 8-10 GB | RTX 3090/4090 | same as above |
| Qwen2.5/3 - 14B (4-bit) | 16-24 GB | A6000, 2x4090 | RunPod, Paperspace |

Also get: 50-100 GB SSD (model weights + vLLM cache), 16+ GB system RAM, Ubuntu 22.04 LTS image.

## 2. Base Server Setup

```bash
# SSH in, then update
sudo apt update && sudo apt upgrade -y

# NVIDIA driver (skip if the provider's image ships one pre-installed)
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify GPU is visible
nvidia-smi
```

## 3. Python Environment

```bash
sudo apt install -y python3.11 python3.11-venv python3-pip git

python3.11 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

pip install --upgrade pip
pip install vllm          # pulls the matching torch/CUDA build automatically
```

## 4. Get the Model onto the Box

If you fine-tuned with LoRA, merge the adapter into the base weights first (see the fine-tuning write-up, Step 5) so vLLM gets a single, standard HF-format model folder. Then either:

```bash
# Option A: push merged model to Hugging Face Hub, then on the VPS:
huggingface-cli login
huggingface-cli download your-username/vedaz-qwen2.5-7b-astro --local-dir /opt/models/vedaz-astro

# Option B: scp the merged folder directly from your training machine
scp -r ./merged_model user@your-vps-ip:/opt/models/vedaz-astro
```

## 5. Launch the vLLM OpenAI-Compatible Server

```bash
vllm serve /opt/models/vedaz-astro \
  --served-model-name vedaz-astro-v1 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key YOUR_SECRET_API_KEY \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype auto
```

Quick test:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_SECRET_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astro-v1",
    "messages": [{"role": "user", "content": "Mera career kaisa rahega is saal?"}]
  }'
```

If you used 4-bit quantization (AWQ/GPTQ) to fit a smaller GPU, add `--quantization awq` (or `gptq`) and point to the quantized weights instead.

## 6. Keep It Running: systemd Service

```ini
# /etc/systemd/system/vllm-astro.service
[Unit]
Description=vLLM Astrologer Model Server
After=network.target

[Service]
User=youruser
WorkingDirectory=/opt/models
ExecStart=/home/youruser/vllm-env/bin/vllm serve /opt/models/vedaz-astro \
  --served-model-name vedaz-astro-v1 \
  --host 0.0.0.0 --port 8000 \
  --api-key YOUR_SECRET_API_KEY \
  --max-model-len 8192 --gpu-memory-utilization 0.90
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-astro
sudo systemctl status vllm-astro
```

## 7. Put Nginx + SSL in Front of It

```nginx
# /etc/nginx/sites-available/vllm-astro
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;   # allow for longer generations
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/vllm-astro /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

## 8. Security & Ops Checklist

- **Firewall**: `ufw allow 22,80,443/tcp`, block 8000 from the public internet (only nginx should reach it — bind vLLM to `127.0.0.1` instead of `0.0.0.0` once nginx is in place).
- **API key**: rotate the `--api-key` periodically; never commit it to git.
- **Rate limiting**: add `limit_req_zone` in nginx to protect against abuse/cost spikes.
- **Monitoring**: `nvidia-smi -l 5` for GPU load; vLLM exposes Prometheus metrics at `/metrics` — wire into Grafana if you want dashboards.
- **Logs**: `journalctl -u vllm-astro -f` for live logs; ship to a log aggregator for the astrology use case since you'll want to audit safety-sensitive conversations (self-harm, health, legal topics).
- **Autoscaling / multi-GPU**: for higher traffic, run vLLM with `--tensor-parallel-size N` across multiple GPUs, or put multiple single-GPU instances behind a load balancer.
- **Cost control**: idle GPU VPS is expensive — if traffic is bursty, consider a serverless GPU provider (RunPod Serverless, Modal) instead of an always-on VPS.

## 9. Updating the Model Later

When you retrain with new conversation data: merge the new LoRA adapter, upload the new folder as a new versioned path (e.g. `/opt/models/vedaz-astro-v2`), point a new systemd service/port at it, smoke-test it, then flip nginx's `proxy_pass` over — avoids downtime during swaps.

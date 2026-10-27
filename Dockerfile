# CPHD ship detection on GPUs: crop builder (CuPy) and detector training (PyTorch) in one image
#   docker build -t cphd-gpu .
#
# Crops on several GPUs (one process per GPU, scenes shared out):
#   docker run -d --name crops --gpus all --user $(id -u):$(id -g) \
#       -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
#       -v /home/tabtahi/SATLOCK/dataset/cphd:/cphd:ro -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/out \
#       cphd-gpu build_crops.py --ann-root /ann --cphd-root /cphd --out /out --gpus 0,1,2,3
#
# Training on 4 GPUs:
#   docker run -d --name train --gpus all --ipc=host --user $(id -u):$(id -g) -e HOME=/tmp \
#       -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
#       -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops:ro -v /home/tabtahi/SATLOCK/runs:/runs \
#       --entrypoint torchrun cphd-gpu --nproc_per_node=4 /app/train_detector.py train \
#       --crops /crops --ann-root /ann --out /runs/r1
#   raw phase-history model: same line with /app/train_raw.py in place of /app/train_detector.py
#
# Box adjust web page (then on the laptop: ssh -L 8765:localhost:8765 tabtahi@geohub3, open http://localhost:8765):
#   mkdir -p /home/tabtahi/SATLOCK/dataset/annotations_adjusted
#   docker run -d --name boxadj --gpus '"device=0"' --user $(id -u):$(id -g) -p 127.0.0.1:8765:8765 \
#       -v /data3/sandbox/jhu/satlock/annotation_tool/test_aligned:/ann:ro \
#       -v /home/tabtahi/SATLOCK/dataset/annotations_adjusted:/ann_adj \
#       -v /home/tabtahi/SATLOCK/dataset/cphd_crops:/crops \
#       cphd-gpu /app/box_adjust_server.py --crops /crops --ann-root /ann --ann-out /ann_adj --host 0.0.0.0 --device cuda:0
# Python 3.12 (sarkit 1.12 needs >= 3.12). The PyTorch CUDA 12.6 wheels bring their own CUDA libraries; the host only
# needs an NVIDIA driver (CUDA 12.x, driver >= 525) and the NVIDIA container toolkit (docker run --gpus all).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TORCH_HOME=/app/torch

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
 && pip install sarkit==1.12.0 pyproj lxml scipy "cupy-cuda12x[ctk]" \
 && python -c "import torchvision; torchvision.models.resnet50(weights='DEFAULT'); torchvision.models.resnet18(weights='DEFAULT')" \
 && chmod -R a+rX /app/torch

RUN pip install tensorboard

# writable caches for any --user (CuPy kernel cache, matplotlib); after the pip layer so it stays cached
ENV HOME=/tmp CUPY_CACHE_DIR=/tmp/cupy_cache MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY build_crops.py train_detector.py train_raw.py train_viz.py fourier_net.py png_output.py box_adjust_server.py heat_regions.py /app/
ENTRYPOINT ["python"]
CMD ["/app/train_detector.py", "--help"]

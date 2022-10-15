cd ..
git clone https://github.com/NVlabs/nvdiffrec.git

pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install \
  --global-option="--no-networks" \
  git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch

imageio_download_bin freeimage


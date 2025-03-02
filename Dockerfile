# Use the official Conda base image with Python 3.7
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /DW-GAN

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  

RUN conda install -c conda-forge timm==1.0.15 -y
RUN pip install opencv-python scikit-image tensorboardx==2.6.2.2 yacs "numpy<2"

COPY . .

CMD ["bash", "-c", "python test.py --test_dir '/DW-GAN/input_data/'"]
# CMD ["python", "test.py", "--test_dir", "/DW-GAN/input_data/"] ##try this instead

## TO RUN IMAGE
# hare run --rm --gpus device=3 \
#     --mount type=bind,source=/homes/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
#     --mount type=bind,source=/homes/ceh94/DW-GAN/output_result,target=/DW-GAN/output_result \
#     --mount type=bind,source=/homes/ceh94/DW-GAN/pytorch_mssim,target=/DW-GAN/pytorch_mssim \
#     --mount type=bind,source=/homes/ceh94/DW-GAN/input_data,target=/DW-GAN/input_data \
#     ceh94/dwimg

hare run --rm --gpus device=3 \
--mount type=bind,source=/homes/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
--mount type=bind,source=/homes/ceh94/DW-GAN/output_result,target=/DW-GAN/output_result \
--mount type=bind,source=/homes/ceh94/DW-GAN/input_data,target=/DW-GAN/input_data \
ceh94/dwimg

hare run --rm --gpus '"device=1,4"' \
--mount type=bind,source=/homes/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
--mount type=bind,source=/homes/ceh94/DW-GAN/output_result,target=/DW-GAN/output_result \
--mount type=bind,source=/homes/ceh94/DW-GAN/input_data,target=/DW-GAN/input_data \
ceh94/dwimg
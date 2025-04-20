# Use the official Conda base image with Python 3.7
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /DW-GAN

# required for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  

RUN conda install -c conda-forge timm==1.0.15 -y
RUN pip install opencv-python scikit-image tensorboardx==2.6.2.2 yacs "numpy<2"

COPY . .

ENTRYPOINT ["python"]
# ENTRYPOINT ["bash", "-c"]

# CMD ["bash", "-c", "python test.py --test_dir '/DW-GAN/input_data/'"]
CMD ["python", "test.py", "--test_dir", "/DW-GAN/input_data/", "--datasets", "Dense_Haze", "DNH_1600x1200_test", "SMOKE_1600x1200_test", "--weights", "dehaze.pkl"]


# TO BUILD IMAGE
# hare build -t ceh94/dw .

# --- TO TEST IMAGES ---
# hare run --rm --gpus '"device=2,3"' \
# --mount type=bind,source=/homes/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
# --mount type=bind,source=/homes/ceh94/DW-GAN/output_result,target=/DW-GAN/output_result \
# --mount type=bind,source=/homes/ceh94/DW-GAN/input_data,target=/DW-GAN/input_data \
# ceh94/dw \
# test.py --test_dir /DW-GAN/input_data/ --datasets "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test" --weights "2025-03-13_13-49-42_NHNH2RB10_epoch06000.pkl"

# hare run --rm --gpus '"device=2,3"' --shm-size=128g \
# --mount type=bind,source=/mnt/fast1/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
# --mount type=bind,source=/mnt/fast1/ceh94/DW-GAN/output_result,target=/DW-GAN/output_result \
# --mount type=bind,source=/mnt/fast1/ceh94/DW-GAN/input_data,target=/DW-GAN/input_data \
# ceh94/dw \
# test.py --test_dir /DW-GAN/input_data/ --datasets "Dense_Haze" "DNH_1600x1200_test" "SMOKE_1600x1200_test" --weights "2025-03-13_13-49-42_NHNH2RB10_epoch06000.pkl"

# --- TO TRAIN MODEL ---
# hare run --rm --gpus '"device=5"' --shm-size=128g \
# --mount type=bind,source=/homes/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
# --mount type=bind,source=/homes/ceh94/DW-GAN/input_training_data,target=/DW-GAN/input_training_data \
# --mount type=bind,source=/homes/ceh94/DW-GAN/check_points,target=/DW-GAN/check_points \
# ceh94/dw \
# train.py --datasets NHNH2RBm10

# hare run --rm --gpus '"device=0,1"' --shm-size=128g \
# --mount type=bind,source=/mnt/faster0/ceh94/DW-GAN/weights,target=/DW-GAN/weights \
# --mount type=bind,source=/mnt/faster0/ceh94/DW-GAN/input_training_data,target=/DW-GAN/input_training_data \
# --mount type=bind,source=/mnt/faster0/ceh94/DW-GAN/check_points,target=/DW-GAN/check_points \
# ceh94/dw \
# train.py --datasets NHNH2RBm10
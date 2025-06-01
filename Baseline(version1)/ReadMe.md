# 更新套件清單
apt-get update

# 安裝 build-essential（裡面包含 gcc、g++、cc1plus...）
apt-get install -y build-essential

nvcc test.cu -o test

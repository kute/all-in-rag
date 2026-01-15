
```shell
conda create --name all-in-rag python=3.12.7 -y
conda activate all-in-rag
cd code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 然后注释掉 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```


```
#!/bin/bash

# 检查并安装必要的包
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-full

# 创建虚拟环境
echo "创建虚拟环境..."
python3.12 -m venv venv_intel_312

# 激活虚拟环境
echo "激活虚拟环境..."
source venv_intel_312/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip setuptools wheel

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

echo "✅ 安装完成！"
echo "使用以下命令激活环境："
echo "source venv_intel_311/bin/activate"
```
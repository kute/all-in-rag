
```shell
conda create --name all-in-rag python=3.12.7 -y
conda activate all-in-rag
cd code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 然后注释掉 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```
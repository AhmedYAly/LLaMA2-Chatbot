import os
import subprocess
import sys
from urllib.request import urlretrieve
from pathlib import Path

from common import model_path, model_url, model_name


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install('torch')
install('ctransformers')
install('langchain')
install('accelerate')
install('bitsandbytes')
install('transformers')
install('sentence_transformers')
install('faiss_cpu')
install('Flask')

filename = os.path.join(model_path, model_name)
if not os.path.exists(filename):
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)
    print("Downloading model...")
    urlretrieve(model_url, filename)
else:
    print("Model already present.")

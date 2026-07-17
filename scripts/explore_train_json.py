import json
import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

with open(ROOT_DIR / 'data/TAO_Amodal_Dataset/ASPIRe_labels/train.json', 'r') as f:
  data = json.load(f)

# data['data'] é a lista principal no seu exemplo. Vamos imprimir apenas o primeiro vídeo/imagem:
print(json.dumps(data['data'][0], indent=4))
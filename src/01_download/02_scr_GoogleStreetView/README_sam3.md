# Checklist para correr 03e_extraer_segmentacion_sam3.py en la sala con GPU

Todo lo que se podía adelantar sin GPU ya está hecho (ver estado abajo). Esta
guía es solo lo que falta hacer una vez estés frente a la máquina con GPU
CUDA — pensada para copiar y pegar comandos, sin tener que pensar nada ahí.

## Estado (hecho de antemano, en el Mac)

- [x] Token de Hugging Face generado, con acceso aprobado a `facebook/sam3`.
- [x] Token guardado en `.hf_token` (junto a este archivo, NO está en git).
- [x] Checkpoint `sam3.pt` (3.45 GB) + `config.json` ya descargados y
      cacheados en `~/.cache/huggingface/hub/models--facebook--sam3/` en
      este Mac.
- [x] `03e_extraer_segmentacion_sam3.py` listo — lee el token automáticamente
      desde `.hf_token`, soporta `CONFIG["device"] = "cuda"` o `"cpu"`.
- [ ] **No se pudo correr un piloto end-to-end en este Mac**: es Intel
      (x86_64) y PyTorch ya no publica wheels de macOS para esa arquitectura.
      La primera corrida real será en la sala.

## 0. Qué llevar / tener a mano

- [ ] Este repo (`tesis_vulnerabilidad`) — clonarlo en la máquina de la sala
      si no está ya ahí (`git clone <url-del-repo>`).
- [ ] El archivo `.hf_token` — **NO viaja con `git clone`** (está en
      `.gitignore` a propósito, ver nota de seguridad al final). Cópialo a
      mano (USB, `scp`, AirDrop) a la misma ruta relativa:
      `src/01_download/02_scr_GoogleStreetView/.hf_token` dentro del repo
      clonado en la sala.
- [ ] Las imágenes descargadas (`gsv/fotos/`) y `gsv/registro_descargas.csv`
      / `gsv/inventario_panos.csv` — si la sala no tiene ya una copia del
      dataset GSV, hay que llevarlos también (son el input real del script,
      independiente de SAM3).
- [ ] (Opcional, ahorra 3.45 GB de descarga) La caché de Hugging Face de
      este Mac: carpeta completa
      `~/.cache/huggingface/hub/models--facebook--sam3/` → copiarla a la
      misma ruta relativa (`~/.cache/huggingface/hub/`) del usuario en la
      máquina de la sala. Es portable entre sistemas (los archivos se
      identifican por hash, no importa el SO). Si no la llevas, no pasa
      nada — se descarga sola la primera vez que corra el script.

## 1. Verificar GPU disponible

```bash
nvidia-smi
```

Debe mostrar al menos una GPU con CUDA 12.6+. Si esto falla, nada de lo
demás va a funcionar — resolver esto primero.

## 2. Clonar e instalar SAM3 (en un entorno virtual APARTE del resto del pipeline)

SAM3 pide PyTorch 2.7+ y Python 3.12+, más nuevo que el `torch>=2.0` que usan
los scripts 03/03b — por eso un venv propio, para no romper esos otros
pipelines si ya están instalados en la máquina.

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3

python3.12 -m venv .venv
source .venv/bin/activate

# PyTorch con soporte CUDA (ajustar cu128 a la versión de CUDA real de la sala)
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e .

# Opcional, para inferencia más rápida:
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
```

## 3. Confirmar que el token se reconoce

Desde el mismo entorno (con `huggingface_hub` ya instalado por la
dependencia de `sam3`):

```bash
python -c "
import os
from pathlib import Path
token_path = Path('<RUTA_AL_REPO_TESIS>/src/01_download/02_scr_GoogleStreetView/.hf_token')
os.environ['HF_TOKEN'] = token_path.read_text().strip()
from huggingface_hub import whoami
print(whoami())
"
```

Si imprime tu usuario de Hugging Face sin error, el token quedó bien.

## 4. Instalar las dependencias del resto del pipeline GSV

En el **venv del proyecto tesis** (no el de sam3), si no lo tienes ya
instalado:

```bash
cd <RUTA_AL_REPO_TESIS>/src/01_download/02_scr_GoogleStreetView
pip install -r requirements.txt
```

(`pandas`, `pyarrow`, `Pillow`, `tqdm` — el script 03e los necesita además
de `sam3`; si usaste el venv de sam3 para todo, instálalos ahí en vez de
crear un tercer entorno.)

## 5. Correr el script

```bash
cd <RUTA_AL_REPO_TESIS>/src/01_download/02_scr_GoogleStreetView
python 03e_extraer_segmentacion_sam3.py
```

`CONFIG["device"]` ya está en `"cuda"` por defecto — no hay que tocar nada
salvo que la sala use varias GPUs y quieras fijar una específica
(`CUDA_VISIBLE_DEVICES=0 python 03e_...py`).

## 6. Qué revisar si algo falla

- **`ModuleNotFoundError: sam3`** → el venv activo no es el que tiene
  `pip install -e .` corrido. Revisar `which python`.
- **Error de autenticación / 401 al descargar el checkpoint** → el
  `.hf_token` no llegó a la ruta esperada, o quedó vacío al copiarlo.
  Verificar con el comando del paso 3.
- **`CUDA out of memory`** → bajar `CONFIG["batch_size"]` no aplica aquí
  (SAM3 procesa una imagen a la vez, ver Módulo 4 del script) — el problema
  sería más bien memoria insuficiente para el modelo mismo; confirmar con
  `nvidia-smi` cuánta VRAM libre hay antes de correr.
- **Tarda mucho / parece colgado** → normal en las primeras imágenes
  (compilación/carga inicial del backbone); si sigue lento después de la
  imagen 10-20, revisar que `CONFIG["device"]` realmente sea `"cuda"` y no
  se haya quedado en `"cpu"` por error.

---

**Nota de seguridad**: `.hf_token` está en `.gitignore` a propósito — es una
credencial de tu cuenta de Hugging Face. No lo pegues en ningún commit, PR,
chat, ni lo subas a ningún lado fuera de copiarlo directamente entre tus
propias máquinas.

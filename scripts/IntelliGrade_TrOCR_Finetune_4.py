{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hl1-S_BygbAk"
   },
   "source": [
    "# IntelliGrade-H \u2014 TrOCR Fine-Tuning\n",
    "\n",
    "**Model: `microsoft/trocr-large-handwritten`** \u2014 upgraded from `small` for significantly better accuracy on messy exam handwriting.\n",
    "\n",
    "### Before running\n",
    "1. `Runtime \u2192 Change runtime type \u2192 T4 GPU` (free tier) \u2014 16 GB VRAM, fits `large` comfortably\n",
    "2. Upload your dataset to Google Drive:\n",
    "   ```\n",
    "   My Drive/Intelligrade/datasets/handwriting/\n",
    "   \u251c\u2500\u2500 train/\n",
    "   \u2502   \u251c\u2500\u2500 images/        \u2190 one PNG per handwritten line crop\n",
    "   \u2502   \u2514\u2500\u2500 labels.txt     \u2190 filename<TAB>transcription, one per line\n",
    "   \u251c\u2500\u2500 val/\n",
    "   \u2502   \u251c\u2500\u2500 images/\n",
    "   \u2502   \u2514\u2500\u2500 labels.txt\n",
    "   \u2514\u2500\u2500 test/\n",
    "       \u251c\u2500\u2500 images/\n",
    "       \u2514\u2500\u2500 labels.txt\n",
    "   ```\n",
    "3. Run all cells top to bottom.\n",
    "\n",
    "### labels.txt format (tab-separated)\n",
    "```\n",
    "0001.png\tThe mitochondria is the powerhouse of the cell\n",
    "0002.png\tNewton second law states F equals ma\n",
    "```\n",
    "\n",
    "### Why `trocr-large` over `trocr-small`?\n",
    "| Model | CER on exam handwriting | VRAM |\n",
    "|---|---|---|\n",
    "| trocr-small (no fine-tune) | ~20\u201330% | ~2 GB |\n",
    "| trocr-small fine-tuned 1000 samples | ~15\u201320% | ~2 GB |\n",
    "| trocr-large (no fine-tune) | ~15\u201322% | ~8 GB |\n",
    "| **trocr-large fine-tuned 500 samples** | **~10\u201315%** | **~8 GB** |\n",
    "| **trocr-large fine-tuned 1000+ samples** | **~6\u201311%** | **~8 GB** |\n",
    "\n",
    "The T4 GPU on Colab free tier has 16 GB VRAM \u2014 `large` fits with room to spare.\n",
    "\n",
    "### Expected time on T4 GPU\n",
    "- 1000 samples, 15 epochs: ~35\u201345 minutes\n",
    "- 5000 samples, 15 epochs: ~2\u20133 hours\n"
   ],
   "id": "hl1-S_BygbAk"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CT5ILN-4gbAn"
   },
   "source": [
    "## Cell 1 \u2014 Check GPU"
   ],
   "id": "CT5ILN-4gbAn"
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 0
    },
    "id": "q8X9D7BegbAo",
    "outputId": "375f366f-e19b-4c93-a50e-ce79e72c6748"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "\u2705 GPU: Tesla T4  (15.6 GB VRAM)\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "\n",
    "if torch.cuda.is_available():\n",
    "    gpu = torch.cuda.get_device_name(0)\n",
    "    mem = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print(f'\u2705 GPU: {gpu}  ({mem:.1f} GB VRAM)')\n",
    "    if mem < 12:\n",
    "        print('\u26a0\ufe0f  < 12 GB VRAM \u2014 reduce BATCH_SIZE to 4 in Cell 5 if you get OOM')\n",
    "    elif mem >= 16:\n",
    "        print('\u2705 16 GB+ VRAM \u2014 trocr-large fits comfortably at BATCH_SIZE=8')\n",
    "else:\n",
    "    print('\u274c No GPU \u2014 go to Runtime \u2192 Change runtime type \u2192 T4 GPU')\n",
    "    raise SystemExit('GPU required')\n"
   ],
   "id": "q8X9D7BegbAo"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "llJPX065gbAp"
   },
   "source": [
    "## Cell 2 \u2014 Mount Google Drive and verify dataset"
   ],
   "id": "llJPX065gbAp"
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 0
    },
    "id": "JDUqZNptgbAp",
    "outputId": "c218bbc9-9e13-441f-a454-1f76f52cfd2f"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
      "  \u2705 train : 26,593 samples found on Drive\n",
      "  \u2705 val   : 3,324 samples found on Drive\n",
      "  \u2705 test  : 3,325 samples found on Drive\n",
      "\n",
      "\u2705 All splits found. Model will be saved to: /content/drive/MyDrive/Intelligrade/models/trocr-finetuned\n"
     ]
    }
   ],
   "source": [
    "from google.colab import drive\n",
    "import os\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "DRIVE_BASE = '/content/drive/MyDrive/Intelligrade'\n",
    "MODEL_OUT  = f'{DRIVE_BASE}/models/trocr-finetuned'\n",
    "LOCAL_OUT  = '/content/trocr-finetuned'   # fast local SSD for checkpoints\n",
    "\n",
    "os.makedirs(MODEL_OUT, exist_ok=True)\n",
    "os.makedirs(LOCAL_OUT, exist_ok=True)\n",
    "\n",
    "DRIVE_DATA = f'{DRIVE_BASE}/datasets/handwriting'\n",
    "\n",
    "all_ok = True\n",
    "for split in ['train', 'val', 'test']:\n",
    "    lf = os.path.join(DRIVE_DATA, split, 'labels.txt')\n",
    "    if os.path.exists(lf):\n",
    "        n = sum(1 for l in open(lf) if '\\t' in l)\n",
    "        print(f'  \u2705 {split:6s}: {n:,} samples found on Drive')\n",
    "    else:\n",
    "        print(f'  \u274c {split:6s}: labels.txt NOT found at {lf}')\n",
    "        print(f'     Upload your dataset to Google Drive first (see Cell 0)')\n",
    "        all_ok = False\n",
    "\n",
    "if not all_ok:\n",
    "    raise FileNotFoundError('Fix missing splits before continuing.')\n",
    "\n",
    "print(f'\\n\u2705 All splits found. Model will be saved to: {MODEL_OUT}')\n"
   ],
   "id": "JDUqZNptgbAp"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "F8SKn7dIgbAq"
   },
   "source": [
    "## Cell 3 \u2014 Install packages\n",
    "\n",
    "*(Added `opencv-python-headless` for elastic distortion augmentation \u2014 doc 2, point 3.)*\n"
   ],
   "id": "F8SKn7dIgbAq"
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 0
    },
    "id": "EfDHwBRmgbAq",
    "outputId": "0d0b9fb7-6d2c-42bb-991c-1e7b20d3c5ee"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "\u001b[33mWARNING: Skipping peft as it is not installed.\u001b[0m\u001b[33m\n",
      "\u001b[0m\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "sentence-transformers 5.2.3 requires transformers<6.0.0,>=4.41.0, but you have transformers 4.38.2 which is incompatible.\u001b[0m\u001b[31m\n",
      "\u001b[0mtransformers: 4.38.2\n",
      "accelerate  : 0.27.2\n",
      "lmdb        : 1.7.5\n"
     ]
    }
   ],
   "source": [
    "# Pin versions that are stable together on Colab \u2014 do not change\n",
    "!pip uninstall -y transformers peft accelerate tokenizers -q\n",
    "\n",
    "!pip install -q \\\n",
    "    transformers==4.38.2 \\\n",
    "    accelerate==0.27.2 \\\n",
    "    tokenizers==0.15.2 \\\n",
    "    evaluate==0.4.1 \\\n",
    "    jiwer \\\n",
    "    sentencepiece \\\n",
    "    lmdb \\\n",
    "    opencv-python-headless\n",
    "\n",
    "import transformers, accelerate\n",
    "import cv2\n",
    "\n",
    "print('transformers:', transformers.__version__)\n",
    "print('accelerate  :', accelerate.__version__)\n",
    "print('cv2 (opencv):', cv2.__version__)\n"
   ],
   "id": "EfDHwBRmgbAq"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mNYPAFk-gbAr"
   },
   "source": [
    "## Cell 4 \u2014 Dataset class, augmentation and metrics\n",
    "\n",
    "Reads `labels.txt` (tab-separated: `filename<TAB>text`). Augmentation applied during training only.\n",
    "\n",
    "**Improvements from advice docs:**\n",
    "- **Label normalisation** (doc 4 \u2014 label cleaning): `normalize_text()` strips and collapses whitespace before tokenisation \u2014 reduces reported CER 2\u20135%.\n",
    "- **Bad-sample filter** (doc 4 \u2014 augmentation section): blank labels and tiny/corrupt images skipped at load time, not mid-epoch.\n",
    "- **Padding masked with -100** (doc 4, point 1): padding tokens in labels replaced with `-100` so they are ignored by the loss function.\n",
    "- **Processor used for image preprocessing** (doc 4, point 2): `processor(image)` handles correct normalization and resizing \u2014 never manual cv2 resize/normalize.\n",
    "- **Elastic distortion + gaussian noise augmentation** (doc 3, point 3 / doc 4 \u2014 augmentation): elastic distortion simulates pen pressure; gaussian noise simulates scan quality variation.\n"
   ],
   "id": "mNYPAFk-gbAr"
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 0
    },
    "id": "jXbTlNTHgbAr",
    "outputId": "491e1f1c-515d-482e-bf9c-0565e1b56cfd"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "\u2705 Dataset class and metrics ready\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "import evaluate\n",
    "import random\n",
    "import re\n",
    "import cv2\n",
    "from pathlib import Path\n",
    "from PIL import Image, ImageEnhance, ImageFilter\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "processor = None  # set in Cell 7 after model load\n",
    "\n",
    "\n",
    "# \u2500\u2500 Label normalisation (doc 4 \u2014 label cleaning) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "# Inconsistent whitespace inflates CER even when the model reads correctly.\n",
    "# 'Hello  world' vs 'Hello world' counts as an error \u2014 normalise at load time.\n",
    "def normalize_text(text: str) -> str:\n",
    "    text = text.strip()\n",
    "    text = re.sub(r'\\s+', ' ', text)            # collapse multiple spaces\n",
    "    text = re.sub(r'\\s([?.!,;:])', r'\\1', text) # remove space before punctuation\n",
    "    return text\n",
    "\n",
    "\n",
    "class HandwritingDataset(Dataset):\n",
    "    \"\"\"\n",
    "    Loads handwritten line crops from a directory.\n",
    "    labels.txt format (tab-separated):\n",
    "        0001.png\\tThe mitochondria is the powerhouse of the cell\n",
    "    \"\"\"\n",
    "    def __init__(self, data_dir, proc, max_length=128, augment=False):\n",
    "        self.processor  = proc\n",
    "        self.max_length = max_length\n",
    "        self.augment    = augment\n",
    "        self.data       = []\n",
    "        data_dir   = Path(data_dir)\n",
    "        images_dir = data_dir / 'images'\n",
    "        skipped    = 0\n",
    "\n",
    "        with open(data_dir / 'labels.txt', encoding='utf-8') as f:\n",
    "            for line in f:\n",
    "                line = line.strip()\n",
    "                if '\\t' not in line:\n",
    "                    continue\n",
    "                fname, text = line.split('\\t', 1)\n",
    "                p = images_dir / fname\n",
    "                if not p.exists():\n",
    "                    continue\n",
    "\n",
    "                # doc 4 \u2014 label cleaning: normalise every label\n",
    "                text = normalize_text(text)\n",
    "                if not text:\n",
    "                    skipped += 1\n",
    "                    continue\n",
    "\n",
    "                # doc 4 \u2014 augmentation section: skip blank/tiny/corrupt images\n",
    "                try:\n",
    "                    img = Image.open(p)\n",
    "                    w, h = img.size\n",
    "                    if w < 32 or h < 8:\n",
    "                        skipped += 1\n",
    "                        img.close()\n",
    "                        continue\n",
    "                    img.close()\n",
    "                except Exception:\n",
    "                    skipped += 1\n",
    "                    continue\n",
    "\n",
    "                self.data.append((str(p), text))\n",
    "\n",
    "        print(f'  {data_dir.name:8s}: {len(self.data):,} samples  '\n",
    "              f'({skipped} skipped \u2014 blank label or corrupt/tiny image)')\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.data)\n",
    "\n",
    "    def _augment(self, image):\n",
    "        # doc 4 \u2014 augmentation: rotation \u00b13\u00b0\n",
    "        if random.random() > 0.5:\n",
    "            image = image.rotate(random.uniform(-3, 3), expand=True, fillcolor=255)\n",
    "\n",
    "        # doc 4 \u2014 augmentation: brightness change\n",
    "        if random.random() > 0.5:\n",
    "            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))\n",
    "\n",
    "        # doc 4 \u2014 augmentation: contrast change\n",
    "        if random.random() > 0.5:\n",
    "            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))\n",
    "\n",
    "        # doc 4 \u2014 augmentation: gaussian blur (simulates scan resolution variation)\n",
    "        if random.random() > 0.7:\n",
    "            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))\n",
    "\n",
    "        # doc 4 \u2014 augmentation: gaussian noise (simulates scan quality variation)\n",
    "        if random.random() > 0.5:\n",
    "            img_np  = np.array(image).astype(np.float32)\n",
    "            noise   = np.random.normal(0, random.uniform(3, 8), img_np.shape)\n",
    "            img_np  = np.clip(img_np + noise, 0, 255).astype(np.uint8)\n",
    "            image   = Image.fromarray(img_np)\n",
    "\n",
    "        # doc 3, point 3: elastic distortion \u2014 simulates pen pressure and paper texture\n",
    "        if random.random() > 0.5:\n",
    "            try:\n",
    "                img_np = np.array(image.convert('L'))\n",
    "                h, w   = img_np.shape\n",
    "                alpha  = w * random.uniform(0.03, 0.06)\n",
    "                sigma  = w * random.uniform(0.04, 0.06)\n",
    "                dx = cv2.GaussianBlur(\n",
    "                    (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma\n",
    "                ) * alpha\n",
    "                dy = cv2.GaussianBlur(\n",
    "                    (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma\n",
    "                ) * alpha\n",
    "                x, y  = np.meshgrid(np.arange(w), np.arange(h))\n",
    "                map_x = (x + dx).astype(np.float32)\n",
    "                map_y = (y + dy).astype(np.float32)\n",
    "                distorted = cv2.remap(\n",
    "                    img_np, map_x, map_y,\n",
    "                    interpolation=cv2.INTER_LINEAR,\n",
    "                    borderMode=cv2.BORDER_REPLICATE\n",
    "                )\n",
    "                image = Image.fromarray(distorted).convert('RGB')\n",
    "            except Exception:\n",
    "                pass  # never abort training due to augmentation failure\n",
    "\n",
    "        return image\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        img_path, text = self.data[idx]\n",
    "        image = Image.open(img_path).convert('RGB')\n",
    "        if self.augment:\n",
    "            image = self._augment(image)\n",
    "\n",
    "        # doc 4, point 2: always use processor for image preprocessing \u2014\n",
    "        # handles correct normalization and resizing for TrOCR\n",
    "        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values.squeeze()\n",
    "\n",
    "        # doc 4, point 1: padding tokens replaced with -100 so loss ignores them\n",
    "        labels = self.processor.tokenizer(\n",
    "            text, padding='max_length', max_length=self.max_length,\n",
    "            truncation=True, return_tensors='pt'\n",
    "        ).input_ids.squeeze()\n",
    "        labels[labels == self.processor.tokenizer.pad_token_id] = -100\n",
    "\n",
    "        return {'pixel_values': pixel_values, 'labels': labels}\n",
    "\n",
    "\n",
    "def compute_metrics(eval_pred):\n",
    "    preds, refs = eval_pred\n",
    "    vocab_size = processor.tokenizer.vocab_size\n",
    "    # Clamp to valid token range \u2014 prevents OverflowError with fp16\n",
    "    preds = np.clip(preds, 0, vocab_size - 1).astype(np.int32)\n",
    "    refs  = np.where(refs != -100, refs, processor.tokenizer.pad_token_id)\n",
    "    refs  = np.clip(refs,  0, vocab_size - 1).astype(np.int32)\n",
    "    pred_str = processor.batch_decode(preds, skip_special_tokens=True)\n",
    "    ref_str  = processor.batch_decode(refs,  skip_special_tokens=True)\n",
    "    cer = evaluate.load('cer').compute(predictions=pred_str, references=ref_str)\n",
    "    wer = evaluate.load('wer').compute(predictions=pred_str, references=ref_str)\n",
    "    return {'cer': cer, 'wer': wer}\n",
    "\n",
    "\n",
    "print('\u2705 Dataset class and metrics ready')\n"
   ],
   "id": "jXbTlNTHgbAr"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BmkKuYnbgbAt"
   },
   "source": [
    "## Cell 5 \u2014 Training configuration\n",
    "\n",
    "**Do not change `MODEL_NAME`** \u2014 `trocr-large-handwritten` is the correct choice for exam handwriting.\n",
    "Only reduce `BATCH_SIZE` if you get a CUDA out-of-memory error.\n"
   ],
   "id": "BmkKuYnbgbAt"
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 0
    },
    "id": "5gC25kgFgbAt",
    "outputId": "b1d1a06a-6668-406b-e153-d1cef3f4207b"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Model         : microsoft/trocr-large-handwritten\n",
      "Epochs        : 15\n",
      "Batch size    : 8  (effective: 32)\n",
      "Learning rate : 5e-05\n",
      "Augmentation  : True\n",
      "GPU           : Tesla T4\n"
     ]
    }
   ],
   "source": [
    "# \u2500\u2500\u2500 MODEL \u2014 always large for production \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "MODEL_NAME    = 'microsoft/trocr-large-handwritten'   # upgraded from small\n",
    "\n",
    "# \u2500\u2500\u2500 Hyperparameters \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "EPOCHS        = 15          # 15 is sweet spot \u2014 more risks overfitting on small data\n",
    "BATCH_SIZE    = 8           # safe for T4 (16 GB) with trocr-large; reduce to 4 if OOM\n",
    "LEARNING_RATE = 5e-5        # optimal for TrOCR fine-tuning \u2014 do not change\n",
    "WARMUP_STEPS  = 300\n",
    "GRAD_ACCUM    = 4           # effective batch = 8 x 4 = 32\n",
    "AUGMENT       = True        # always True \u2014 critical for messy handwriting generalisation\n",
    "\n",
    "print(f'Model         : {MODEL_NAME}')\n",
    "print(f'Epochs        : {EPOCHS}')\n",
    "print(f'Batch size    : {BATCH_SIZE}  (effective: {BATCH_SIZE * GRAD_ACCUM})')\n",
    "print(f'Learning rate : {LEARNING_RATE}')\n",
    "print(f'Augmentation  : {AUGMENT}')\n",
    "print(f'GPU           : {torch.cuda.get_device_name(0)}')\n"
   ],
   "id": "5gC25kgFgbAt"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rozyT0R6gbAu"
   },
   "source": [
    "## Cell 6 \u2014 Copy dataset from Drive to local Colab SSD\n",
    "\n",
    "Training directly from Google Drive is ~10\u00d7 slower than local SSD.\n",
    "This one-time copy takes 3\u20135 minutes and makes every epoch significantly faster.\n"
   ],
   "id": "rozyT0R6gbAu"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 0
    },
    "id": "0s2nUEJmgbAu",
    "outputId": "5bde4efa-7b99-4ed6-dfbc-53806318d930"
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Removing old dataset...\n",
      "\n",
      "\ud83d\udce6 Creating zip archive (one-time)...\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import lmdb\n",
    "import shutil\n",
    "import time\n",
    "from pathlib import Path\n",
    "from tqdm import tqdm\n",
    "\n",
    "src = \"/content/drive/MyDrive/Intelligrade/datasets/handwriting\"\n",
    "zip_path = \"/content/drive/MyDrive/Intelligrade/datasets/handwriting.zip\"\n",
    "\n",
    "local_zip = \"/content/handwriting.zip\"\n",
    "dst_root = \"/content/datasets\"\n",
    "dst = Path(\"/content/datasets/handwriting\")\n",
    "\n",
    "lmdb_path = \"/content/datasets/handwriting_lmdb\"\n",
    "\n",
    "if dst.exists():\n",
    "    print(\"Removing old dataset...\")\n",
    "    shutil.rmtree(dst)\n",
    "\n",
    "if not os.path.exists(zip_path):\n",
    "    print(\"\\n\ud83d\udce6 Creating zip archive (one-time)...\")\n",
    "    start = time.time()\n",
    "    !zip -rq \"{zip_path}\" \"{src}\"\n",
    "    print(f\"Zip created in {time.time()-start:.1f} seconds\")\n",
    "\n",
    "print(\"\\n\ud83d\ude80 Copying zip to local SSD...\")\n",
    "start = time.time()\n",
    "!cp \"{zip_path}\" \"{local_zip}\"\n",
    "print(f\"Copy completed in {time.time()-start:.1f} seconds\")\n",
    "\n",
    "print(\"\\n\ud83d\udce6 Extracting dataset...\")\n",
    "start = time.time()\n",
    "!unzip -q \"{local_zip}\" -d \"{dst_root}\"\n",
    "print(f\"Extraction completed in {time.time()-start:.1f} seconds\\n\")\n",
    "\n",
    "TRAIN_DIR = dst / \"train\"\n",
    "VAL_DIR = dst / \"val\"\n",
    "TEST_DIR = dst / \"test\"\n",
    "\n",
    "print(\"\ud83d\udcca Dataset Statistics\")\n",
    "print(\"-\" * 30)\n",
    "\n",
    "total = 0\n",
    "\n",
    "for split in [\"train\",\"val\",\"test\"]:\n",
    "\n",
    "    label_file = dst / split / \"labels.txt\"\n",
    "\n",
    "    if not label_file.exists():\n",
    "        print(f\"{split:6s}: labels.txt not found\")\n",
    "        continue\n",
    "\n",
    "    with open(label_file) as f:\n",
    "        n = sum(1 for line in f if \"\\t\" in line)\n",
    "\n",
    "    print(f\"{split:6s}: {n:,} samples\")\n",
    "    total += n\n",
    "\n",
    "print(\"-\" * 30)\n",
    "print(f\"Total : {total:,} samples\")\n",
    "\n",
    "print(\"\\n\ud83d\udcc8 Dataset Quality Assessment\")\n",
    "\n",
    "if total < 200:\n",
    "    print(\"\u26a0\ufe0f  Very few samples \u2014 results will be limited.\")\n",
    "    print(\"Aim for at least 800+ training samples.\")\n",
    "\n",
    "elif total < 800:\n",
    "    print(f\"\u2139\ufe0f  {total:,} samples \u2014 expect CER around 12\u201318%.\")\n",
    "    print(\"More data will significantly improve results.\")\n",
    "\n",
    "else:\n",
    "    print(f\"\u2705 {total:,} samples \u2014 good dataset size.\")\n",
    "    print(\"Fine-tuned model should achieve CER < 12%.\")\n",
    "\n",
    "print(\"\\n\ud83d\ude80 Converting dataset to LMDB...\")\n",
    "start = time.time()\n",
    "\n",
    "env = lmdb.open(\n",
    "    lmdb_path,\n",
    "    map_size=1099511627776,\n",
    "    subdir=True,\n",
    "    meminit=False,\n",
    "    map_async=True\n",
    ")\n",
    "\n",
    "cache = {}\n",
    "cnt = 1\n",
    "\n",
    "def write_cache(env, cache):\n",
    "    with env.begin(write=True) as txn:\n",
    "        for k, v in cache.items():\n",
    "            txn.put(k.encode(), v)\n",
    "\n",
    "all_lines = []\n",
    "\n",
    "for split in [\"train\",\"val\",\"test\"]:\n",
    "\n",
    "    label_file = dst / split / \"labels.txt\"\n",
    "\n",
    "    if not label_file.exists():\n",
    "        continue\n",
    "\n",
    "    with open(label_file,\"r\",encoding=\"utf-8\") as f:\n",
    "        for line in f:\n",
    "            if \"\\t\" in line:\n",
    "                all_lines.append((split,line.strip()))\n",
    "\n",
    "for split, line in tqdm(all_lines, desc=\"Building LMDB\", unit=\"sample\"):\n",
    "\n",
    "    img_name, label = line.split(\"\\t\")\n",
    "\n",
    "    img_path = dst / split / img_name\n",
    "\n",
    "    if not img_path.exists():\n",
    "        continue\n",
    "\n",
    "    with open(img_path,\"rb\") as img_file:\n",
    "        image_bin = img_file.read()\n",
    "\n",
    "    image_key = f\"image-{cnt:09d}\"\n",
    "    label_key = f\"label-{cnt:09d}\"\n",
    "\n",
    "    cache[image_key] = image_bin\n",
    "    cache[label_key] = label.encode()\n",
    "\n",
    "    if cnt % 1000 == 0:\n",
    "        write_cache(env, cache)\n",
    "        cache = {}\n",
    "\n",
    "    cnt += 1\n",
    "\n",
    "cache[\"num-samples\"] = str(cnt-1).encode()\n",
    "write_cache(env, cache)\n",
    "\n",
    "print(f\"\\n\u2705 LMDB created with {cnt-1:,} samples\")\n",
    "print(f\"\u23f1 Conversion time: {time.time()-start:.1f} seconds\")\n",
    "print(f\"\ud83d\udcc1 LMDB location: {lmdb_path}\")"
   ],
   "id": "0s2nUEJmgbAu"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rDTVd8JEgbAv"
   },
   "source": [
    "## Cell 7 \u2014 Load model and train\n",
    "\n",
    "Logs CER and WER after each epoch. Early stopping halts if no improvement for 4 epochs.\n",
    "\n",
    "**Improvements from advice docs:**\n",
    "- **`decoder_start_token_id` / `pad_token_id` / `vocab_size`** (doc 4, point 3): required for stable teacher-forcing during training.\n",
    "- **Beam search on `model.config`** (doc 4, point 6): `num_beams=4`, `no_repeat_ngram_size=3`, `length_penalty=1.0` set before Trainer so per-epoch eval CER uses beam search.\n",
    "- **Gradient clipping `max_grad_norm=1.0`** (doc 4, point 7): stabilises transformer training, prevents exploding gradients.\n",
    "- **`fp16=True`** (doc 4, point 8): 40% faster training, lower VRAM usage.\n",
    "- **`dataloader_num_workers=4`, `dataloader_pin_memory=True`**: keeps GPU fed with augmented batches.\n"
   ],
   "id": "rDTVd8JEgbAv"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "LK1uuxRZgbAv"
   },
   "outputs": [],
   "source": [
    "import os\nfrom pathlib import Path\nfrom transformers import (\n    TrOCRProcessor, VisionEncoderDecoderModel,\n    Seq2SeqTrainer, Seq2SeqTrainingArguments,\n    default_data_collator, EarlyStoppingCallback\n)\n\nLOCAL_OUT = '/content/trocr-finetuned'\nMODEL_OUT = '/content/drive/MyDrive/Intelligrade/models/trocr-finetuned'\nos.makedirs(LOCAL_OUT, exist_ok=True)\nos.makedirs(MODEL_OUT, exist_ok=True)\n\n# \u2500\u2500 Verify local SSD copies \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nfor split, path in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:\n    lf = Path(path) / 'labels.txt'\n    if lf.exists():\n        n = sum(1 for l in open(lf) if '\\t' in l)\n        print(f'  \u2705 {split:6s}: {n:,} samples')\n    else:\n        raise FileNotFoundError(f'{split} not found \u2014 run Cell 6 first')\n\n# \u2500\u2500 Load processor and model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nprint(f'\\nLoading {MODEL_NAME}...')\nprocessor = TrOCRProcessor.from_pretrained(MODEL_NAME)\nmodel     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)\n\nmodel.config.decoder_start_token_id = processor.tokenizer.cls_token_id\nmodel.config.pad_token_id           = processor.tokenizer.pad_token_id\nmodel.config.vocab_size             = model.config.decoder.vocab_size\nmodel.gradient_checkpointing_enable()  # saves VRAM \u2014 essential for large model on T4\n\n# doc 2, point 5: set decoder generation config on model.config BEFORE Trainer.\n# Seq2SeqTrainer uses model.config during predict_with_generate (per-epoch eval),\n# not the generate() kwargs in the test cell. Without this, training-time CER\n# used greedy decoding while test-time used beam search \u2014 inconsistent metric.\nmodel.config.num_beams            = 4\nmodel.config.no_repeat_ngram_size = 3\nmodel.config.length_penalty       = 1.0\nmodel.config.max_length           = 64\n\nn_params = sum(p.numel() for p in model.parameters()) / 1e6\nprint(f'\u2705 Model loaded | {n_params:.0f}M parameters')\nprint(f'   decoder_start_token_id : {model.config.decoder_start_token_id}')\nprint(f'   num_beams              : {model.config.num_beams}')\nprint(f'   no_repeat_ngram_size   : {model.config.no_repeat_ngram_size}')\nprint(f'   max_grad_norm          : 1.0\\n')\n\n# \u2500\u2500 Load datasets \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nprint('Loading datasets from local SSD...')\ntrain_ds = HandwritingDataset(TRAIN_DIR, processor, augment=AUGMENT)\nval_ds   = HandwritingDataset(VAL_DIR,   processor, augment=False)\n\n# \u2500\u2500 Training arguments \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\ntraining_args = Seq2SeqTrainingArguments(\n    output_dir                  = LOCAL_OUT,\n    evaluation_strategy         = 'epoch',\n    save_strategy               = 'epoch',\n    save_total_limit            = 2,            # keep only 2 checkpoints\n    learning_rate               = LEARNING_RATE,\n    per_device_train_batch_size = BATCH_SIZE,\n    per_device_eval_batch_size  = BATCH_SIZE,\n    gradient_accumulation_steps = GRAD_ACCUM,\n    weight_decay                = 0.01,\n    max_grad_norm               = 1.0,    # doc 4, point 7: gradient clipping \u2014 stabilises training\n    warmup_steps                = WARMUP_STEPS,\n    num_train_epochs            = EPOCHS,\n    predict_with_generate       = True,\n    generation_max_length       = 128,\n    fp16                        = True,         # mixed precision \u2014 saves VRAM\n    logging_steps               = 50,\n    load_best_model_at_end      = True,\n    metric_for_best_model       = 'cer',\n    greater_is_better           = False,        # lower CER = better\n    remove_unused_columns       = False,\n    dataloader_num_workers      = 4,    # doc 2, point 7: was 2 \u2014 keeps GPU fed\n    dataloader_pin_memory       = True, # faster CPU\u2192GPU batch transfer\n    report_to                   = [],           # disable wandb/tensorboard noise\n)\n\ntrainer = Seq2SeqTrainer(\n    model           = model,\n    args            = training_args,\n    train_dataset   = train_ds,\n    eval_dataset    = val_ds,\n    data_collator   = default_data_collator,\n    compute_metrics = compute_metrics,\n    callbacks       = [EarlyStoppingCallback(early_stopping_patience=4)],\n)\n\nprint('\ud83d\ude80 Training started...')\nprint('   Target: val CER < 0.12 (12%). Early stopping if no improvement for 4 epochs.\\n')\ntrainer.train()\nprint('\\n\u2705 Training complete! Best model checkpoint restored.')\n"
   ],
   "id": "LK1uuxRZgbAv"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2G94pt33gbAw"
   },
   "source": [
    "## Cell 8 \u2014 Evaluate on test set"
   ],
   "id": "2G94pt33gbAw"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "sYHBivs8gbAw"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "device      = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "test_ds     = HandwritingDataset(TEST_DIR, processor, augment=False)\n",
    "test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)\n",
    "\n",
    "cer_metric = evaluate.load('cer')\n",
    "wer_metric = evaluate.load('wer')\n",
    "all_preds, all_refs = [], []\n",
    "\n",
    "model.to(device)\n",
    "model.eval()\n",
    "\n",
    "with torch.no_grad():\n",
    "    for batch in test_loader:\n",
    "        px      = batch['pixel_values'].to(device)\n",
    "        gen_ids = model.generate(px, max_length=128, num_beams=4, early_stopping=True)\n",
    "        preds   = processor.batch_decode(gen_ids, skip_special_tokens=True)\n",
    "        labels  = batch['labels'].cpu().numpy()\n",
    "        labels  = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)\n",
    "        labels  = np.clip(labels, 0, processor.tokenizer.vocab_size - 1).astype(np.int32)\n",
    "        refs    = processor.batch_decode(labels, skip_special_tokens=True)\n",
    "        all_preds.extend(preds)\n",
    "        all_refs.extend(refs)\n",
    "\n",
    "cer = cer_metric.compute(predictions=all_preds, references=all_refs)\n",
    "wer = wer_metric.compute(predictions=all_preds, references=all_refs)\n",
    "\n",
    "print('\\n' + '='*52)\n",
    "print('   IntelliGrade-H \u2014 Final Test Results')\n",
    "print('='*52)\n",
    "print(f'  Character Error Rate (CER): {cer*100:.2f}%')\n",
    "print(f'  Word Error Rate (WER)     : {wer*100:.2f}%')\n",
    "print(f'  Samples evaluated         : {len(test_ds):,}')\n",
    "print('='*52)\n",
    "\n",
    "if cer < 0.08:\n",
    "    print('\u2705 Excellent! CER < 8% \u2014 competitive with Google Vision on exam data.')\n",
    "elif cer < 0.12:\n",
    "    print('\u2705 Good. CER < 12% \u2014 significantly better than base trocr-large.')\n",
    "elif cer < 0.20:\n",
    "    print('\u2139\ufe0f  Acceptable. Consider collecting more messy handwriting samples.')\n",
    "else:\n",
    "    print('\u26a0\ufe0f  CER > 20%. Collect more data \u2014 aim for 1000+ training samples.')\n"
   ],
   "id": "sYHBivs8gbAw"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GlkGpcsHgbAx"
   },
   "source": [
    "## Cell 9 \u2014 Save model to Google Drive"
   ],
   "id": "GlkGpcsHgbAx"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "MvXNRv9DgbAx"
   },
   "outputs": [],
   "source": [
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "LOCAL_OUT = '/content/trocr-finetuned'\n",
    "MODEL_OUT = '/content/drive/MyDrive/Intelligrade/models/trocr-finetuned'\n",
    "\n",
    "trainer.save_model(LOCAL_OUT)\n",
    "processor.save_pretrained(LOCAL_OUT)\n",
    "\n",
    "print(f'Copying model to Drive: {MODEL_OUT}')\n",
    "if Path(MODEL_OUT).exists():\n",
    "    shutil.rmtree(MODEL_OUT)\n",
    "shutil.copytree(LOCAL_OUT, MODEL_OUT)\n",
    "\n",
    "# Verify \u2014 IntelliGrade-H detects the model by checking for config.json\n",
    "config_ok = (Path(MODEL_OUT) / 'config.json').exists()\n",
    "print(f'\\n\u2705 Model saved to Google Drive!')\n",
    "print(f'   config.json present: {config_ok}  \u2190 IntelliGrade-H auto-detects via this file')\n",
    "print(f'   Location: {MODEL_OUT}')\n",
    "print(\"\"\"\n",
    "\u2500\u2500 Deploy to IntelliGrade-H \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "1. In Google Drive: right-click trocr-finetuned/ \u2192 Download as ZIP\n",
    "   Path: My Drive/Intelligrade/models/trocr-finetuned/\n",
    "\n",
    "2. Extract to your project:\n",
    "   IntelliGrade-H/models/trocr-finetuned/\n",
    "   (the folder must contain config.json \u2014 that triggers auto-detection)\n",
    "\n",
    "3. Restart the backend:\n",
    "   python run.py\n",
    "\n",
    "4. The system will log:\n",
    "   Fine-tuned TrOCR model found at models/trocr-finetuned \u2014 using it.\n",
    "\n",
    "No .env changes needed \u2014 ocr_module.py detects the model automatically.\n",
    "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    "\"\"\")\n"
   ],
   "id": "MvXNRv9DgbAx"
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "j5uizHOKgbAy"
   },
   "source": [
    "## Cell 10 \u2014 Visual test (optional)\n",
    "\n",
    "Picks 5 random samples from the val set and shows prediction vs ground truth side by side.\n"
   ],
   "id": "j5uizHOKgbAy"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "2kHhQS7dgbAy"
   },
   "outputs": [],
   "source": [
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "from PIL import Image\n",
    "\n",
    "samples = [l.strip().split('\\t') for l in open(Path(VAL_DIR) / 'labels.txt') if '\\t' in l]\n",
    "random.shuffle(samples)\n",
    "test_cases = samples[:5]\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "model.eval()\n",
    "\n",
    "fig, axes = plt.subplots(len(test_cases), 1, figsize=(14, len(test_cases) * 2.5))\n",
    "if len(test_cases) == 1:\n",
    "    axes = [axes]\n",
    "\n",
    "all_cer = []\n",
    "for ax, (fname, ground_truth) in zip(axes, test_cases):\n",
    "    img_path = Path(VAL_DIR) / 'images' / fname\n",
    "    img = Image.open(img_path).convert('RGB')\n",
    "    px  = processor(img, return_tensors='pt').pixel_values.to(device)\n",
    "    with torch.no_grad():\n",
    "        ids  = model.generate(px, max_length=128, num_beams=4)\n",
    "    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]\n",
    "\n",
    "    sample_cer = evaluate.load('cer').compute(predictions=[pred], references=[ground_truth])\n",
    "    all_cer.append(sample_cer)\n",
    "    icon = '\u2705' if sample_cer < 0.10 else ('\u26a0\ufe0f' if sample_cer < 0.25 else '\u274c')\n",
    "\n",
    "    ax.imshow(img, cmap='gray')\n",
    "    ax.set_title(\n",
    "        f'{icon} GT: {ground_truth}\\n   PR: {pred}  |  CER: {sample_cer*100:.1f}%',\n",
    "        fontsize=9, loc='left'\n",
    "    )\n",
    "    ax.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('/content/sample_predictions.png', dpi=100, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f'Average CER on these {len(test_cases)} samples: {sum(all_cer)/len(all_cer)*100:.1f}%')\n",
    "print('Figure saved to /content/sample_predictions.png')\n"
   ],
   "id": "2kHhQS7dgbAy"
  }
 ]
}
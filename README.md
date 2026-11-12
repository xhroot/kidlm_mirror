# KidLM-5M: Offline Micro-LLM Training & Deployment on Seeed Studio XIAO ESP32-S3

A self-contained micro-LLM (~5.23M parameters) targeting an **8-year-old reading level (Grades 3–6)**, trained on `tafseer-nayeem/KidLM-corpus` (EMNLP 2024), and deployed fully offline on a **Seeed Studio XIAO ESP32-S3**.

---

## 1. Hardware Constraints & Security Profile

* **Target Hardware**: Seeed Studio XIAO ESP32-S3
  * **Flash**: 8 MB (`0x190000` to `0x7F0000` allocated for model weights = **6.375 MB**)
  * **RAM**: 8 MB Octal PSRAM + 512 KB Internal SRAM
* **Air-Gapped Security**:
  * Wi-Fi and Bluetooth libraries completely excluded at compile time.
  * Zero RF peripheral initialization in firmware.
  * No U.FL antenna attached (near-field RF drops below usable thresholds beyond ~5 cm).

---

## 2. Repository Structure

```text
kidlm_mirror/
├── kidlm_shards/                  # 4 compressed Parquet shards (~116 MB total, all <50 MB)
│   ├── kidlm_shard_1_of_4.parquet # 48.0 MB (71,641 docs)
│   ├── kidlm_shard_2_of_4.parquet # 27.3 MB (71,641 docs)
│   ├── kidlm_shard_3_of_4.parquet # 20.2 MB (71,641 docs)
│   └── kidlm_shard_4_of_4.parquet # 20.9 MB (71,641 docs)
├── kidlm_tokenizer/               # Custom BPE Tokenizer (vocab_size = 4,096)
│   └── tokenizer.json
├── kidlm_firmware/                # Arduino/ESP-IDF C++ Inference Engine
│   ├── kidlm_firmware.ino
│   └── partitions.csv             # Custom 6.375 MB partition table
├── model.py                       # Llama-style decoder model & INT8/INT4 exporter
├── train_kidlm.py                 # 100% offline training pipeline
├── flash_kidlm.sh                 # High-speed esptool flashing script
└── README.md
```

---

## 3. Sharding

All Parquet shards are `< 50 MB`, completely bypassing GitHub's 100 MiB limit without requiring Git LFS.

### Step 1: Clone 
```bash
git clone https://github.com/<your-username>/kidlm_mirror.git
cd kidlm_mirror
```

### Step 2: Run Training (100% Offline)
```bash
python3 train_kidlm.py --epochs 2 --batch_size 32
```
* **Dataset loading**: loads directly from local Parquet shards via `pyarrow`.
* **Output**: saves `kidlm_checkpoints/kidlm_5m.pt` and automatically exports the INT8 quantized binary `kidlm_checkpoints/kidlm_5m_q8.bin` (**~5.5 MB**).

---

## 4. Deploying to Seeed Studio XIAO ESP32-S3

### Step 1: Flash the Model Weights Binary
Connect the XIAO ESP32-S3 via USB-C and run:
```bash
bash flash_kidlm.sh /dev/ttyACM0
```
*(Flashes `kidlm_5m_q8.bin` to Flash offset `0x190000` at 921,600 baud)*.

### Step 2: Upload Firmware via Arduino IDE
1. Open [kidlm_firmware/kidlm_firmware.ino](file:///home/xhroot/side/kidlm_mirror/kidlm_firmware/kidlm_firmware.ino) in Arduino IDE.
2. Select board: **XIAO_ESP32S3**.
3. Tools settings:
   - **PSRAM**: `OPI PSRAM` (Enabled)
   - **USB CDC On Boot**: `Enabled`
   - **Flash Size**: `8MB (64Mb)`
   - **Partition Scheme**: uses `partitions.csv` in sketch folder automatically.
4. Click **Upload** (`Ctrl+U`).

### Step 3: Inference over Serial CDC
Open Serial Monitor at `115200` baud. Type a prompt (e.g. `"The little dragon wanted to find"`) and watch the on-chip micro-LLM stream tokens in real-time from PSRAM/Flash!

---

## 5. License & Attribution

This repository and the underlying dataset shards are distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC-BY-NC-SA-4.0](LICENSE))** to remain 100% congruent with the original KidLM corpus release.

### Citation
If you use this work or the KidLM dataset, please cite the authors' EMNLP 2024 paper:

```bibtex
@inproceedings{nayeem-rafiei-2024-kidlm,
    title = "{K}id{LM}: Advancing Language Models for Children {--} Early Insights and Future Directions",
    author = "Nayeem, Mir Tafseer and Rafiei, Davood",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.277",
    pages = "4813--4836"
}
```


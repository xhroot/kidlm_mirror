/*
 * KidLM-5M Offline Micro-LLM Firmware for Seeed Studio XIAO ESP32-S3
 * Llama-style decoder-only (RMSNorm, RoPE, SwiGLU, tied embeddings)
 * 
 * Hardware Specs:
 * - Board: Seeed Studio XIAO ESP32-S3 (8MB Flash, 8MB PSRAM)
 * - Model Partition: 0x190000, Size 0x660000 (6.375 MB)
 * - Air-Gapped: Wi-Fi & Bluetooth peripherals uninitialized.
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "esp_partition.h"

// Model Architecture Hyperparameters
#define DIM 256
#define HIDDEN_DIM 688
#define N_LAYERS 6
#define N_HEADS 8
#define HEAD_DIM (DIM / N_HEADS) // 32
#define VOCAB_SIZE 4096
#define MAX_SEQ_LEN 256

// Partition & Weight Pointers
esp_partition_mmap_handle_t mmap_handle;
const uint8_t* model_base = NULL;
bool model_ready = false;

// Quantized Tensor Representation (INT8 + FP32 Scale)
struct QuantizedWeight {
  float scale;
  const int8_t* q;
};

// Transformer Weights Struct
struct TransformerWeights {
  QuantizedWeight token_embedding_table;
  float* rms_att_weight[N_LAYERS];
  QuantizedWeight wq[N_LAYERS];
  QuantizedWeight wk[N_LAYERS];
  QuantizedWeight wv[N_LAYERS];
  QuantizedWeight wo[N_LAYERS];
  float* rms_ffn_weight[N_LAYERS];
  QuantizedWeight w1[N_LAYERS]; // gate
  QuantizedWeight w2[N_LAYERS]; // down
  QuantizedWeight w3[N_LAYERS]; // up
  float* rms_final_weight;
} w;

// Activation State Buffers (Allocated in PSRAM)
float* x = NULL;          // (DIM)
float* xb = NULL;         // (DIM)
float* xb2 = NULL;        // (DIM)
float* hb = NULL;         // (HIDDEN_DIM)
float* hb2 = NULL;        // (HIDDEN_DIM)
float* q = NULL;          // (DIM)
float* k = NULL;          // (DIM)
float* v = NULL;          // (DIM)
float* att = NULL;        // (N_HEADS * MAX_SEQ_LEN)
float* logits = NULL;     // (VOCAB_SIZE)
float* key_cache = NULL;   // (N_LAYERS * MAX_SEQ_LEN * DIM)
float* value_cache = NULL; // (N_LAYERS * MAX_SEQ_LEN * DIM)

// Math Utilities
inline float silu(float val) {
  return val / (1.0f + expf(-val));
}

void rmsnorm(float* o, float* in, float* weight, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += in[j] * in[j];
  }
  ss /= size;
  ss += 1e-5f;
  float inv = 1.0f / sqrtf(ss);
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (in[j] * inv);
  }
}

// INT8 Matmul: out = (W_q * scale) @ x
void matmul_q8(float* xout, float* xin, const QuantizedWeight& qw, int n, int d) {
  float scale = qw.scale;
  const int8_t* w_data = qw.q;

  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    const int8_t* row = w_data + (i * n);
    for (int j = 0; j < n; j++) {
      val += xin[j] * (float)row[j];
    }
    xout[i] = val * scale;
  }
}

// RoPE Rotary Positional Embedding
void apply_rope(float* vec, int pos, int head_dim, int num_heads) {
  for (int h = 0; h < num_heads; h++) {
    float* h_vec = vec + h * head_dim;
    for (int i = 0; i < head_dim; i += 2) {
      float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
      float val = (float)pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);

      float v0 = h_vec[i];
      float v1 = h_vec[i + 1];
      h_vec[i] = v0 * fcr - v1 * fci;
      h_vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}

// Single Token Forward Pass
float* forward_token(int token, int pos) {
  // 1. Token embedding lookup (tied embedding with INT8 dequantization)
  const int8_t* emb_row = w.token_embedding_table.q + (token * DIM);
  float emb_scale = w.token_embedding_table.scale;
  for (int i = 0; i < DIM; i++) {
    x[i] = (float)emb_row[i] * emb_scale;
  }

  // 2. Transformer Blocks
  for (int l = 0; l < N_LAYERS; l++) {
    // Attention RMSNorm
    rmsnorm(xb, x, w.rms_att_weight[l], DIM);

    // Q, K, V projections
    matmul_q8(q, xb, w.wq[l], DIM, DIM);
    matmul_q8(k, xb, w.wk[l], DIM, DIM);
    matmul_q8(v, xb, w.wv[l], DIM, DIM);

    // RoPE
    apply_rope(q, pos, HEAD_DIM, N_HEADS);
    apply_rope(k, pos, HEAD_DIM, N_HEADS);

    // Cache KV
    int layer_offset = l * MAX_SEQ_LEN * DIM;
    int pos_offset = pos * DIM;
    memcpy(key_cache + layer_offset + pos_offset, k, DIM * sizeof(float));
    memcpy(value_cache + layer_offset + pos_offset, v, DIM * sizeof(float));

    // Multi-Head Attention
    for (int h = 0; h < N_HEADS; h++) {
      float* q_head = q + h * HEAD_DIM;
      float* att_head = att + h * MAX_SEQ_LEN;

      for (int t = 0; t <= pos; t++) {
        float* k_head = key_cache + layer_offset + t * DIM + h * HEAD_DIM;
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
          score += q_head[d] * k_head[d];
        }
        att_head[t] = score / sqrtf((float)HEAD_DIM);
      }

      // Softmax over t = 0..pos
      float max_val = att_head[0];
      for (int t = 1; t <= pos; t++) {
        if (att_head[t] > max_val) max_val = att_head[t];
      }
      float sum = 0.0f;
      for (int t = 0; t <= pos; t++) {
        att_head[t] = expf(att_head[t] - max_val);
        sum += att_head[t];
      }
      float inv_sum = 1.0f / (sum + 1e-12f);
      for (int t = 0; t <= pos; t++) {
        att_head[t] *= inv_sum;
      }

      // Context = Att @ V
      float* xb_head = xb + h * HEAD_DIM;
      memset(xb_head, 0, HEAD_DIM * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        float* v_head = value_cache + layer_offset + t * DIM + h * HEAD_DIM;
        float a = att_head[t];
        for (int d = 0; d < HEAD_DIM; d++) {
          xb_head[d] += a * v_head[d];
        }
      }
    }

    // Output projection
    matmul_q8(xb2, xb, w.wo[l], DIM, DIM);

    // Residual connection
    for (int i = 0; i < DIM; i++) x[i] += xb2[i];

    // FFN RMSNorm
    rmsnorm(xb, x, w.rms_ffn_weight[l], DIM);

    // SwiGLU: w2(silu(w1(x)) * w3(x))
    matmul_q8(hb, xb, w.w1[l], DIM, HIDDEN_DIM);
    matmul_q8(hb2, xb, w.w3[l], DIM, HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) {
      hb[i] = silu(hb[i]) * hb2[i];
    }
    matmul_q8(xb, hb, w.w2[l], HIDDEN_DIM, DIM);

    // Residual connection
    for (int i = 0; i < DIM; i++) x[i] += xb[i];
  }

  // 3. Final RMSNorm
  rmsnorm(x, x, w.rms_final_weight, DIM);

  // 4. Classifier Head (Tied embeddings)
  matmul_q8(logits, x, w.token_embedding_table, DIM, VOCAB_SIZE);
  return logits;
}

// Greedy / Temperature Sampler
int sample_token(float* logit_arr, float temperature = 0.7f) {
  if (temperature <= 0.05f) {
    int max_i = 0;
    float max_val = logit_arr[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
      if (logit_arr[i] > max_val) {
        max_val = logit_arr[i];
        max_i = i;
      }
    }
    return max_i;
  }

  // Softmax with temperature
  float max_l = logit_arr[0] / temperature;
  for (int i = 1; i < VOCAB_SIZE; i++) {
    float val = logit_arr[i] / temperature;
    if (val > max_l) max_l = val;
  }

  float sum = 0.0f;
  for (int i = 0; i < VOCAB_SIZE; i++) {
    logit_arr[i] = expf((logit_arr[i] / temperature) - max_l);
    sum += logit_arr[i];
  }

  float r = ((float)rand() / (float)RAND_MAX) * sum;
  float c = 0.0f;
  for (int i = 0; i < VOCAB_SIZE; i++) {
    c += logit_arr[i];
    if (c >= r) return i;
  }
  return VOCAB_SIZE - 1;
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\n=======================================================");
  Serial.println(" KidLM-5M Offline Micro-LLM on Seeed Studio XIAO ESP32-S3");
  Serial.println(" Reading Level: Grades 3-6 (Ages 8-12)");
  Serial.println(" Security: Air-gapped (No Wi-Fi/BLE Drivers Initialized)");
  Serial.println("=======================================================");

  // 1. Initialize PSRAM Buffers
  if (!psramInit()) {
    Serial.println("FATAL: PSRAM initialization failed! Enable OPI PSRAM in Tools menu.");
    while (1) delay(1000);
  }

  Serial.printf("PSRAM Total: %d KB, Free: %d KB\n", ESP.getPsramSize()/1024, ESP.getFreePsram()/1024);

  // Allocate KV caches & state buffers in PSRAM
  size_t kv_sz = N_LAYERS * MAX_SEQ_LEN * DIM * sizeof(float);
  key_cache = (float*)ps_malloc(kv_sz);
  value_cache = (float*)ps_malloc(kv_sz);

  x = (float*)ps_malloc(DIM * sizeof(float));
  xb = (float*)ps_malloc(DIM * sizeof(float));
  xb2 = (float*)ps_malloc(DIM * sizeof(float));
  hb = (float*)ps_malloc(HIDDEN_DIM * sizeof(float));
  hb2 = (float*)ps_malloc(HIDDEN_DIM * sizeof(float));
  q = (float*)ps_malloc(DIM * sizeof(float));
  k = (float*)ps_malloc(DIM * sizeof(float));
  v = (float*)ps_malloc(DIM * sizeof(float));
  att = (float*)ps_malloc(N_HEADS * MAX_SEQ_LEN * sizeof(float));
  logits = (float*)ps_malloc(VOCAB_SIZE * sizeof(float));

  if (!key_cache || !value_cache || !logits) {
    Serial.println("FATAL: Failed to allocate activation buffers in PSRAM!");
    while (1) delay(1000);
  }
  Serial.printf("Allocated KV Cache (%d KB) and State Buffers successfully.\n", (int)(kv_sz * 2 / 1024));

  // 2. Memory-map model partition at 0x190000
  const esp_partition_t* part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, 0x40, "model");

  if (!part) {
    Serial.println("WARNING: 'model' partition not found in partition table (0x190000, 0x660000)!");
    Serial.println("Please flash partitions.csv and kidlm_5m_q8.bin.");
    return;
  }

  Serial.printf("Found 'model' partition: offset 0x%x, size %d bytes (%.2f MB)\n", 
                part->address, part->size, (float)part->size / (1024.0f * 1024.0f));

  esp_err_t err = esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, (const void**)&model_base, &mmap_handle);
  if (err != ESP_OK) {
    Serial.printf("FATAL: Failed to mmap model partition, err=0x%x\n", err);
    return;
  }

  // 3. Map weight pointers from header (256 bytes)
  const uint8_t* ptr = model_base + 256;

  auto read_qw = [&](QuantizedWeight& qw, int n, int d) {
    qw.scale = *(const float*)ptr; ptr += sizeof(float);
    qw.q = (const int8_t*)ptr; ptr += (n * d);
  };

  auto read_fp = [&](float*& fp, int sz) {
    fp = (float*)ptr; ptr += (sz * sizeof(float));
  };

  read_qw(w.token_embedding_table, DIM, VOCAB_SIZE);

  for (int l = 0; l < N_LAYERS; l++) {
    read_fp(w.rms_att_weight[l], DIM);
    read_qw(w.wq[l], DIM, DIM);
    read_qw(w.wk[l], DIM, DIM);
    read_qw(w.wv[l], DIM, DIM);
    read_qw(w.wo[l], DIM, DIM);
    read_fp(w.rms_ffn_weight[l], DIM);
    read_qw(w.w1[l], DIM, HIDDEN_DIM);
    read_qw(w.w2[l], HIDDEN_DIM, DIM);
    read_qw(w.w3[l], DIM, HIDDEN_DIM);
  }
  read_fp(w.rms_final_weight, DIM);

  model_ready = true;
  Serial.printf("KidLM-5M mapped! Consumed %d bytes of Flash (%.2f MB).\n", (int)(ptr - model_base), (float)(ptr - model_base) / (1024.0f * 1024.0f));
  Serial.println("Type prompt over Serial CDC to generate text...\n");
}

void loop() {
  if (Serial.available() > 0) {
    String prompt = Serial.readStringUntil('\n');
    prompt.trim();

    if (prompt.length() > 0 && model_ready) {
      Serial.printf("\nPrompt: \"%s\"\nGenerating: ", prompt.c_str());

      // Simple generation demo loop
      int current_token = 1; // <s>
      unsigned long t0 = millis();
      int gen_len = 50;

      for (int pos = 0; pos < gen_len && pos < MAX_SEQ_LEN; pos++) {
        float* token_logits = forward_token(current_token, pos);
        int next_token = sample_token(token_logits, 0.7f);
        if (next_token == 2) break; // </s> EOS
        current_token = next_token;
        Serial.printf("[tok_%d] ", current_token);
      }
      unsigned long elapsed = millis() - t0;
      Serial.printf("\n(Generated %d tokens in %lu ms, %.1f tok/s)\n\n", gen_len, elapsed, (float)gen_len / ((float)elapsed / 1000.0f));
    }
  }
}

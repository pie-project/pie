#pragma once
//
// Qwen3-0.6B architecture constants — pulled from the authoritative HF config
// (Qwen/Qwen3-0.6B config.json) and the driver's interpretation
// (driver/cuda/src/loader/hf_config.cpp). NOT assumed. Used to shape the Metal
// decoder-layer parity kernels.

namespace qwen3::arch {

constexpr int   HIDDEN            = 1024;
constexpr int   N_Q_HEADS         = 16;
constexpr int   N_KV_HEADS        = 8;      // GQA factor = 2
constexpr int   HEAD_DIM          = 128;    // explicit — NOT hidden/heads (=64)
constexpr int   Q_DIM             = N_Q_HEADS * HEAD_DIM;   // 2048
constexpr int   KV_DIM            = N_KV_HEADS * HEAD_DIM;  // 1024
constexpr int   INTERMEDIATE      = 3072;
constexpr float RMS_EPS           = 1e-6f;
constexpr float ROPE_THETA        = 1000000.0f;
constexpr int   VOCAB             = 151936;
constexpr int   N_LAYERS          = 28;
constexpr bool  USE_QK_NORM       = true;   // per-head RMSNorm(head_dim), pre-RoPE
constexpr bool  ATTENTION_BIAS    = false;
constexpr bool  TIE_WORD_EMB      = true;
constexpr bool  USE_SLIDING_WIN   = false;
constexpr bool  USE_SOFTCAP       = false;
constexpr int   GQA_FACTOR        = N_Q_HEADS / N_KV_HEADS;  // 2

}  // namespace qwen3::arch

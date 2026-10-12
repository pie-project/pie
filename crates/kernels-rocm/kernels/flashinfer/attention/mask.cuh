














#ifndef FLASHINFER_ATTENTION_MASK_CUH_
#define FLASHINFER_ATTENTION_MASK_CUH_

namespace flashinfer {

enum class MaskMode {
  kNone = 0U,
  kCausal = 1U,
  kCustom = 2U,
  kMultiItemScoring = 3U,
};

}

#endif

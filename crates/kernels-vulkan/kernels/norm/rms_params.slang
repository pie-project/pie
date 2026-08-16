// Shared host/shader ABI for the RMS-family kernels. The Slang half of
// `norm/rms_params.glsl`, identical to it: a struct of four-byte scalars is
// spelled the same in both languages, so this file is the GLSL one with the
// include guard renamed. It exists separately only so the two trees can be
// deleted independently as the port finishes.
//
// The Metal sibling is `norm/rms_params.h` and these are the same three structs
// with the same fields in the same order, because the same host builds both:
// a struct that drifted between the two backends would be a layout the driver
// fills one way and the shader reads the other, and nothing reports it.
//
// std430 is the layout, and the fields are all four bytes, so the packing is
// the C one and the two headers agree member for member. That is a property of
// THESE structs, not of std430 -- a `vec3` or a nested struct would not, which
// is why nothing here has one.

#ifndef PIE_VULKAN_RMS_PARAMS_SLANG
#define PIE_VULKAN_RMS_PARAMS_SLANG

struct RmsParams {
    float eps;
    uint axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
};

// `RmsParams` with the rotation's four scalars appended, for the fused
// `rms_rope`. Appended and not interleaved, and that order is an ABI rather
// than a preference: `driver-vulkan`'s `encode` mints this block as THE
// STATEMENT'S PARAMS RUN followed by whatever the routine packed in after it,
// which is the same shape `row_gather`'s `{ width, count }` has. The first
// five fields are therefore the five a norm statement already states, field
// for field, and the last four are the routine's own.
//
// It is one block and not a block plus a push range because `binding::params`
// can place a launch's scalars in one shape or the other and not both -- it
// asks push first, and a module answering with the right push size "is not
// also hiding a parameter buffer". A kernel needing both would be a kernel
// that cannot be given its scalars.
//
// There is no Metal or GLSL sibling for this one, because there is no Metal
// `rms_rope`. When there is, it goes beside the others here.
struct RmsRopeParams {
    float eps;
    uint axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
    // The distance between two tokens' rows, which is the whole token and not
    // the head: the base this kernel builds is two-level.
    uint row_pitch;
    // The channels that rotate, counted from zero. Not always the head --
    // gemma-4 rotates a quarter of each full-attention head.
    uint rotary;
    float scale;
    // `log2(theta)`, because the body raises two to it. Under `PIE_FREQS`
    // this field carries YaRN's `mscale` instead, which is the same
    // substitution `neox.slang` makes in its push block.
    float base_or_mscale;
};

struct VNormParams {
    float eps;
    uint axis_size;
};

struct GatedRmsParams {
    float eps;
    uint vd;
};

struct LayerScalarParams {
    uint hidden;
};

struct PleCombineParams {
    float inv_sqrt2;
    uint n;
};

struct SoftcapParams {
    float cap;
    uint n;
};

#endif  // PIE_VULKAN_RMS_PARAMS_SLANG

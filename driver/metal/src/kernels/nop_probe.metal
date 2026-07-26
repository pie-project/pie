// A dispatch that does nothing and touches nothing: the marginal cost of one
// more barrier-separated dispatch in the decode step, with no compute and no
// memory traffic mixed in.  Diagnostic only.
#include <metal_stdlib>
[[kernel]] void nop_probe(uint tid [[thread_position_in_grid]]) { (void)tid; }

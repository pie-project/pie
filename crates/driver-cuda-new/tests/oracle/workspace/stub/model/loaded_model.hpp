#pragma once
// Stub for csrc/src/model/loaded_model.hpp.
//
// `workspace.hpp` includes it and uses nothing from it — the include is there
// for `HfConfig`, which reaches it transitively. Empty on purpose: if the
// shipping header ever starts contributing something `workspace.cpp` needs,
// this file failing to provide it is the signal.
#include "model/config.hpp"

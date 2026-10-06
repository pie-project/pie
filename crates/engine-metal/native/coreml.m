// The CoreML side of the Neural Engine MLP split: load a compiled model and
// run it over fp16 rows that live in Metal shared buffers, without copies.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <string.h>

static void say(NSError *error, const char *what, char *err, int cap) {
  if (cap <= 0) return;
  NSString *text = error ? error.localizedDescription : @"";
  snprintf(err, (size_t)cap, "%s: %s", what, text.UTF8String);
}

// `function` picks one function of a multifunction model (NULL: the default).
// units: 0 = CPU + Neural Engine, 1 = CPU only, 2 = CPU + GPU.
void *pie_coreml_load(const char *path, const char *function, int units, char *err, int cap) {
  @autoreleasepool {
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    if (function && function[0]) {
      config.functionName = [NSString stringWithUTF8String:function];
    }
    config.computeUnits = units == 1 ? MLComputeUnitsCPUOnly
                        : units == 2 ? MLComputeUnitsCPUAndGPU
                                     : MLComputeUnitsCPUAndNeuralEngine;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError *error = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
    if (!model) {
      say(error, "load", err, cap);
      return NULL;
    }
    return (__bridge_retained void *)model;
  }
}

void pie_coreml_free(void *model) {
  if (model) {
    MLModel *owned = (__bridge_transfer MLModel *)model;
    (void)owned;
  }
}

static MLMultiArray *rows_of(void *base, long rows, long width, char *err, int cap) {
  NSError *error = nil;
  MLMultiArray *array = [[MLMultiArray alloc]
      initWithDataPointer:base
                    shape:@[ @(rows), @(width) ]
                 dataType:MLMultiArrayDataTypeFloat16
                  strides:@[ @(width), @1 ]
              deallocator:nil
                    error:&error];
  if (!array) say(error, "wrap", err, cap);
  return array;
}

// One prediction over `rows` x `width_in` fp16 rows at `input`, writing
// `rows` x `width_out` fp16 rows to `output`. Returns 0 on success.
int pie_coreml_predict(void *model, const char *in_name, void *input, long rows, long width_in,
                       const char *out_name, void *output, long width_out, char *err, int cap) {
  @autoreleasepool {
    MLModel *m = (__bridge MLModel *)model;
    MLMultiArray *x = rows_of(input, rows, width_in, err, cap);
    MLMultiArray *y = rows_of(output, rows, width_out, err, cap);
    if (!x || !y) return 1;
    NSString *xin = [NSString stringWithUTF8String:in_name];
    NSString *yout = [NSString stringWithUTF8String:out_name];
    NSError *error = nil;
    MLDictionaryFeatureProvider *features = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{xin : [MLFeatureValue featureValueWithMultiArray:x]}
                     error:&error];
    if (!features) {
      say(error, "features", err, cap);
      return 2;
    }
    MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
    options.outputBackings = @{yout : y};
    id<MLFeatureProvider> out = [m predictionFromFeatures:features options:options error:&error];
    if (!out) {
      say(error, "predict", err, cap);
      return 3;
    }
    MLMultiArray *got = [out featureValueForName:yout].multiArrayValue;
    if (!got) {
      say(nil, "the model has no such output", err, cap);
      return 4;
    }
    // CoreML may decline the backing (layout or type it cannot honour);
    // then the result lives in its own array and is copied over.
    if (got.dataPointer != output) {
      if (got.dataType != MLMultiArrayDataTypeFloat16 || got.count != rows * width_out) {
        say(nil, "the output is not the fp16 rows asked for", err, cap);
        return 5;
      }
      __block int ok = 1;
      [got getBytesWithHandler:^(const void *bytes, NSInteger size) {
        if ((long)size < rows * width_out * 2) { ok = 0; return; }
        memcpy(output, bytes, (size_t)(rows * width_out * 2));
      }];
      if (!ok) {
        say(nil, "the output is shorter than its rows", err, cap);
        return 6;
      }
    }
    return 0;
  }
}

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <string.h>

static void say(NSError *error, const char *what, char *err, int cap) {
  if (cap <= 0) return;
  NSString *text = error ? error.localizedDescription : @"";
  snprintf(err, (size_t)cap, "%s: %s", what, text.UTF8String);
}

void *pie_coreml_load(const char *path, const char *function, int units, char *err, int cap) {
  @autoreleasepool {
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    if (function && function[0]) {
      config.functionName = [NSString stringWithUTF8String:function];
    }
    config.computeUnits = units == 2 ? MLComputeUnitsCPUAndGPU : MLComputeUnitsCPUAndNeuralEngine;
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
    if (got.dataPointer != output) {
      if (got.dataType != MLMultiArrayDataTypeFloat16 || got.count != rows * width_out) {
        say(nil, "the output is not the fp16 rows asked for", err, cap);
        return 5;
      }
      NSArray<NSNumber *> *strides = got.strides;
      if (got.shape.count != 2 || got.shape[0].longValue != rows ||
          got.shape[1].longValue != width_out || strides[1].longValue != 1) {
        say(nil, "the output is not laid out as fp16 rows", err, cap);
        return 6;
      }
      const long pitch = strides[0].longValue;
      __block int ok = 1;
      [got getBytesWithHandler:^(const void *bytes, NSInteger size) {
        if (pitch < width_out || (long)size < ((rows - 1) * pitch + width_out) * 2) {
          ok = 0;
          return;
        }
        for (long r = 0; r < rows; r++) {
          memcpy((char *)output + r * width_out * 2, (const char *)bytes + r * pitch * 2,
                 (size_t)(width_out * 2));
        }
      }];
      if (!ok) {
        say(nil, "the output is shorter than its rows", err, cap);
        return 7;
      }
    }
    return 0;
  }
}

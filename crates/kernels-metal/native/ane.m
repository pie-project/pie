// TEMPORARY: calls Apple's private AppleNeuralEngine.framework, an undocumented
// interface. macOS updates may change or remove it; pie_ane_available() checks
// every class and method signature first, and when any is missing pie leaves
// the Neural Engine alone and the GPU runs the whole MLP. Replace once Apple
// ships a public API.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <string.h>
#include <sys/qos.h>

@protocol PieAneModel
+ (id)modelAtURL:(NSURL *)url key:(NSString *)key;
- (NSDictionary *)modelAttributes;
@end
@protocol PieAneClient
+ (id)sharedConnection;
- (BOOL)compileModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)compiledModelExistsFor:(id)model;
- (void)purgeCompiledModel:(id)model;
- (BOOL)loadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(unsigned)qos error:(NSError **)error;
- (BOOL)evaluateWithModel:(id)model options:(NSDictionary *)options request:(id)request qos:(unsigned)qos
                    error:(NSError **)error;
@end
@protocol PieAneSurface
+ (id)objectWithIOSurface:(IOSurfaceRef)surface;
@end
@protocol PieAneRequest
+ (id)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices outputs:(NSArray *)outputs
          outputIndices:(NSArray *)outputIndices weightsBuffer:(id)weights perfStats:(id)stats
         procedureIndex:(NSNumber *)procedure sharedEvents:(id)events transactionHandle:(NSNumber *)transaction;
- (void)setCompletionHandler:(void (^)(BOOL success, NSError *error))handler;
@end
@protocol PieAneEvents
+ (id)waitEventWithValue:(uint64_t)value sharedEvent:(id)event eventType:(uint64_t)type;
+ (id)signalEventWithValue:(uint64_t)value symbolIndex:(unsigned)symbol eventType:(int64_t)type sharedEvent:(id)event;
+ (id)sharedEventsWithSignalEvents:(NSArray *)signals waitEvents:(NSArray *)waits;
@end

typedef struct {
  const char *owner;
  char kind;
  const char *selector;
  const char *types;
} Signature;

// Every class and method the bridge calls, with the type encoding it expects.
// A framework that lacks or changed one is reported unavailable rather than
// called.
static const Signature kSignatures[] = {
    {"_ANEModel", '+', "modelAtURL:key:", "@@:@@"},
    {"_ANEModel", '-', "modelAttributes", "@@:"},
    {"_ANEClient", '+', "sharedConnection", "@@:"},
    {"_ANEClient", '-', "compileModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "compiledModelExistsFor:", "B@:@"},
    {"_ANEClient", '-', "purgeCompiledModel:", "v@:@"},
    {"_ANEClient", '-', "loadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "unloadModel:options:qos:error:", "B@:@@I^@"},
    {"_ANEClient", '-', "evaluateWithModel:options:request:qos:error:", "B@:@@@I^@"},
    {"_ANEIOSurfaceObject", '+', "objectWithIOSurface:", "@@:^{__IOSurface=}"},
    {"_ANERequest", '+',
     "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"
     "transactionHandle:",
     "@@:@@@@@@@@@"},
    {"_ANERequest", '-', "setCompletionHandler:", "v@:@?"},
    {"_ANESharedWaitEvent", '+', "waitEventWithValue:sharedEvent:eventType:", "@@:Q@Q"},
    {"_ANESharedSignalEvent", '+', "signalEventWithValue:symbolIndex:eventType:sharedEvent:", "@@:QIq@"},
    {"_ANESharedEvents", '+', "sharedEventsWithSignalEvents:waitEvents:", "@@:@@"},
};

static const unsigned kQos = QOS_CLASS_DEFAULT;

static void say(char *err, int cap, NSString *text) {
  if (err && cap > 0) snprintf(err, (size_t)cap, "%s", text.UTF8String);
}

static id<PieAneClient> gClient;
static Class gModel, gSurface, gEvents, gSignal, gWait, gRequest;
static dispatch_queue_t gQueue;
static NSString *gUnavailable;

static NSString *resolve(void) {
  if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
    const char *reason = dlerror();
    return [NSString stringWithFormat:@"AppleNeuralEngine does not load: %s", reason ? reason : "no reason"];
  }
  for (size_t i = 0; i < sizeof kSignatures / sizeof kSignatures[0]; i++) {
    const Signature *s = &kSignatures[i];
    Class owner = NSClassFromString(@(s->owner));
    if (!owner) return [NSString stringWithFormat:@"AppleNeuralEngine lacks %s", s->owner];
    SEL selector = sel_registerName(s->selector);
    Method method = s->kind == '+' ? class_getClassMethod(owner, selector) : class_getInstanceMethod(owner, selector);
    if (!method) return [NSString stringWithFormat:@"AppleNeuralEngine lacks %c[%s %s]", s->kind, s->owner, s->selector];
    const char *encoding = method_getTypeEncoding(method);
    NSMutableString *types = [NSMutableString string];
    for (const char *c = encoding ? encoding : ""; *c; c++)
      if (*c < '0' || *c > '9') [types appendFormat:@"%c", *c];
    if (![types isEqualToString:@(s->types)])
      return [NSString stringWithFormat:@"AppleNeuralEngine changed %c[%s %s]", s->kind, s->owner, s->selector];
  }
  gModel = NSClassFromString(@"_ANEModel");
  gSurface = NSClassFromString(@"_ANEIOSurfaceObject");
  gEvents = NSClassFromString(@"_ANESharedEvents");
  gSignal = NSClassFromString(@"_ANESharedSignalEvent");
  gWait = NSClassFromString(@"_ANESharedWaitEvent");
  gRequest = NSClassFromString(@"_ANERequest");
  Class client = NSClassFromString(@"_ANEClient");
  id shared = [(Class<PieAneClient>)client sharedConnection];
  if (![shared isKindOfClass:client]) return @"AppleNeuralEngine lacks a shared connection";
  gClient = shared;
  gQueue = dispatch_queue_create("pie.ane", DISPATCH_QUEUE_SERIAL);
  return nil;
}

int pie_ane_available(char *err, int cap) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    @autoreleasepool {
      @try {
        gUnavailable = resolve();
      } @catch (NSException *exception) {
        gUnavailable = [NSString stringWithFormat:@"interface lookup raised %@: %@", exception.name, exception.reason];
      }
    }
  });
  if (gUnavailable) {
    say(err, cap, gUnavailable);
    return 0;
  }
  return 1;
}

typedef struct {
  void *surface;
  void *base;
  uint64_t bytes;
  uint32_t stride;
} PieAneSurface;

// An IOSurface of `rows` rows of `width` elements: rows stride-aligned to 64
// bytes, the allocation to 16 KB, in the L008 (int8) or L00h (fp16) format.
int pie_ane_surface(uint32_t rows, uint32_t width, int int8, PieAneSurface *out, char *err, int cap) {
  @autoreleasepool {
    const uint32_t element = int8 ? 1 : 2;
    const uint32_t stride = (width * element + 63) / 64 * 64;
    const uint64_t bytes = ((uint64_t)stride * rows + 16383) / 16384 * 16384;
    NSDictionary *properties = @{
      (id)kIOSurfaceWidth : @(width),
      (id)kIOSurfaceHeight : @(rows),
      (id)kIOSurfaceBytesPerElement : @(element),
      (id)kIOSurfaceBytesPerRow : @(stride),
      (id)kIOSurfaceAllocSize : @(bytes),
      (id)kIOSurfacePixelFormat : @(int8 ? 0x4c303038 : 0x4c303068),
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (!surface) {
      say(err, cap, @"IOSurface creation failed");
      return 0;
    }
    if (IOSurfaceGetBytesPerRow(surface) != stride || IOSurfaceGetAllocSize(surface) != bytes) {
      CFRelease(surface);
      say(err, cap, @"IOSurface took another row stride or size");
      return 0;
    }
    out->surface = surface;
    out->base = IOSurfaceGetBaseAddress(surface);
    out->bytes = bytes;
    out->stride = stride;
    return 1;
  }
}

void pie_ane_surface_free(void *surface) {
  if (surface) CFRelease((IOSurfaceRef)surface);
}

typedef struct {
  __strong id model;
  NSMutableArray<NSString *> *functions;
  NSMutableArray<NSArray<NSString *> *> *inputs;
  NSMutableArray<NSArray<NSNumber *> *> *inputSymbols;
  NSMutableArray<NSNumber *> *outputSymbol;
  BOOL loaded;
} Program;

// Runs `body` on the bridge's serial queue, turning a thrown exception into a
// failure message.
static BOOL run(dispatch_block_t body, NSString **failure) {
  __block NSString *message = nil;
  dispatch_sync(gQueue, ^{
    @autoreleasepool {
      @try {
        body();
      } @catch (NSException *exception) {
        message = [NSString stringWithFormat:@"%@: %@", exception.name, exception.reason];
      }
    }
  });
  if (message) *failure = message;
  return message == nil;
}

// Reads the loaded program's procedures (name, input symbols, the one output
// symbol) out of its model attributes.
static NSString *describe(Program *program) {
  NSDictionary *attributes = [program->model modelAttributes];
  if (![attributes isKindOfClass:NSDictionary.class]) return @"the program does not describe itself";
  NSDictionary *description = attributes[@"ANEFModelDescription"];
  NSArray *symbols = description[@"kANEFModelInputSymbolsArrayKey"];
  NSDictionary *functions = description[@"kANEFModelProcedureNameToIDMapKey"];
  NSArray *entries = description[@"ANEFModelProcedures"];
  if (![description isKindOfClass:NSDictionary.class] || ![symbols isKindOfClass:NSArray.class] ||
      ![functions isKindOfClass:NSDictionary.class] || ![entries isKindOfClass:NSArray.class] ||
      entries.count != functions.count)
    return @"the program does not describe its procedures";
  NSUInteger count = entries.count;
  program->functions = [NSMutableArray arrayWithCapacity:count];
  program->inputs = [NSMutableArray arrayWithCapacity:count];
  program->inputSymbols = [NSMutableArray arrayWithCapacity:count];
  program->outputSymbol = [NSMutableArray arrayWithCapacity:count];
  for (NSUInteger i = 0; i < count; i++) {
    [program->functions addObject:@""];
    [program->inputs addObject:@[]];
    [program->inputSymbols addObject:@[]];
    [program->outputSymbol addObject:@0];
  }
  for (NSString *function in functions) {
    NSNumber *index = functions[function];
    if (![index isKindOfClass:NSNumber.class] || index.unsignedIntegerValue >= count)
      return @"the program names an unknown procedure";
    program->functions[index.unsignedIntegerValue] = function;
  }
  for (NSDictionary *entry in entries) {
    NSNumber *index = entry[@"ANEFModelProcedureID"];
    NSArray *outputs = entry[@"ANEFModelOutputSymbolIndexArray"];
    NSArray *inputs = entry[@"ANEFModelInputSymbolIndexArray"];
    if (![index isKindOfClass:NSNumber.class] || index.unsignedIntegerValue >= count ||
        ![outputs isKindOfClass:NSArray.class] || outputs.count != 1 || ![inputs isKindOfClass:NSArray.class])
      return @"the program has a procedure of other than one output";
    NSMutableArray *names = [NSMutableArray array];
    for (NSNumber *symbol in inputs) {
      if (![symbol isKindOfClass:NSNumber.class] || symbol.unsignedIntegerValue >= symbols.count)
        return @"the program names an unknown input";
      [names addObject:symbols[symbol.unsignedIntegerValue]];
    }
    program->inputs[index.unsignedIntegerValue] = names;
    program->inputSymbols[index.unsignedIntegerValue] = inputs;
    program->outputSymbol[index.unsignedIntegerValue] = outputs[0];
  }
  return nil;
}

// Compiles (unless a compiled copy exists) and loads the MIL program in
// `directory` (`model.mil` beside `weights.bin`), keyed by `key`. A compiled
// copy that no longer loads is purged and compiled again.
void *pie_ane_program(const char *directory, const char *key, char *err, int cap) {
  if (!pie_ane_available(err, cap)) return NULL;
  Program *program = (Program *)calloc(1, sizeof(Program));
  __block NSString *problem = nil;
  NSString *failure = nil;
  BOOL ran = run(^{
    program->model = [gModel modelAtURL:[NSURL fileURLWithPath:@(directory) isDirectory:YES] key:@(key)];
    if (!program->model) {
      problem = @"model creation failed";
      return;
    }
    NSError *error = nil;
    BOOL compiled = [gClient compiledModelExistsFor:program->model];
    NSDictionary *options = @{@"kANEFModelType" : @"kANEFModelMIL", @"kANEFNetPlistFilenameKey" : @"model.mil"};
    if (!compiled && ![gClient compileModel:program->model options:options qos:kQos error:&error]) {
      problem = [NSString stringWithFormat:@"compilation failed: %@", error.description];
      return;
    }
    if (![gClient loadModel:program->model options:@{} qos:kQos error:&error]) {
      if (!compiled) {
        problem = [NSString stringWithFormat:@"load failed: %@", error.description];
        return;
      }
      [gClient purgeCompiledModel:program->model];
      error = nil;
      if (![gClient compileModel:program->model options:options qos:kQos error:&error] ||
          ![gClient loadModel:program->model options:@{} qos:kQos error:&error]) {
        problem = [NSString stringWithFormat:@"load failed: %@", error.description];
        return;
      }
    }
    program->loaded = YES;
    problem = describe(program);
  }, &failure);
  if (!ran || problem) {
    say(err, cap, failure ? failure : problem);
    if (program->loaded) {
      NSString *ignored = nil;
      run(^{
        [gClient unloadModel:program->model options:@{} qos:kQos error:nil];
      }, &ignored);
    }
    program->model = nil;
    free(program);
    return NULL;
  }
  return program;
}

void pie_ane_program_free(void *handle) {
  Program *program = (Program *)handle;
  if (!program) return;
  if (program->loaded) {
    NSString *ignored = nil;
    run(^{
      [gClient unloadModel:program->model options:@{} qos:kQos error:nil];
    }, &ignored);
  }
  program->model = nil;
  program->functions = nil;
  program->inputs = nil;
  program->inputSymbols = nil;
  program->outputSymbol = nil;
  free(program);
}

int pie_ane_procedure(void *handle, const char *function) {
  Program *program = (Program *)handle;
  NSUInteger index = [program->functions indexOfObject:@(function)];
  return index == NSNotFound ? -1 : (int)index;
}

int pie_ane_input_count(void *handle, int procedure) {
  return (int)((Program *)handle)->inputs[procedure].count;
}

const char *pie_ane_input_name(void *handle, int procedure, int input) {
  return ((Program *)handle)->inputs[procedure][input].UTF8String;
}

typedef struct {
  NSNumber *procedure;
  NSArray *inputs, *inputIndices, *outputs, *outputIndices;
} Binding;

// Wraps the procedure's input surfaces and its output surface as the objects a
// request takes, in the procedure's own input order.
void *pie_ane_bind(void *handle, int procedure, void *const *inputs, int count, void *output, char *err, int cap) {
  Program *program = (Program *)handle;
  if (count != (int)program->inputs[procedure].count) {
    say(err, cap, @"the binding's input count is not the procedure's");
    return NULL;
  }
  @autoreleasepool {
    NSMutableArray *objects = [NSMutableArray arrayWithCapacity:count];
    for (int i = 0; i < count; i++) {
      id object = [gSurface objectWithIOSurface:(IOSurfaceRef)inputs[i]];
      if (!object) {
        say(err, cap, @"surface object creation failed");
        return NULL;
      }
      [objects addObject:object];
    }
    id out = [gSurface objectWithIOSurface:(IOSurfaceRef)output];
    if (!out) {
      say(err, cap, @"surface object creation failed");
      return NULL;
    }
    Binding *binding = (Binding *)calloc(1, sizeof(Binding));
    binding->procedure = @(procedure);
    binding->inputs = [objects copy];
    binding->inputIndices = [program->inputSymbols[procedure] copy];
    binding->outputs = @[ out ];
    binding->outputIndices = @[ program->outputSymbol[procedure] ];
    return binding;
  }
}

void pie_ane_binding_free(void *handle) {
  Binding *binding = (Binding *)handle;
  if (!binding) return;
  binding->procedure = nil;
  binding->inputs = nil;
  binding->inputIndices = nil;
  binding->outputs = nil;
  binding->outputIndices = nil;
  free(binding);
}

typedef void (*PieAneReport)(void *context, int success);

// Queues one evaluation of `bound` that waits for `event` to reach `wait` and
// signals `signal` when done. `report` fires exactly once, from the completion
// handler or from the failure path. Returns 1 when queued, 2 when the request
// was handed over but evaluation failed (`report` has fired), 0 when nothing
// was handed over (`report` will not fire).
int pie_ane_enqueue(void *handle, void *bound, void *waitsOn, void *signals, uint64_t wait, uint64_t signal,
                    PieAneReport report, void *context, char *err, int cap) {
  Program *program = (Program *)handle;
  Binding *binding = (Binding *)bound;
  __block int fired = 0;
  BOOL handed = NO;
  @autoreleasepool {
    @try {
      id ready = (__bridge id)waitsOn;
      id done = (__bridge id)signals;
      SEL port = NSSelectorFromString(@"eventPort");
      if (![ready respondsToSelector:port] || ![done respondsToSelector:port]) {
        say(err, cap, @"the Neural Engine cannot share this Metal event");
        return 0;
      }
      id signalEvent = [gSignal signalEventWithValue:signal symbolIndex:0 eventType:0 sharedEvent:done];
      id waitEvent = [gWait waitEventWithValue:wait sharedEvent:ready eventType:0];
      id events = [gEvents sharedEventsWithSignalEvents:@[ signalEvent ] waitEvents:@[ waitEvent ]];
      id request = [gRequest requestWithInputs:binding->inputs
                                  inputIndices:binding->inputIndices
                                       outputs:binding->outputs
                                 outputIndices:binding->outputIndices
                                 weightsBuffer:nil
                                     perfStats:nil
                                procedureIndex:binding->procedure
                                  sharedEvents:events
                             transactionHandle:nil];
      if (!events || !request) {
        say(err, cap, @"request creation failed");
        return 0;
      }
      [request setCompletionHandler:^(BOOL success, NSError *error __attribute__((unused))) {
        if (!__atomic_exchange_n(&fired, 1, __ATOMIC_ACQ_REL)) report(context, success ? 1 : 0);
      }];
      handed = YES;
      NSError *error = nil;
      if (![gClient evaluateWithModel:program->model options:@{} request:request qos:kQos error:&error]) {
        say(err, cap, [NSString stringWithFormat:@"evaluation failed: %@", error.description]);
        if (!__atomic_exchange_n(&fired, 1, __ATOMIC_ACQ_REL)) report(context, 0);
        return 2;
      }
      return 1;
    } @catch (NSException *exception) {
      say(err, cap, [NSString stringWithFormat:@"evaluation raised %@: %@", exception.name, exception.reason]);
      if (handed) {
        if (!__atomic_exchange_n(&fired, 1, __ATOMIC_ACQ_REL)) report(context, 0);
        return 2;
      }
      return 0;
    }
  }
}

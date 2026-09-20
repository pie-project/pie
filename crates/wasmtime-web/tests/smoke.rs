mod common;

use wasmtime::Store;
use wasmtime::component::{Component, Linker};

const GUEST: &str = r#"
(component
  (import "wasi:io/error@0.2.12" (instance $ioerr
    (export "error" (type (sub resource)))))
  (alias export $ioerr "error" (type $error))

  (import "wasi:io/poll@0.2.12" (instance $poll
    (export "pollable" (type $p (sub resource)))
    (export "poll" (func (param "in" (list (borrow $p))) (result (list u32))))))
  (alias export $poll "pollable" (type $pollable))

  (import "wasi:io/streams@0.2.12" (instance $streams
    (export "error" (type $e (eq $error)))
    (export "output-stream" (type $os (sub resource)))
    (type $se (variant (case "last-operation-failed" (own $e)) (case "closed")))
    (export "stream-error" (type $stream-error (eq $se)))
    (export "[method]output-stream.blocking-write-and-flush"
      (func (param "self" (borrow $os)) (param "contents" (list u8))
            (result (result (error $stream-error)))))))
  (alias export $streams "output-stream" (type $output-stream))

  (import "wasi:cli/stdout@0.2.12" (instance $stdout
    (export "output-stream" (type $os (eq $output-stream)))
    (export "get-stdout" (func (result (own $os))))))

  (import "wasi:clocks/monotonic-clock@0.2.12" (instance $clock
    (export "pollable" (type $p (eq $pollable)))
    (export "now" (func (result u64)))
    (export "subscribe-duration" (func (param "when" u64) (result (own $p))))))

  (import "wasi:random/random@0.2.12" (instance $random
    (export "get-random-bytes" (func (param "len" u64) (result (list u8))))))

  (core module $mem
    (memory (export "memory") 1)
    (global $next (mut i32) (i32.const 4096))
    (func (export "cabi_realloc") (param i32 i32 i32 i32) (result i32)
      (local $p i32)
      (local.set $p (global.get $next))
      (global.set $next (i32.add (global.get $next) (i32.add (local.get 3) (i32.const 16))))
      (local.get $p)))
  (core instance $memi (instantiate $mem))
  (alias core export $memi "memory" (core memory $memory))
  (alias core export $memi "cabi_realloc" (core func $realloc))

  (core func $get_stdout (canon lower (func $stdout "get-stdout")))
  (core func $write (canon lower (func $streams "[method]output-stream.blocking-write-and-flush") (memory $memory)))
  (core func $drop_stream (canon resource.drop $output-stream))
  (core func $now (canon lower (func $clock "now")))
  (core func $sleep (canon lower (func $clock "subscribe-duration")))
  (core func $drop_pollable (canon resource.drop $pollable))
  (core func $poll (canon lower (func $poll "poll") (memory $memory) (realloc $realloc)))
  (core func $random (canon lower (func $random "get-random-bytes") (memory $memory) (realloc $realloc)))

  (core module $main
    (import "env" "memory" (memory 1))
    (import "stdout" "get-stdout" (func $get_stdout (result i32)))
    (import "streams" "write" (func $write (param i32 i32 i32 i32)))
    (import "streams" "drop" (func $drop_stream (param i32)))
    (import "clock" "now" (func $now (result i64)))
    (import "clock" "subscribe-duration" (func $sleep (param i64) (result i32)))
    (import "poll" "drop" (func $drop_pollable (param i32)))
    (import "poll" "poll" (func $poll (param i32 i32 i32)))
    (import "random" "get-random-bytes" (func $random (param i64 i32)))
    (data (i32.const 0) "hello from wasm\n")
    (func (export "run")
      (local $stream i32) (local $t0 i64) (local $pollable i32)
      ;; stdout
      (local.set $stream (call $get_stdout))
      (call $write (local.get $stream) (i32.const 0) (i32.const 16) (i32.const 512))
      (if (i32.load8_u (i32.const 512)) (then unreachable))
      (call $drop_stream (local.get $stream))
      ;; sleep 1ms through subscribe-duration + poll
      (local.set $t0 (call $now))
      (local.set $pollable (call $sleep (i64.const 1000000)))
      (i32.store (i32.const 600) (local.get $pollable))
      (call $poll (i32.const 600) (i32.const 1) (i32.const 608))
      (if (i32.ne (i32.load (i32.const 612)) (i32.const 1)) (then unreachable))
      (call $drop_pollable (local.get $pollable))
      (if (i64.lt_u (i64.sub (call $now) (local.get $t0)) (i64.const 1000000)) (then unreachable))
      ;; random
      (call $random (i64.const 8) (i32.const 700))
      (if (i32.ne (i32.load (i32.const 704)) (i32.const 8)) (then unreachable))))

  (core instance $maini (instantiate $main
    (with "env" (instance (export "memory" (memory $memory))))
    (with "stdout" (instance (export "get-stdout" (func $get_stdout))))
    (with "streams" (instance (export "write" (func $write)) (export "drop" (func $drop_stream))))
    (with "clock" (instance (export "now" (func $now)) (export "subscribe-duration" (func $sleep))))
    (with "poll" (instance (export "drop" (func $drop_pollable)) (export "poll" (func $poll))))
    (with "random" (instance (export "get-random-bytes" (func $random))))))

  (func (export "run") (canon lift (core func $maini "run")))
)
"#;

#[test]
fn stdout_clock_random_and_timer_reach_the_host() {
    let engine = common::engine();
    let component = Component::new(&engine, GUEST).expect("guest component");
    let mut linker = Linker::<common::State>::new(&engine);
    wasmtime_web::add_to_linker(&mut linker).expect("add_to_linker");

    let (state, stdout, _stderr) = common::state();
    let mut store = Store::new(&engine, state);
    common::block_on(async {
        let instance = linker
            .instantiate_async(&mut store, &component)
            .await
            .expect("instantiate");
        let run = instance
            .get_typed_func::<(), ()>(&mut store, "run")
            .expect("run export");
        run.call_async(&mut store, ()).await.expect("run");
    });

    assert_eq!(stdout.lock().unwrap().as_slice(), b"hello from wasm\n");
}

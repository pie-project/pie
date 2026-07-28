# `compiler/` 코드 이슈 목록 — 유지보수성 & code smell

Date: 2026-07-28
Scope: `compiler/{ir,plan,eval,codegen,dsl,tests}` — 25,818 LOC Rust
(`src/` 19,049 + 전용 테스트 파일 6,769)
Baseline: `89d05b10e` (`tasks/ptir-2/agents/alpha`)
Purpose: 전면 리팩토링 착수 전 현행 코드의 구조적 문제 정리. `ptir-refactor.md`가
*무엇을 어디로 옮겼는가*를 다룬다면, 이 문서는 *옮기고 난 뒤에도 남아 있는 것*을 다룬다.

---

## 0. 요약

표면 건강 지표는 매우 좋다. 문제는 전부 **구조**에 있다.

| 지표 | 값 | 판정 |
|---|---|---|
| clippy 경고 (기본 lint, `--all-targets`) | ~~3건~~ → **실측 29건** (compiler 한정). 워크스페이스 전체는 277건 | ⚠️ |
| 테스트 | 152건 → **165건**, 전부 통과 | ✅ |
| `unsafe` 블록 | 0 | ✅ |
| `TODO`/`FIXME`/`HACK`/`XXX` | 0 | ⚠️ (§m10) |
| `no_std` / `std` 분리 | ~~3개~~ → 4~5곳. cfg soup은 없으나 **`--no-default-features`가 컴파일되지 않았음** | ❌ |
| codegen 순수성 (`HashMap` / I/O / env / time) | 전부 없음 — `Plan -> String` 주장 성립 | ✅ |
| 100줄 초과 함수 (clippy `too_many_lines`) | **17개** (compiler 내), 최대 643줄 | ❌ |
| 인지 복잡도 25 초과 | 4개 (최대 39) | ❌ |
| 비테스트 코드의 `unwrap`/`expect`/`panic!`/`unreachable!` | ~~190건~~ → **68건** (부록 A의 grep이 `#[cfg(test)]`를 포함했다) | ⚠️ |

한 줄 요약:

> **컴파일러가 잡아줄 수 있었던 정합성을 사람이 손으로 지키고 있다.**
> 위험은 "코드가 지저분하다"가 아니라 "테이블 6벌 중 하나가 어긋나면 조용히
> 잘못된 커널이 나간다"이다.

---

## 0-A. 검증 결과 (2026-07-29, baseline `c6fcd1b0f`)

이 문서의 주장을 코드로 하나씩 재현해 봤다. **다섯 건이 반증됐고, 문서에 없던
CRITICAL이 더 나왔다.** 아래 표가 정본이고, 본문의 개별 항목에는 인라인으로
정정 표시를 달았다. 리팩토링 착수 시 §1의 CRITICAL 목록을 그대로 신뢰하지 말 것 —
실제보다 과장돼 있다.

### 반증된 것 (고치지 않음)

| 항목 | 문서의 주장 | 실측 |
|---|---|---|
| C3 | 두 구현이 드리프트했다 | 완전 일치. 중복은 사실이나 드리프트는 없음 |
| M4 | 미localize 시그니처가 캐시 키를 오염 | 미localize 경로 자체가 없음. capacity만 다른 두 트레이스가 서로 다른 서명을 냄 |
| M17 | `Family`로 4벌을 치환 | 치환하면 nucleus 융합이 깨짐. 네 곳은 서로 다른 질문에 답하고 있음 |
| m4 | live bug | 도달 불가 |
| P3(부록 외) | FNV-1a 64비트가 캐시 키의 전부 | 두 캐시 모두 전체 바이트를 비교. 해시는 인덱스지 신원이 아님 |

### 문서에 없던 CRITICAL (전부 수정 완료)

- **CUDA nucleus-skip이 소비자 검사 없이 노드를 삭제** → 조용한 오답
- **Metal이 *모든* intrinsic을 logits 버퍼에 바인딩** → 조용한 오답
- **C++ 오라클이 삭제됨**(`32c2a4a09`) → 골든이 재유도 불가능한 기록물로 전락.
  `PTIR_REGEN=1`이 `#` 헤더는 보존한 채 본문만 갈아치워 **출처를 세탁**하고 있었음
- **`pie-ir --no-default-features` 파손** — §4-1이 "유지해야 할 것"으로 적어둔 항목이
  애초에 컴파일되지 않았음. 선언만 있고 아무도 빌드하지 않은 `no_std`
- **`decode_bound`가 readiness phase를 미검증으로 통과** (M7의 실체)
- **`bounded_count`가 입력 1바이트당 op 1개를 허용** — 게스트 입력 + 전역 락 = DoS
- **`ExecutableCacheKey` 호출자 0** — 드라이버가 손으로 다른 키를 만들고 있었음.
  드리프트가 아니라 아무도 안 쓴 설계

### 코퍼스가 가려온 것

기존 코퍼스(골든 19 + synthetic 4)의 실제 도달 범위는 **op 38/55, intrinsic 5/8,
schedule 3/4, stage 3/4**였다. 미도달분을 채우자 hidden/layer/attn_score 3케이스가
곧바로 Metal 바인딩 버그를 드러냈다 — 위 두 번째 CRITICAL은 가설이 아니었다.

---

### 심각도 정의

| 등급 | 의미 |
|---|---|
| **CRITICAL** | 드리프트/무결성 위험. 조용히 틀린 결과를 낼 수 있거나, 리팩토링을 실질적으로 막는다. |
| **MAJOR** | 실제 유지보수 부담. 기능 추가 비용을 반복적으로 올린다. |
| **MINOR** | 정리 대상. 위험은 낮지만 신뢰도를 깎는다. |

---

## 1. CRITICAL

### C1. Op 정의에 단일 진실 공급원이 없다 (shotgun surgery)

**증상.** op 하나를 추가하려면 최소 **9~11곳**을 손으로 고쳐야 한다. 어느 하나를
빠뜨려도 컴파일은 되고, 어긋난 사실은 런타임에 드러난다.

| # | 위치 | 추가해야 하는 것 |
|---|---|---|
| 1 | `compiler/ir/src/op.rs:107` `enum Op` | variant |
| 2 | `compiler/ir/src/op.rs:307` `Op::result_count()` | 결과 개수 |
| 3 | `compiler/ir/src/op.rs:317` `Op::operands()` | 피연산자 추출 |
| 4 | `compiler/ir/src/op.rs:390` `Op::map_operands()` | 피연산자 재작성 (3번의 가변 사본) |
| 5 | `compiler/ir/src/op.rs:476` `Op::tag()` | wire tag |
| 6 | `compiler/ir/src/op.rs:572` `OP_TABLE` | 선언적 행 (tag/name/family/arity/results) |
| 7 | `compiler/ir/src/container.rs:260` `encode_op()` | 인코딩 |
| 8 | `compiler/ir/src/container.rs:695` `decode_op()` | 디코딩 |
| 9 | `compiler/ir/src/infer.rs:160` `infer()` | shape/dtype 규칙 |
| 10 | `compiler/eval/src/interp.rs:1016` `eval_op()` | 의미론 |
| 11 | `compiler/ir/src/op.rs:975` 테스트의 `reps` 배열 | 대표 인스턴스 |

**핵심 모순.** `op.rs:3`의 모듈 문서는 `OP_TABLE`을 두고
*"the single source of truth for op ids, names, families, and arities"*라고 선언한다.
그러나 **어떤 프로덕션 함수도 `OP_TABLE`을 소비하지 않는다.** 실제 소비처는
`codegen/src/header.rs:101,117`(C 헤더 생성)과 `metal/validate.rs:229`,
`cuda/singleton.rs:30`뿐이고, `result_count()`/`operands()`/`tag()`는 같은 사실을
match 문으로 **재선언**한다. 테이블은 진실의 원천이 아니라 네 번째 사본이다.

**현재 방어선.** `ir/src/op.rs:971` `table_matches_op_metadata` 테스트가
`reps.len() == OP_TABLE.len()`으로 누락을 잡고, tag/results/arity/`map_operands`
일치를 검사한다. 실제로 잘 작동하지만 — 이 `reps` 배열 자체가 손으로 유지되는
**11번째 사본**이다. 그리고 이 테스트는 `encode/decode` 왕복이나 `infer`/`eval_op`의
커버리지는 검사하지 않는다.

**영향.** 새 op의 비용이 높고, 리뷰 표면이 넓으며, 머지 충돌이 잦다. `plan`/`codegen`
쪽 사본(C2)까지 합치면 op 하나가 15곳 이상을 건드린다.

**처방.** `OP_TABLE`을 선언적 입력으로 승격하고, 매크로(또는 build script)로
enum variant · `tag()` · arity 함수 · encode/decode 스켈레톤을 생성한다.
편집 지점을 1~2곳으로 줄이고, 나머지는 컴파일러가 강제하게 한다.
이것이 리팩토링의 1순위다.

---

### C2. wire tag가 6개 이상의 파일에 손으로 복사돼 있다

**증상.** `ir`이 정의한 op tag를 다운스트림이 import하지 않고 **재선언**한다.
총 51개의 `const OP_*` / `INTR_*` 선언이 흩어져 있다.

```
compiler/codegen/src/cuda/fused.rs:42-69      OP_EXP..OP_INTRINSIC_VAL  (28개)
compiler/codegen/src/metal/fused.rs:23-27     OP_PIVOT_THRESHOLD, OP_CHAN_{TAKE,READ,PUT}, OP_INTRINSIC_VAL
compiler/codegen/src/metal/topk.rs:17         OP_TOP_K
compiler/codegen/src/metal/validate.rs:24-26  OP_PIVOT_THRESHOLD, OP_KERNEL_CALL, OP_SINK_CALL
compiler/codegen/src/launch.rs:38-48          OP_CONST, OP_CHAN_*, OP_INTRINSIC_VAL, INTR_*
compiler/plan/src/compile.rs:2085+            raw hex (상수명조차 없음)
```

중복 횟수:

| 상수 | 값 | 선언 횟수 |
|---|---|---|
| `OP_CHAN_TAKE` | `0x90` | **4** |
| `OP_CHAN_READ` | `0x91` | **4** |
| `OP_CHAN_PUT` | `0x92` | **4** |
| `OP_INTRINSIC_VAL` | `0xA0` | 3 |
| `OP_PIVOT_THRESHOLD` | `0x58` | 3 |
| `INTR_LOGITS` / `INTR_MTP_LOGITS` | `0` / `1` | 3 |

`plan/src/compile.rs:2083` `scan_planned_op`은 상수조차 없이 원시 16진 범위
(`0x01..=0x06`, `0x10..=0x1D`, …)로 분기한다. 읽는 사람은 다른 crate의 테이블을
대조해야 한다. `codegen/src/launch.rs:567` `grouped_supported_tag`도 같은 문제 —
`0x01..=0x07 | 0x10..=0x20 | ...` 범위 리터럴이라 새 tag가 틈에 떨어지면
**조용히 미지원 처리**된다.

**영향.** 16진수 한 자리 오타 = 조용한 오코드 생성. 지금은 우연히 전부 일치하지만
이를 강제하는 장치가 없다.

**처방.** `pie_ir::op`에 `pub mod tags { pub const CHAN_TAKE: u8 = 0x90; ... }`를
`OP_TABLE`에서 파생시켜 두고, 다운스트림은 전부 import. 범위 매칭은 `OP_TABLE`의
`family` 조회로 교체.

---

### C3. composed op이 두 곳에 독립 구현돼 있고, "정본" 쪽은 죽어 있다

> **정정 (REFUTED).** 두 구현의 op 시퀀스는 오늘 **완전히 일치**한다. 중복은
> 사실이고 통합할 가치는 있지만, "이미 갈라졌다"는 전제는 틀렸다. CRITICAL이
> 아니라 MAJOR다.

**증상.** `softmax` / `log_softmax` / `l2norm` / `mask_apply` / `gumbel`의 op 시퀀스가
두 곳에 각각 작성돼 있다.

| | 정본이라 주장하는 쪽 | 실제 사용자 경로 |
|---|---|---|
| 위치 | `compiler/ir/src/expand.rs:32-91` | `compiler/dsl/src/value.rs:639-700` |
| 형태 | `fn softmax(ops: &mut Vec<Op>, x, shape) -> ValueId` | `fn softmax(x: impl AsTensor) -> Tensor` |
| 타입 추적 | 없음 (`infer`에 위임) | 있음 (`ValueType`을 손으로 부착) |

`dsl/src/value.rs:638`의 주석이 그대로 자백한다:

```rust
// -- normalize (echo's expand.rs expansions, type-tracked) --
```

**결정적 문제.** `ir/expand.rs`의 프로덕션 호출자는 **0명**이다. 호출처는 전부
테스트뿐이다:

```
compiler/ir/src/validate.rs:868-870     (#[cfg(test)] 픽스처)
compiler/tests/tests/common/traces.rs:88,89,179
compiler/tests/tests/ptir_golden.rs:926,1270,1271
```

즉 사용자가 실제로 타는 경로는 dsl의 사본이고, `ir/expand.rs`는 골든 픽스처
빌더로만 살아 있다. **두 정의가 일치하는지 검사하는 테스트는 없다.**
`expand.rs`의 수치 안정성 수정이 dsl에 반영되지 않아도 골든은 전부 통과한다.

**영향.** `softmax` / `l2norm`은 프로덕션 코드 경로다. 조용한 의미론 드리프트 위험.

**처방.** `ir/expand.rs`를 유일한 정의로 만들고 dsl이 이를 호출하도록 한다.
타입 부착은 `infer::body_types`로 위임하면 dsl의 손 계산이 사라진다.
전환 전이라면 최소한 "두 확장이 같은 op 시퀀스를 낸다"는 파리티 테스트를 추가한다.

---

### C4. 바이트 `Reader`가 3중 중복

**증상.** 구조·메서드가 사실상 동일하고 에러 타입만 다른 리더가 셋 있다.

| 위치 | 타입 | 메서드 |
|---|---|---|
| `compiler/ir/src/container.rs:518` | `Reader<'a>` | `u8 u16 u32 take bounded_count` |
| `compiler/plan/src/sidecar.rs:144` | `Reader<'a>` | `remaining take u8 u16 u32 u64 bounded_count` |
| `compiler/plan/src/compile.rs:1990` | `PlanReader<'a>` | `remaining take u8 u16 u32 u64 bounded_count` |

약 130줄의 중복. 경계 검사·오버플로 검사(`checked_add`)·할당 폭탄 방어
(`bounded_count`) 로직이 세 벌 존재하므로, 한 곳의 보안 수정이 다른 두 곳에
반영되지 않을 수 있다.

**처방.** `pie_ir`에 `Reader<'a, E: ReaderError>` 하나를 두고 세 곳이 재사용.

---

### C5. lane-table 구조체가 5곳에 정의돼 있다 (codegen)

**증상.** 디바이스가 읽는 구조체 레이아웃이 다섯 군데에 따로 적혀 있고, 이들이
일치한다는 보장은 골든 테스트뿐이다.

| 계열 | 정의 위치 |
|---|---|
| `PtirLane*` (CUDA) | `codegen/src/header.rs` (생성) · `codegen/include/ptir_abi.h` (체크인) · `codegen/runtime/cuda/fused_block0.cuh` |
| `M1*` (Metal) | `codegen/src/metal/preamble.rs:24` · `codegen/runtime/metal/ptir_m1_runtime.metal` |
| `M3Lane*` (grouped) | `codegen/src/metal/preamble.rs` |

특히 `struct M1Status`는 **3곳에 동일한 한 줄이 중복**돼 있다:

```
compiler/codegen/src/metal/preamble.rs:24        (raw MSL 리터럴)
compiler/codegen/src/metal/effects.rs:151        (push_str 인라인)
compiler/codegen/src/metal/effects.rs:214        (push_str 인라인)
```

`effects.rs`의 두 곳은 `preamble.rs`의 `common_effect_preamble()`을 참조하는 대신
같은 문자열을 다시 밀어 넣는다. 필드를 하나 추가하면 4곳을 고쳐야 한다.

또한 `M1OpParams`가 CUDA(21필드)와 Metal(16필드)에서 다르다. 의도된 차이지만
문서화도, 검사도 없다.

**영향.** 오프셋이 어긋나면 커널이 잘못된 메모리를 읽는다. 컴파일 에러가 아니라
런타임 오답이다.

**처방.** Rust 쪽에 구조체 정의를 하나 두고 CUDA·MSL 텍스트를 **양쪽 다 거기서
생성**한다. 지금은 CUDA 헤더만 생성되고 MSL은 손으로 쓴 리터럴이다.

---

### C6. release 빌드에서 조용히 깨지는 불변식

> **정정.** 아래 (a)의 `with_session`은 실제 위험이 낮다. **진짜 사례는 같은
> 파일의 `Recorder::push`**였다 — SSA id 카운터가 `next_id`와 `result_tys.len()`
> 두 벌로 존재하고 둘의 일치를 `debug_assert_eq!`로만 확인했다. release에서
> 어긋나면 그 뒤 **모든 value id가 시프트**된다. 카운터를 하나로 줄이고 진짜
> `assert_eq!`로 바꿔 수정 완료(`c6fcd1b0f`).

**증상.** 핵심 불변식이 `debug_assert!` / sentinel 값으로만 지켜진다.
release 빌드에서는 검사가 사라지고 손상된 상태가 그대로 흘러간다.

**(a) 추적 세션 중첩 — `compiler/dsl/src/context.rs:174, 192`**

```rust
pub(crate) fn with_session<R>(f: impl FnOnce() -> R) -> (...) {
    SESSION.with_borrow_mut(|s| {
        debug_assert!(s.is_none(), "nested trace session");   // release에선 사라짐
        *s = Some(Session::new());
    });
    let r = f();                                              // ← panic 시 정리 없음
    ...
}
```

두 가지 문제가 겹쳐 있다:
1. 중첩 세션이 release에서 기존 세션을 조용히 덮어쓴다 (`trace_stage`도 `:192`에서 동일).
2. `f()`가 panic하면 `s.take()`에 도달하지 못해 thread-local이 오염된 채 남는다.
   `catch_unwind` 뒤에 같은 스레드에서 다시 추적하면 debug 빌드는 assert로 죽고,
   release 빌드는 잘못된 상태를 이어받는다.

**(b) 미매핑 sentinel — `compiler/plan/src/compile.rs:454, 484`**

```rust
let mut value_map = vec![u32::MAX; ...];   // "아직 매핑 안 됨"
...
debug_assert_ne!(mapped, u32::MAX);        // release에선 사라짐
```

release에서는 인덱스 `4294967295`가 그대로 흘러 인덱싱 panic 또는 오동작이 된다.
`Option<ValueId>`가 정답이다.

**처방.** (a) `with_session`을 RAII 가드로 바꾸고 중첩을 `Result`로 거절.
(b) sentinel을 `Option`으로 교체 — 컴파일러가 미처리 지점을 전부 찾아준다.

---

## 2. MAJOR

### M1. God functions

clippy `-W clippy::too_many_lines` 측정치 (주석·공백 제외):

| 함수 | 위치 | 줄 | 뒤엉킨 책임 |
|---|---|---:|---|
| `eval_op` | `eval/src/interp.rs:1016` | **643** | 전 op의 의미론 — elementwise, reduce, broadcast, matmul, gather/scatter, mask, rng, intrinsic, channel, kernel, sink |
| `infer` | `ir/src/infer.rs:160` | **377** | 전 op의 shape/dtype 규칙 |
| `bind` | `ir/src/validate.rs:296` | **275** | 구조 검사 + 포트 + geometry + extern + SSA + intrinsic 스코프/게이팅 + T10/T11 + SPSC + readiness 분류 |
| `match_nucleus_add_order` | `plan/src/compile.rs:1384` | **256** | 15개 op 해체 + shape 검증 + 소비자 연결성 + scaled-input 변종 + `LibraryMatch` 생성 |
| `generate_c_header` | `codegen/src/header.rs:15` | **192** | enum·struct·op/dtype/stage/port/intrinsic/sink 테이블 전부 |
| `decode_op` | `ir/src/container.rs:695` | **177** | 전 op 디코딩 |
| `emit_fused_region` (CUDA) | `codegen/src/cuda/fused.rs:240` | **144** | reshape aliasing + nucleus-skip 분석 + 런타임 조립 + 노드 순회 + 에필로그 |
| `encode_op` | `ir/src/container.rs:260` | **142** | 전 op 인코딩 |
| `symbolic_result_type` | `plan/src/compile.rs:658` | **136** | ~20 op의 심볼릭 타입 전파 |
| `Builder::build` | `dsl/src/builder.rs:125` | **123** | 채널 gid 재키 + 이름표 정렬/재매핑 + host-role 도출 + lint + `Traced` 생성 |
| `emit_body` (CUDA) | `codegen/src/cuda/fused.rs:405` | **121** | 15개 이상 if-else 분기 |
| `emit_grouped_fused_region` (Metal) | `codegen/src/metal/fused.rs:153` | **120** | preamble + grouped 채널 셋업 + 노드 순회 + MTP/logits 인라인 확장 |
| `decode` | `ir/src/container.rs:569` | **119** | 컨테이너 전체 디코딩 |
| `decode_bound` | `plan/src/sidecar.rs:324` | **115** | 사이드카 디코딩 |
| `OpView::of` | `codegen/src/op_view.rs:67` | **113** | ~30 op 투영 |
| `ValidateError::fmt` | `ir/src/validate.rs:177` | **106** | 전 에러 변종 표시 |
| `Instance::step` | `eval/src/interp.rs:476` | **106** | readiness + overlay + stage dispatch + 포트 해석 + 레이어 루프 + commit |

인지 복잡도 25 초과: `eval_op`(39), `match_nucleus_add_order`(32), `fold_scalar`(31,
`plan/src/compile.rs:931`), `bind`(28).

**처방.** op별 거대 match는 C1의 테이블 파생으로 상당 부분 해소된다. `bind`,
`Builder::build`, `generate_c_header`, `emit_fused_region`은 책임별 함수 분해.

---

### M2. `plan/src/compile.rs`는 3,598줄 단일 모듈이고 API가 무차별 공개다

- 라이브러리 코드 2,399줄 + 테스트 1,199줄이 한 파일에 있다.
- `plan/src/lib.rs:29`가 `pub use compile::*;` — **35개 심볼이 큐레이션 없이 공개**된다.
  공개 API 표면이 파일 내용과 자동 동기화되므로, 내부 헬퍼를 `pub`로 바꾸는 순간
  의도치 않게 공개 API가 된다.
- 테스트 헬퍼(`program()`, `nucleus_program()`, `softmax_program()` 등,
  `compile.rs:2591+`)가 `#[cfg(test)]` 안에 갇혀 통합 테스트·벤치에서 못 쓴다.

**제안 분할** (호출 관계 기준):

| 모듈 | 내용 |
|---|---|
| `types.rs` | `SymbolicExtent`, `Dimension`, `SymbolicType`, `ValueDomain`, `NormalizedStage`, `RuntimeExtents`, `encode_symbolic_*` |
| `normalize.rs` | `normalize_stage`, `result_layout`, `live_ops`, `simplify_alias`, `fold_*`, `cse_*`, `canonicalize_commutative`, `localize_stage`, `symbolic_*_type`, `value_domain` |
| `signature.rs` | `StageSignature`, `stage_signature`, `encode_planned_op`, `signature_ports` |
| `partition.rs` | `Region`, `RegionKind`, `RegionPartition`, `PartitionKind`, `ScheduleTemplate`, `LibraryOp`, `singleton_partition`, `fused_partition`, `build_region`, `region_kind_for_node`, `compatible_schedule` |
| `nucleus.rs` | `match_nucleus_dataflow`, `match_nucleus_add_order`, `symbolic_*_match*` |
| `wire.rs` | `PlanReader`, `PlanDecodeError`, `EncodedPlanHeader`, `encode_stage_plan`, `decode_plan_header`, `scan_*` |
| `identity.rs` | `stage_identity`, `ExecutableCacheKey`, `BackendKind`, `SemanticMode`, `LaneTableHeader`, `LaneRecord`, `LaneChannelSlot`, `PlanMetrics` |

동시에 `pub use compile::*`를 명시적 재수출 목록으로 교체.

---

### M3. plan 파이프라인이 `Result`를 쓰지 않는다 — 잘못된 IR은 panic

`compile_bound` / `compile_stage` / `normalize_stage` / `stage_signature` /
`*_partition`은 전부 오류 채널이 없다. 계약 위반은 그 자리에서 죽는다:

| 위치 | 호출 |
|---|---|
| `plan/src/compile.rs:739` | `.expect("matmul right rank")` |
| `plan/src/compile.rs:1200` | `.expect("port channel localized")` |
| `plan/src/compile.rs:1280` | `.expect("shape-bearing op defines a value")` |
| `plan/src/compile.rs:1708` | `.expect("library match has nodes")` |

`ir`의 wire 디코더와 `plan`의 사이드카 디코더는 제대로 `Result`를 쓰는데,
그 사이의 컴파일 단계만 panic 규약이다. 일관성이 없고, 호스트가 신뢰할 수 없는
컨테이너를 처리할 때 프로세스를 죽인다.

---

### M4. 패스 순서가 타입으로 강제되지 않는다

> **정정 (REFUTED).** `stage_signature`는 private이고 `compile_stage_at`이
> 유일한 호출자이며 그 안에서 localize가 선행한다 — 미localize로 부를 경로가
> **없다**. capacity만 다른 두 트레이스의 서명은 `36744c33` vs `bd38c301`로
> 실제로 갈린다. 다른 스테이지의 채널 변화에 무반응인 것은 캐시 키가 스테이지
> 단위이므로 정상이다. `NormalizedStage`/`LocalizedStage` 타입 분리는 근거가
> 이 항목이었으므로 함께 소멸.

`compile_stage_at`의 호출 순서 `normalize_stage → localize_stage → stage_signature
→ *_partition`은 **관례로만** 지켜진다. `localize_stage`는 `NormalizedStage`를
제자리 변경하고, `stage_signature`는 그 결과(`channel_bindings`, `names`)를 읽는다.
localize 안 된 stage로 `stage_signature`를 불러도 컴파일되고, 조용히 틀린 시그니처가
나온다 — 그리고 시그니처는 실행 캐시 키다.

**처방.** `NormalizedStage` / `LocalizedStage`를 별도 타입으로 분리해 순서를
타입으로 강제.

---

### M5. `build_region`이 O(N²)

`plan/src/compile.rs:1795`

```rust
fn build_region(stage: &NormalizedStage, nodes: Vec<u32>, kind: RegionKind) -> Region {
    let (bases, producer) = result_layout(&stage.ops);           // stage 전체 재계산
    let mut consumers: Vec<Vec<u32>> = vec![Vec::new(); stage.value_types.len()];
    for (node, op) in stage.ops.iter().enumerate() { ... }        // stage 전체 재구축
```

`singleton_partition`(`:1321`)은 이것을 **op마다** 호출한다. N개 op면 N번의 O(N)
작업 = O(N²). `result_layout`은 별도로 5곳(`:411, :450, :1333, :1797` + `:1903`의
인라인 합산)에서 반복 계산되고, 소비자 맵 구축 루프는 3곳(`:585, :1334, :1798`)에
복사돼 있다.

**처방.** stage 진입 시 `StageIndex { bases, producer, consumers }`를 한 번 만들어
전달.

---

### M6. 침묵하는 catch-all — 새 op이 조용히 잘못 처리된다

| 위치 | 코드 | 결과 |
|---|---|---|
| `codegen/src/op_view.rs:180` | `_ => view.args = op.operands()` | 특수 인코딩이 필요한 새 op(예: `PivotThreshold`의 predicate 필드)이 조용히 **오투영** → 잘못된 커널 |
| `codegen/src/cuda/fused.rs:405` `emit_body` 말미 | `else { fallback(source) }` | 블록 병렬 형태를 가질 수 있는 새 op이 조용히 **단일 스레드 경로**로 → 성능 버그, 컴파일 에러 없음 |
| `codegen/src/launch.rs:567` `grouped_supported_tag` | 원시 16진 범위 | 범위 틈에 떨어진 새 tag가 조용히 **미지원 처리** |

반면 `encode_op`/`decode_op`/`infer`/`eval_op`의 최상위 match는 exhaustive다 (✅).
문제는 codegen 쪽에만 있다.

**처방.** catch-all을 제거하고 명시적 arm으로 전환하거나, 최소한 `OpView::of`는
`OP_TABLE`에서 파생.

---

### M7. 사이드카가 모든 바이트를 두 번 읽는다

> **정정 (실체 확인).** 두 번 읽는 것 자체보다, 두 파서의 **유일한 비대칭**이
> 문제였다: readiness phase 태그를 `preflight_bound`만 검증하고 `decode_bound`는
> 검증 없이 push했다. `Shape::new`가 이미 zero-dim과 overflow를 거부하므로
> preflight의 나머지 검사는 전부 중복이었다. 검사를 decode로 옮기고 preflight를
> 삭제 + 회귀 테스트 추가로 수정 완료(`c6fcd1b0f`).

`plan/src/sidecar.rs`의 `preflight_bound`(`:216`, 106줄)와 `decode_bound`(`:324`,
115줄)는 **같은 검증을 두 번** 수행한다. `decode_bound`가 `preflight_bound`를 부른
뒤 처음부터 다시 읽으며 태그·rank·개수 검사를 반복한다.

레코드 레이아웃의 정의는 어디에도 없고 **모듈 상단 주석(`:17-33`)에만** 존재한다.
인코더(`encode_bound_with_plans:54`)·프리플라이트·디코더 셋이 그 주석을 각자
해석한다. 한 곳의 검증 수정이 다른 곳에 반영되지 않으면 조용히 갈라진다.

---

### M8. 인터프리터가 모든 연산에서 전량 복사한다

`eval/src/interp.rs:760`

```rust
fn lanes_f32(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(x) => x.clone(),   // 이미 f32인데도 full clone
        ...
    }
}
```

모든 binary op(`bin_arith:792`, `cmp_op:819`)가 피연산자마다 이것을 호출하므로
연산 하나에 텐서 2벌이 복제된다. `[batch, vocab]` 규모에서는 수백만 원소가 무의미하게
복사된다. `broadcast_value:1720`은 `O(numel)` 인덱스 벡터를 별도 할당하고,
`canonical_reduce:855`는 입력 행을 `to_vec()`으로 복사한다.

성능 자체보다 **패턴이 굳어 있다는 점**이 문제다 — 새 op을 추가하는 사람은 이
관례를 그대로 따른다.

---

### M9. 인터프리터의 원시 인덱싱 panic — 오라클이 죽는다

> **정정.** 말미의 "`interp.rs` 전체에 44건"은 `#[cfg(test)]`를 포함한 수치다.
> 프로덕션 코드에는 **2건**. 첨자 지적 자체는 유효하나 심각도는 MINOR에 가깝다.

`eval/src/interp.rs`의 비테스트 코드에 경계 검사 없는 첨자가 최소 8곳:
`:688`(`types[id as usize]`), `:698`, `:701`, `:703`, `:710`, `:711`, `:1001`,
`:1024`(`vals[id as usize]`, `eval_op` 진입부).

`bind`를 통과한 트레이스라면 안전하다는 전제인데, 인터프리터는 bind를 거치지 않는
테스트 하네스에서도 쓰인다. 골든 모델이 `StepError::Fault` 대신 panic하면 진단이
어려워진다.

`eval/src/interp.rs` 전체에 `unwrap`/`expect`/`panic!` 44건.

---

### M10. dsl의 암묵적 전역 상태와 28개의 panic 지점

**설계.** 추적 컨텍스트가 thread-local 3개로 구현돼 있다:

```
compiler/dsl/src/context.rs:142   SESSION: RefCell<Option<Session>>
compiler/dsl/src/context.rs:148   CHANNELS_BY_GID: RefCell<BTreeMap<u64, ChannelRef>>
compiler/dsl/src/model.rs:24      MODEL: Cell<TraceConstants>
```

`value.rs`의 모든 op 함수가 `emit()` → `SESSION.with_borrow_mut(...)`을 통과한다.
시그니처에 드러나지 않는 의존이다. 결과:

- **컨텍스트 밖 호출 = panic.** `context.rs:234` `.expect("emit outside a traced stage")`
  외 세션 상태 단언 16곳.
- **`CHANNELS_BY_GID` 정리 없음.** `channel.rs:97`의 `register_channel_state`가
  `Channel::build()`마다 삽입하지만 제거하는 코드가 없다. 장수 호스트 프로세스에서
  누적된다.
- **비전역 대안 없음.** `Context`를 인자로 넘기는 경로가 아예 존재하지 않는다.

dsl 비테스트 코드의 panic 지점 총 **28곳**: `context.rs` 17, `value.rs` 8
(`:154, :219-224, :333, :564, :826, :876, :879, :888/:891/:895, :1000`),
`channel.rs` 2 (`:243, :252` — host 채널 take를 Tensor로 쓰면 panic).
저작용 라이브러리로서는 과하다.

---

### M11. `value.rs`는 한 파일 안의 세 파일

`compiler/dsl/src/value.rs` 1,087줄에 세 관심사가 섞여 있다.

| 영역 | 대략 범위 | 내용 |
|---|---|---|
| 타입 계층 | ~1-330 | `Tensor`, `TensorInner`, `ConstData`, `Arg`, `AsTensor`, `IntoConst`, `IntoShape`, `materialize_const`, `reconcile` |
| 코어 op | ~330-640 | `emit_unary`, `emit_binary` + 30여 개 자유 함수 |
| 합성 op | ~640-1087 | `softmax`, `log_softmax`, `l2norm`, `top_k`, `pivot_threshold`, `mask_apply`, `causal_mask`, `row_membership`, `masked_argmax`, `gumbel_max`, `entropy`, `nucleus_sample` |

`tensor.rs` / `ops.rs` / `composed.rs`로 분할 권장. 합성 op 영역은 C3 해결 시
상당 부분 `ir/expand.rs` 호출로 축소된다.

---

### M12. CUDA/Metal fused emitter의 슬롯 구성 + 노드 순회가 3벌

동일한 골격 — *검증 → `OpView`/`result_bases` 구성 → `region.nodes` 순회 →
슬롯 셋업 → op 호출 → status 검사 → chan-put 보정* — 이 세 번 작성돼 있다.

```rust
// compiler/codegen/src/metal/fused.rs:47  slots_for()
if !op.args.is_empty()  { slots.a0 = value_ptr(op.args[0]); }
if op.args.len() > 1    { slots.a1 = value_ptr(op.args[1]); }
if op.tag == OP_PIVOT_THRESHOLD { slots.a1 = value_ptr(op.pred_payload); }
if op.results > 0       { slots.o0 = value_ptr(base); }

// compiler/codegen/src/cuda/fused.rs:272  (인라인, 동일 로직)
if !op.args.is_empty()  { a0 = pointer(op.args[0], &aliases); }
if op.args.len() > 1    { a1 = pointer(op.args[1], &aliases); }
if op.tag == OP_PIVOT_THRESHOLD { a1 = pointer(op.pred_payload, &aliases); }
if op.results > 0       { o0 = pointer(base, &aliases); }
```

세 번째 사본은 `metal/fused.rs:237-278`(grouped 경로)이다. 이미 비대칭이 생겼다 —
CUDA에는 alias 해석이 있고 Metal에는 없다.

백엔드별로 **진짜 다른 것**은 시그니처 문법(`extern "C" __global__` vs
`kernel` + `[[buffer(N)]]`), CUDA의 블록 병렬 디스패치(`emit_body`, 121줄),
Metal 전용 grouped intrinsic 확장(`emit_mtp_drafts:282`, `emit_logits_gather:322`)
정도다. 순회·슬롯 로직 ~65줄은 공유 가능하다.

**처방.** `Slots` + `walk_region` 헬퍼를 공유 모듈로 올리고,
`trait BackendDialect { fn function_signature(..); fn emit_op_call(..);
fn emit_status_check(..); fn runtime_preamble(..); }`로 매개화.
`cuda/singleton.rs:35`와 `metal/singleton.rs:14`도 같은 방식으로 합쳐진다.

---

### M13. codegen의 문자열 조립 soup + 타입 없는 에러

- `push_str` / `write!` / `writeln!` / `format!` 호출 약 **380회**
  (`metal/fused.rs` 89, `cuda/fused.rs` 87, `metal/effects.rs` 79, `header.rs` 44 …).
- **들여쓰기 수동 관리.** 예: `cuda/fused.rs:440`의
  `"    for (m1_u32 i = threadIdx.x; ...) {\n"` — 앞 공백 개수를 잘못 세면 조용히
  들여쓰기가 틀어진다. `Indenter`나 스코프 가드가 없다.
- **출력 문법 검증 없음.** 세미콜론 누락·중괄호 불균형은 실제 디바이스 컴파일
  시점에야 드러난다. 골든은 회귀만 잡고 신규 경로의 오류는 못 잡는다.
- **에러가 `Result<String, String>`.** 어느 stage / region / node / op에서 났는지
  문맥이 없다. `program.rs:137` `emit_metal_stage`의 거부 사유를 추적하려면 계층을
  넘나들며 문자열 매칭을 해야 한다.

**처방.** `CodeWriter { fn line(); fn block(); fn indent(); }` 도입 +
`enum EmitError { .. }`에 stage/region/node 문맥 부착.

---

### M14. 생성된 헤더를 문자열 검색으로 되파싱

`compiler/codegen/src/cuda/runtime.rs:15-30`

```rust
let header = crate::rng::generate_cuda_header();
let open  = "inline constexpr char PTIR_RNG_CUDA_PREAMBLE[] = R\"PTIR_RNG_CUDA(";
let close = ")PTIR_RNG_CUDA\";";
let start = header.find(open).expect("the generated RNG header defines PTIR_RNG_CUDA_PREAMBLE") + open.len();
let end   = header[start..].find(close).expect("the RNG preamble literal is terminated") + start;
header[start..end].into()
```

자기가 방금 생성한 텍스트를 다시 파싱한다. 헤더 생성기의 공백·포맷이 조금만 바뀌어도
런타임에 `expect`로 죽는다.

**처방.** preamble 본문을 별도 함수로 분리하고, 헤더 생성기와 런타임 조립기가
**둘 다 그 함수를 호출**한다. 문자열 검색 제거.

---

### M15. 약한 타입 — id가 전부 type alias

```rust
compiler/ir/src/types.rs:11   pub type ValueId = u32;
compiler/ir/src/op.rs:40      pub type ChannelIndex = u32;
compiler/ir/src/op.rs:42      pub type NameIndex = u16;
```

newtype이 아니라 alias이므로 value id · node index · lane slot · channel slot ·
region id를 서로 바꿔 넣어도 컴파일된다. `plan` 쪽은 아예 원시 타입이다:
`Region.nodes: Vec<u32>`, `inputs: Vec<u32>`, `outputs: Vec<u32>`
(`compile.rs:152-154`), `ChannelSink { channel_slot: u32, value: u32 }`(`:147-148`),
`NormalizedStage.channel_bindings: Vec<u32>`(`:105`).

C6(b)의 `u32::MAX` sentinel도 같은 뿌리다.

**처방.** C1·M2 이후에 newtype을 도입하면 컴파일러가 나머지 혼용 지점을 전부
찾아준다. **순서가 중요하다** — 먼저 하면 대량의 기계적 수정이 리팩토링과 뒤엉킨다.

---

### M16. 컨테이너가 불변식을 구조적으로 보호하지 않는다

`ir/src/container.rs:168`

```rust
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TraceContainer {
    pub names: Vec<String>,      // "Sorted + deduped for canonicality"
    pub channels: Vec<ChannelDecl>,
    pub ports: Vec<PortBinding>, // "Sorted by port tag, unique."
    pub stages: Vec<StageProgram>,
    pub externs: Vec<ExternDecl>,
}
```

주석이 요구하는 정렬·유일성 불변식을 강제하는 것이 없다. 전 필드가 `pub`이고
`Default`까지 파생되어 있어 누구나 불변식을 깨는 컨테이너를 만들 수 있다.
유일한 게이트는 `bind()`이고, 손으로 만든 컨테이너는 불평 없이 `encode()`된다.
`BoundTrace`, `StageProgram`, `PortBinding`, `ChannelDecl`도 동일하다.

또한 wire 내부 헬퍼가 불필요하게 공개돼 있다: `put_u16`(`:461`), `put_u32`(`:464`),
`encode_shape`(`:433`).

---

### M17. 라이브러리 op 집합이 4곳에 열거돼 있다

> **정정 (처방 REFUTED).** 네 곳은 같은 집합을 우연히 공유할 뿐 **서로 다른
> 질문**에 답한다(값 도메인 / 리전 종류 / 융합 가능성 / 방출 형태).
> `Family::Library` 하나로 치환하면 nucleus 융합이 깨진다. 중복 열거를 없애려면
> 네 개의 독립적인 술어를 `OpSpec`에 각각 두어야 하고, 그건 이 항목의 처방이
> 아니다.

`{TopK, SortDesc, CumSum, CumProd, MatMul}`이 다음 4곳에 각각 나열된다:

```
compiler/plan/src/compile.rs:1095       value_domain
compiler/plan/src/compile.rs:1771-1777  region_kind_for_node
compiler/plan/src/compile.rs:1786       compatible_schedule (첫 번째)
compiler/plan/src/compile.rs:1790       compatible_schedule (두 번째)
```

라이브러리 op을 하나 추가하면 넷 다 고쳐야 하고, 하나를 빠뜨리면 그 op이 생성
리전에 잘못 융합된다. `OP_TABLE`의 `family` 필드가 이미 존재하므로 그것으로
질의해야 한다.

---

## 3. MINOR

### m1. 공개 rustdoc에 남은 내부 코드네임 28건

`dsl/**`에 `"echo's canonical TraceContainer"` 류가 **26회**,
`"charlie's C++ tables"`가 2회. 지금은 존재하지 않는 워크스트림 이름이 공개 문서에
그대로 나간다.

```
compiler/dsl/src/lib.rs:11        "bindings into echo's canonical ..."
compiler/dsl/src/builder.rs:6     "echo's canonical [`TraceContainer`]"
compiler/dsl/src/builder.rs:204   "Build echo's channel declarations ..."
compiler/dsl/src/context.rs:21    "re-export of echo's canonical [`Stage`]"
compiler/dsl/src/error.rs:93      "echo's validator on the canonical bytes"
compiler/dsl/src/lint.rs:3,7      "Echo's [`bind`] is the ..."
compiler/dsl/src/value.rs:2       "Ops emit echo's canonical ..."
compiler/ir/src/op.rs:554         "the declarative identity charlie's C++ tables are generated from"
compiler/dsl/src/intrinsics.rs:42 "... DISTINCT shapes — charlie's ..."
```

전부 `pie-ir` 또는 "the IR"로 치환.

### m2. `codegen/src/lib.rs`의 크레이트 문서가 낡았다

`compiler/codegen/src/lib.rs:16-19`:

> The CUDA and Metal *region* emitters — today's `fused_codegen.hpp`,
> `singleton_codegen.hpp`, and `m1_codegen.cpp` in the drivers — **land here next**

이미 `codegen/src/cuda/`(5개 파일)와 `codegen/src/metal/`(8개 파일)로 들어와 있다.
또한 문서가 `header`와 `rng`만 나열하고, 실제 존재하는 `launch`, `program`,
`op_view`, `region_analysis`, `cuda`, `metal` 모듈은 언급조차 없다.

### m3. `program_hash`와 `container_hash`가 완전히 동일한 함수

`compiler/ir/src/lib.rs:74`와 `:89`. 후자는 전자에 그대로 위임한다. 이름이 서로 다른
의미론을 암시하지만 차이가 없다. 게다가 FNV 상수가 `plan`에도 세 번째로 복사돼 있다:
`plan/src/compile.rs:3552, 3555` (`0xcbf2_9ce4_8422_2325`, `0x0000_0100_0000_01b3`,
알고리즘 이름을 밝히는 주석 없음).

### m4. `elem_size`가 중복이고 dtype 추가 시 조용히 틀린다

> **정정 (REFUTED).** 도달 불가한 경로다. 중복 자체는 사실이므로 정리 대상으로는
> 유효하지만 "조용히 틀린다"는 오늘 성립하지 않는다.

```rust
// compiler/ir/src/container.rs:469
pub fn const_elem_size(dtype: DType) -> usize { match dtype { DType::Bool => 1, _ => 4 } }

// compiler/dsl/src/value.rs:160
fn elem_size(d: DType) -> usize { match d { DType::Bool => 1, _ => 4 } }
```

바이트 단위로 동일한 중복. 현재 `DType`이 `{F32, I32, U32, Bool}` 넷뿐이라 맞지만,
F16/BF16/E8M0 같은 dtype이 추가되면 **컴파일 에러 없이 잘못된 크기**를 낸다
(loader는 이미 E8M0 dtype을 도입했다). `_ =>`를 exhaustive match로 바꿔야 한다.
`DType::is_float`(`types.rs:29`)이 `matches!(self, DType::F32)`인 것도 같은 취약점.

### m5. 깨진 intra-doc 링크

`compiler/ir/src/op.rs:7`이 `[`crate::types::Op`]`를 참조하지만 그런 경로는 없다
(PSIR v4의 `Op`를 뜻하는 듯). rustdoc 링크 해석 실패.

### m6. 테스트 헬퍼가 공개 API에 유출

`compiler/ir/src/registry.rs:255` `ModelProfile::dummy()`가 `pub`.
`#[cfg(test)]`나 `test-support` feature 뒤로 옮겨야 한다.

### m7. 이름 없는 매직 상수

| 값 | 위치 | 의미 |
|---|---|---|
| `32_768` | `plan/src/compile.rs:1855` | 계층적 스케줄 전환 임계값. 왜 32K인지 설명 없음 |
| `MAX_SIDECAR_STAGES = 4` | `plan/src/sidecar.rs:214` | `Stage` enum의 variant 수에서 파생되지 않은 하드코딩 |
| `0x200`/`0x300`/`0x400`/`0x480`/`0x500`/`0x700`/`0x780` | `codegen/src/metal/effects.rs:61,75,81,89,100,186,202` | 디바이스 fault 코드 공간. 이름도 문서도 없음 |

### m8. 중복 헬퍼 (eval)

| 쌍 | 위치 | 차이 |
|---|---|---|
| `zeros` / `placeholder` | `interp.rs:269` / `pareval.rs:188` | import 경로만 다름 |
| `gather_flat` / `gather_flat_fill0` | `interp.rs:1688` / `:1698` | fallback 값만 다름 → `fill: Option<T>` 하나로 |
| `canonical_max`/`canonical_min`/`element_max`/`element_min` | `interp.rs:932/948/964/980` | NaN 정책만 다름. 4×16줄 → ~15줄 |
| `bin_arith` / `cmp_op` | `interp.rs:792` / `:819` | 반환 dtype만 다름 |
| `const_value` / `Value::from_le_bytes` | `interp.rs:643` / `:77` | LE 디코드 로직 복붙 |

### m9. 사소한 API 잡음

- `dsl/src/channel.rs:292` `pub struct Put(())` — 항상 버려지고 await되지 않는 무의미한 래퍼.
- `dsl/src/intrinsics.rs:25` `pub const activation_type: DType = DType::F32` — 바로 위
  `:23`의 문서는 "late-bound backend activation dtype"이라고 하지만 실제로는 F32 하드코딩.
- `dsl/src/intrinsics.rs:54` `hidden()`이 `Shape::matrix(rows, vocab())`를,
  `:64` `query()`가 `Shape::vector(vocab())`를 쓴다. hidden/query 폭은 vocab이 아니다.
  `:56`의 주석이 근사임을 자인하지만(`"modeled here as the activation rows"`),
  shape는 `infer`/`plan`이 실제로 소비하는 값이라 근사가 아래로 전파된다.
- DSL op 하나 추가에 3곳 편집 필요: `value.rs` + `lib.rs:69` 재수출 + `lib.rs:93` prelude.

### m10. `TODO`/`FIXME`가 0개

부채가 전부 코드 밖(`ptir-refactor.md`, 이 문서)에서만 추적된다. 파일을 여는 사람에게는
지뢰가 보이지 않는다. 위 항목 중 즉시 고치지 않을 것들은 해당 지점에 마커를 남기고
이 문서의 항목 번호를 참조하도록 권장.

---

## 4. 유지해야 할 것 (리팩토링 중 깨뜨리지 말 것)

리팩토링이 이것들을 후퇴시키면 순손실이다.

1. ~~**`no_std` 분리가 깨끗하다.**~~ — **정정 (거짓).** cfg soup이 없다는 것과
   `pie-ir`의 의존성이 0개라는 것은 맞지만, cfg 지점은 3곳이 아니라 4~5곳이고
   무엇보다 **`cargo check -p pie-ir --no-default-features`가 컴파일되지 않았다**:
   `validate.rs`에 `alloc::string::String` import가 없었고 `std` 프렐류드가 그것을
   가려주고 있었다. 한 번도 빌드된 적 없는 `no_std`는 `no_std`가 아니다.
   import 추가 + CI 가드로 수정 완료(`5e24b16d7`). 이제부터는 진짜로 지킬 것.
2. **codegen의 순수성이 실제로 성립한다.** `HashMap`/`HashSet` 0개, `std::{env,fs,io,time}`
   0개, 런타임 템플릿은 전부 `include_str!`. 반복은 전부 `Vec` 순서. `header.rs`와
   `rng.rs`에 `f() == f()` 결정성 테스트가 있다. → GPU 없이 골든 테스트가 되는 근거.
3. **`ValidateError`가 모범적이다** (`ir/src/validate.rs:76-175`). 구조화돼 있고 문제의
   op/intrinsic/channel/stage를 실어 나르며 `Display`가 상세하다. 다른 에러 타입의
   목표 수준.
4. **컨테이너 디코더의 방어가 견고하다** — 단, **부분적으로 거짓이었다.**
   `take`의 `checked_add`와 재인코딩 canonical 강제는 사실이다. 그러나
   `bounded_count`는 "입력에 남은 바이트 수"만을 상한으로 삼아 `n_ops`에
   `bounded_count(n, 1, ...)`을 쓰고 있었다 — 1바이트당 op 1개, 즉 상한이 사실상
   없었다. 게스트가 제출하는 입력이고 디코드가 전역 락 안에서 일어나므로 DoS다.
   구조적 상한(`MAX_OPS = 1<<16` 등)을 추가해 수정 완료(`c6fcd1b0f`).
5. **`pareval`가 `eval_op`를 실제로 재사용한다.** README 주장 검증됨 —
   `pareval.rs:25`가 `eval_op`를 import하고 `:168`에서 호출한다. 채널/커널/intrinsic
   단락 처리는 올바른 특수화이지 중복이 아니다. 두 번째 평가기는 없다.
6. **생성물 3종 모두 drift 테스트가 걸려 있다.** `ptir_abi.h`, `rng_contract.generated.h`,
   `ptir_rng.generated.metal` — 생성기 1개 : 테스트 1개, 누락 없음.
7. **핵심 op 디스패치는 exhaustive하다.** `encode_op`/`decode_op`/`infer`/`eval_op`의
   최상위 match에 catch-all이 없어 컴파일러가 누락을 잡는다. (문제는 codegen 쪽 — M6.)
8. **`table_matches_op_metadata`**(`op.rs:971`)가 `tag`/`results`/`arity`/`map_operands`
   정합을 실제로 검증한다. C1을 해결해도 이 테스트의 의도는 남겨야 한다.

---

## 5. 권장 착수 순서

| 단계 | 작업 | 해소 항목 | 근거 |
|---|---|---|---|
| **1** | `OP_TABLE`을 선언적 입력으로 승격, 매크로/build script로 enum·`tag()`·arity·encode/decode 스켈레톤 생성. 다운스트림은 `pie_ir::op::tags::*` import | **C1, C2**, M6 일부, M17 | 편집 지점 11→2. 이후 모든 작업의 기반 |
| **2** | `ir/expand.rs`를 유일 정의로 만들고 dsl이 호출. 타입 부착은 `infer`에 위임 | **C3**, M11 일부 | 프로덕션 의미론 드리프트 제거 |
| **3** | `Reader<'a, E>`를 `ir`로 추출, sidecar/plan 재사용. sidecar preflight/decode를 단일 패스로 | **C4**, M7 | wire 처리 로직 3벌 → 1벌 |
| **4** | `compile.rs` 7개 모듈 분할, 테스트 1,199줄을 `tests/`로, `pub use compile::*` → 명시적 재수출 | **M2**, M1 일부 | 이후 작업의 리뷰 가능성 확보 |
| **5** | `debug_assert` 불변식을 타입/`Result`로 승격 (`u32::MAX`→`Option`, 세션 가드→RAII) | **C6**, M3 일부 | release 무결성 |
| **6** | newtype 도입 (`ValueId`, `NodeIndex`, `ChannelSlot`, `LaneId`) | **M15**, M16 | 4·5 이후에 해야 기계적 수정이 뒤엉키지 않음 |
| **7** | codegen: `CodeWriter` + `BackendDialect` trait, catch-all 제거, `EmitError` 도입, RNG 문자열 수술 제거 | **M6, M12, M13, M14**, C5 일부 | 새 백엔드 추가 비용의 실체 |
| **8** | C5 잔여: MSL 구조체를 Rust에서 생성 (CUDA 헤더처럼) | **C5** | 레이아웃 불일치의 마지막 구멍 |
| **9** | 문서/이름 정리 (m1~m10) | MINOR 전부 | 저비용, 별도 커밋 |

> **진행 상황 (`c6fcd1b0f` 기준).**
> - **1단계 완료.** `declare_ops!`/`declare_intrinsics!` 매크로로 승격, `tags`/
>   `spec()`/`family_of()`/`*::ALL` 제공. 다운스트림 중복 상수 64개 제거,
>   `launch.rs`의 원시 hex 범위도 `spec()` 조회로 교체. `op.rs` 1136→864줄.
>   **M17은 여기서 함께 처리되지 않는다** — 위 정정 참조.
> - **3단계 절반 완료.** sidecar preflight는 삭제됐다(단일 패스). `Reader` 3벌
>   추출(C4)은 미착수.
> - **5단계 일부 완료.** C6의 실제 사례(`Recorder::push`)는 수정됐다. 세션 가드
>   RAII화는 미착수.
> - **문서에 없던 CRITICAL 7건**(§0-A)이 모두 이 순서 바깥에 있었다. 착수 순서를
>   따르되 그것이 전부라고 가정하지 말 것.
> - **방어망 신설**: `op_table_drift.rs`(Rust `OP_TABLE` ↔ C++ `op_info` 5테스트),
>   `corpus_coverage.rs`(op/intrinsic/schedule/stage 커버리지 tripwire),
>   `provenance.rs`(오라클 출처 골든의 무단 재생성 차단).
>   6단계·7단계의 대규모 이동은 이 그물이 있어야 안전하다.

**가장 위험한 순서는 1 → 2 → 3이다.** 이 셋은 "손으로 유지하는 사본이 어긋나면
조용히 잘못된 커널/의미론이 나간다"는 동일한 병증이고, 이를 먼저 없애야 4번 이후의
대규모 이동이 안전해진다.

`ptir-refactor.md` §2의 수용 테스트 — *"`compiler/ir`, `compiler/plan`, 또는 드라이버의
IR 지식을 건드리지 않고 세 번째 백엔드를 추가할 수 있는가"* — 는 **현재 아니오**다.
C2(tag 사본 6벌)와 M12(emitter 로직 3벌) 때문이다. 7단계가 그 답을 예로 바꾸는 작업이다.

---

## 부록 A. 측정 방법

```bash
# 길이/복잡도
cargo clippy -p pie-ir -p pie-plan -p pie-eval -p pie-codegen -p pie-dsl --lib -- \
  -W clippy::cognitive_complexity -W clippy::too_many_lines -W clippy::too_many_arguments

# 기본 lint. `--no-deps`가 없으면 cargo가 RUSTC_WORKSPACE_WRAPPER로 경로 의존
# (pie-driver-abi 등)까지 린트해서 compiler 밖 경고가 섞여 든다 — 이 문서의
# "3건"이 실제로 29건이었던 것과는 별개의 오차원이지만 둘 다 계수를 흐린다.
cargo clippy --no-deps --all-targets \
  -p pie-ir -p pie-plan -p pie-eval -p pie-codegen -p pie-dsl -p pie-compiler-tests

# 테스트
cargo test -p pie-ir -p pie-plan -p pie-eval -p pie-codegen -p pie-dsl -p pie-compiler-tests

# panic 지점 분포.
# 주의: 아래 grep은 `src/` 안의 `#[cfg(test)]` 모듈을 함께 센다. 이 문서 초판의
# "190건"이 그 결과다 (프로덕션 실측은 68건). 테스트를 빼려면 파일별 계수가
# 아니라 모듈 경계를 봐야 하므로, 계수를 인용하기 전에 반드시 눈으로 확인할 것.
grep -rn --include='*.rs' -E "\.unwrap\(\)|\.expect\(|panic!|unreachable!" compiler/*/src \
  | cut -d: -f1 | sort | uniq -c | sort -rn
```

## 부록 B. crate별 규모

`src/` 기준 (테스트 모듈 포함), `find <crate>/src -name '*.rs' | xargs wc -l`.

| Crate | src LOC | 최대 파일 | panic 지점 |
|---|---:|---|---:|
| `ir` | 5,175 | `validate.rs` 1,358 | 20 |
| `codegen` | 4,136 | `cuda/fused.rs` 604 | 17 |
| `plan` | 4,127 | `compile.rs` 3,598 | 70 |
| `eval` | 2,835 | `interp.rs` 2,199 | 54 |
| `dsl` | 2,776 | `value.rs` 1,087 | 29 |
| `tests` | 4,922 | `ptir_golden.rs` 2,025 | — |

`plan`은 크레이트 LOC의 **87%가 파일 하나**(`compile.rs`)에 있다 — M2의 근거.

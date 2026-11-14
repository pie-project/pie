# Adapter slots on CUDA (sc/np-cuda-tables-2)

How LoRA adapters live in the pool on CUDA, for Metal, Vulkan and XLA to build the same model. REDESIGN 2.6,
contract 0d2b5dfc2 plus the content key below. Paths are relative to `crates/`.

**Contract change (content-keyed adapters).** One new verb, `Engine::adapter_key(&InstanceBinding) ->
Result<Option<u128>>` (engine/src/engine.rs), default `Ok(None)`.
- It returns a digest of the planes the binding lands. The runtime asks for it before taking slots.
- Bindings with one key share one adapter item: the same `adapter_slots` refs, taken at the first holder
  from `ProgramFacts::adapter`, refcounted, and freed at the last close.
- The engine lands planes only into refs it has not landed already.
- No field of `ProgramFacts`, `InstanceBinding` or `Rebind` changed.

1. **What keys a slot: the adapter's content.** The runtime keys the adapter item by `adapter_key`
   (`Owner::Adapter(u128)`, runtime/src/store/roles.rs). The item holds one part (its slot refs) and the set
   of instances attached to it.
   - CUDA's key digests the sink's placement constant and its plane seeds, in the sink's plane order
     (`adapter_key_of`, engine-cuda/src/api.rs). The planes are a function of exactly these and the load's
     banks, so the key is rank-independent under TP.
   - The engine stores each landed adapter by its first slot ref, with its holders (`AdapterIds`,
     engine-cuda/src/store.rs).
   - A fire routes each lane to one id per adapter, not per instance (`ids_for`). Ids under `ZERO_IDS = 2`
     (engine-cuda/src/store/layout.rs:70) name a zero plane, for arming's synthetic rows.
2. **A shared adapter is one set of slots and one landing.** A second binding with the same key gets the
   first holder's refs (`take_instance`) and takes no slots.
   - The engine sees the refs already landed and joins them instead of writing (`adapter_of` →
     `Shell::join_adapter`).
   - Tests: runtime `a_shared_adapter_is_one_set_of_slots_freed_at_the_last_close`; engine-cuda
     `store::tests::a_shared_adapter_is_one_landing_and_one_id` (main's `a_shared_adapter_is_one_slot_and_one_load`
     gate, in slots); and the twin in the `golden_adapter` probe.
3. **Operator-declared adapters have no place on CUDA.** `register_adapter` refuses and names the
   replacement (engine-cuda/src/api.rs). `[model.adapters] seats` is refused at parse
   (worker/src/config/removed.rs:145).
   - The roster (`registered`) still parses for the other engines.
   - Whether CUDA should host operator adapters is open with the user.
4. **Kernels read a table of addresses in `Reserved::Tables`.**
   - Per bank: `ids` u64 entries, entry `id` the address of that adapter's plane of the bank.
   - Region layout (`AdapterShape::region_bytes`, layout.rs:161): a zero plane, then each bank's table.
   - The correction takes `AdapterTables { a, b }` (kernels-cuda/src/linear/lora.rs). The down projection
     is `select_gemv` with `ExpertTable::located(tables.a, ..)`. The up projection reads
     `expert_plane(bank_b, .., table_b, 1, 0, id)` (kernels-cuda/kernels/linear/lora.cu:34). ROCm matches.
   - There is no base + seat × slot form: a zero table is refused.
5. **Adapter slots park to host and are never dropped** (`SlotRole::Adapter` is not replayable,
   engine/src/chunk.rs:256).
   - A shared adapter parks only once every holder is that process's or parked already (`moves_with`,
     `Owner::Adapter`). It unparks with any holder's process (`unparks_with`).
   - After it moves, `stale` rebinds each holder on the device with `Rebind::Instance.adapter` naming the
     same new refs. A holder still parked is rebound when it unparks.
   - The engine retargets the adapter's id once its last holder has been told (`AdapterIds::rebind`). Guest
     slots copy device↔host in `copy_guest_slots`.
   - An evacuation restages every bank's table. Close detaches the instance; at the last close the runtime
     frees the slots and the engine forgets the landing and its id.
6. **`ProgramFacts::adapter`** (engine/src/program.rs:106) is the item's `Footprint`: `bytes` = the sum of
   the bank planes, `slots` = the bank count (one plane per slot), `contiguous: false`. It is zero when the
   program has no `lora` sink, and is taken only for the first holder of a key.
   - **`InstanceBinding::adapter_slots`** (program.rs:88) holds the shared item's refs in bank-name order,
     the same for every holder.
   - At bind the engine lands the planes there (`land_adapter`), or joins them if they are already landed.

# Adapter slots on CUDA (sc/np-cuda-tables-2)

How LoRA adapters live in the pool on CUDA, for Metal, Vulkan and XLA to build the same model. REDESIGN 2.6,
contract 0d2b5dfc2. Paths are relative to `crates/`.

1. **What keys a slot: the instance.** An adapter is the planes a bound instance's program carries in its
   `lora` sink (its seeds). The engine stores its slot refs under the `InstanceId` (`AdapterIds::refs`,
   engine-cuda/src/store.rs:4887). There is no adapter id in the contract.
   - Within a fire, a lane routes to a small per-fire **id** (`Pools::adapter_ids`, store.rs:3526; LRU in
     `ids_for`, store.rs:4925).
   - Ids under `ZERO_IDS = 2` (engine-cuda/src/store/layout.rs:70) always name a zero plane, for arming's
     synthetic rows. A fire has at most `ZERO_IDS + max_lanes` ids.
2. **Shared adapters are not one slot any more.** Each binding takes its own Adapter slots and lands its
   own copy, so two instances with the same planes hold two copies. The old engine-side LRU that shared a
   bank row by content is deleted.
   - `blob::Adapters` (engine-cuda/src/blob.rs) still reads a mounted directory once per plane file
     (`Blobs`), but nothing binds through it now.
3. **Operator-declared adapters have no place on CUDA.** `register_adapter` refuses and names the
   replacement (engine-cuda/src/api.rs:829). `[model.adapters] seats` is refused at parse
   (worker/src/config/removed.rs:145).
   - The roster (`registered`) still parses; on the other engines it fills the model-text banks.
   - No lane could route to these rows on CUDA before this change either.
4. **Kernels read a table of addresses in `Reserved::Tables`.**
   - Per bank: `ids` u64 entries, entry `id` the address of that adapter's plane of the bank.
   - Region layout (`AdapterShape::region_bytes`, layout.rs:161): a zero plane, then each bank's table,
     after the weight cache's regions.
   - The correction takes `AdapterTables { a, b }` (kernels-cuda/src/linear/lora.rs). The down projection
     is `select_gemv` with `ExpertTable::located(tables.a, ..)` (lora.rs:113). The up projection reads
     `expert_plane(bank_b, .., table_b, 1, 0, id)` (kernels-cuda/kernels/linear/lora.cu:34). ROCm matches.
   - There is no base + seat × slot form: a zero table is refused.
5. **Adapter slots park to host and are never dropped** (`SlotRole::Adapter` is not replayable,
   engine/src/chunk.rs:256).
   - The runtime parks them with the instance's other parts. Guest slots copy device↔host in
     `copy_guest_slots` (store.rs:4186).
   - After an unpark, `Rebind::Instance.adapter` (engine/src/program.rs:197) names the new refs.
     `rebind_adapter` (store.rs:3494) stores them and rewrites the instance's id entries if it holds one.
   - An evacuation restages every bank's table (store.rs:2331, `stage_adapter_tables` store.rs:3403).
   - Close forgets the refs and frees the id (store.rs:3518); the runtime takes the slots back.
6. **`ProgramFacts::adapter`** (engine/src/program.rs:106) is the item's `Footprint`, filled at
   engine-cuda/src/serve.rs:832: `bytes` = the sum of the bank planes, `slots` = the bank count (one plane
   per slot), `contiguous: false`. It is zero when the program has no `lora` sink.
   - The Adapter slot's bytes are the largest plane, rounded to 256 B (`AdapterShape`, layout.rs:86).
   - **`InstanceBinding::adapter_slots`** (program.rs:88) holds those refs in bank-name order. The runtime
     takes them in `take_instance` (runtime/src/store/roles.rs:417) as the instance item's last part.
   - Bind lands each plane in its slot on the stream (`land_adapter`, store.rs:3436; called from
     `adapter_of`, api.rs:472).

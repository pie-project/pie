//! DIAGNOSTIC (temporary): does the inferlet's sugar lowering (build_sampler /
//! lower_sampler, params baked as constant_f32_dyn immediates) produce the SAME
//! bytecode + hash as standard_program (params as HostSubmit inputs, delta's
//! recognizer reference)? If not, recognize() misses the real inferlet program.
#![allow(deprecated)] // the whole point is to exercise the deprecated sugar `lower_sampler`.
use pie_sampling_ir::program_hash;
use sampling_edsl::{SamplerSpec, StandardSampler, lower_sampler, standard_program};

#[test]
fn diag_sugar_vs_standard() {
    let v = 151936u32;

    let s_topp = lower_sampler(SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, v).unwrap();
    let (b_topp, _) = standard_program(StandardSampler::TopP, v).unwrap();
    eprintln!(
        "TopP: sugar len={} hash={:016x} | standard len={} hash={:016x} | match={}",
        s_topp.bytecode.len(),
        program_hash(&s_topp.bytecode),
        b_topp.len(),
        program_hash(&b_topp),
        program_hash(&s_topp.bytecode) == program_hash(&b_topp)
    );

    let s_minp = lower_sampler(SamplerSpec::MinP { temperature: 0.8, p: 0.05 }, v).unwrap();
    let (b_minp, _) = standard_program(StandardSampler::MinP, v).unwrap();
    eprintln!(
        "MinP: sugar len={} hash={:016x} | standard len={} hash={:016x} | match={}",
        s_minp.bytecode.len(),
        program_hash(&s_minp.bytecode),
        b_minp.len(),
        program_hash(&b_minp),
        program_hash(&s_minp.bytecode) == program_hash(&b_minp)
    );

    let s_temp = lower_sampler(SamplerSpec::Multinomial { temperature: 0.8 }, v).unwrap();
    let (b_temp, _) = standard_program(StandardSampler::Temperature, v).unwrap();
    eprintln!(
        "Temp: sugar len={} hash={:016x} | standard len={} hash={:016x} | match={}",
        s_temp.bytecode.len(),
        program_hash(&s_temp.bytecode),
        b_temp.len(),
        program_hash(&b_temp),
        program_hash(&s_temp.bytecode) == program_hash(&b_temp)
    );

    eprintln!("delta baked TopP: len=161 hash=fdebb8135fe248e7");
}

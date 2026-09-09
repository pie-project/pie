use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

use libloading::Library;

use super::color::rgb8_to_nv12_pitched;
use super::y4m::frame_rate;

const API_MAJOR: u32 = 13;
const API_MINOR_MAX: u32 = 1;

const fn api_version(major: u32, minor: u32) -> u32 {
    major | (minor << 24)
}

const fn struct_version(api: u32, ver: u32) -> u32 {
    api | (ver << 16) | (0x7 << 28)
}

const fn struct_version_ex(api: u32, ver: u32) -> u32 {
    struct_version(api, ver) | (1 << 31)
}

type Status = c_int;
const NV_ENC_SUCCESS: Status = 0;
const NV_ENC_ERR_NEED_MORE_INPUT: Status = 17;

const STATUS_NAMES: [&str; 27] = [
    "SUCCESS",
    "NO_ENCODE_DEVICE",
    "UNSUPPORTED_DEVICE",
    "INVALID_ENCODERDEVICE",
    "INVALID_DEVICE",
    "DEVICE_NOT_EXIST",
    "INVALID_PTR",
    "INVALID_EVENT",
    "INVALID_PARAM",
    "INVALID_CALL",
    "OUT_OF_MEMORY",
    "ENCODER_NOT_INITIALIZED",
    "UNSUPPORTED_PARAM",
    "LOCK_BUSY",
    "NOT_ENOUGH_BUFFER",
    "INVALID_VERSION",
    "MAP_FAILED",
    "NEED_MORE_INPUT",
    "ENCODER_BUSY",
    "EVENT_NOT_REGISTERD",
    "GENERIC",
    "INCOMPATIBLE_CLIENT_KEY",
    "UNIMPLEMENTED",
    "RESOURCE_REGISTER_FAILED",
    "RESOURCE_NOT_REGISTERED",
    "RESOURCE_NOT_MAPPED",
    "NEED_MORE_OUTPUT",
];

fn status_name(s: Status) -> &'static str {
    STATUS_NAMES.get(s as usize).copied().unwrap_or("UNKNOWN")
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

const H264_GUID: Guid = Guid {
    data1: 0x6bc8_2762,
    data2: 0x4e63,
    data3: 0x4ca4,
    data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
};
const PRESET_P4_GUID: Guid = Guid {
    data1: 0x90a7_b826,
    data2: 0xdf06,
    data3: 0x4862,
    data4: [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
};
const H264_PROFILE_HIGH_GUID: Guid = Guid {
    data1: 0xe7cb_c309,
    data2: 0x4f7a,
    data3: 0x4b89,
    data4: [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
};

const MIN_WIDTH: u32 = 145;
const MIN_HEIGHT: u32 = 49;

const NV_ENC_DEVICE_TYPE_CUDA: u32 = 1;
const NV_ENC_BUFFER_FORMAT_NV12: u32 = 1;
const NV_ENC_PIC_STRUCT_FRAME: u32 = 1;
const NV_ENC_PIC_TYPE_UNKNOWN: u32 = 0xff;
const NV_ENC_PIC_FLAG_EOS: u32 = 0x8;
const NV_ENC_TUNING_INFO_HIGH_QUALITY: u32 = 1;
const NV_ENC_PARAMS_RC_VBR: u32 = 1;

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct RcParams {
    version: u32,
    rate_control_mode: u32,
    const_qp: [u32; 3],
    average_bitrate: u32,
    max_bitrate: u32,
    vbv_buffer_size: u32,
    vbv_initial_delay: u32,
    flags: u32,
    tail: [u32; 22],
}
const _: () = assert!(size_of::<RcParams>() == 128);
const RC_FLAG_LOOKAHEAD: u32 = 1 << 5;
const RC_FLAG_ZERO_REORDER_DELAY: u32 = 1 << 9;
const RC_FLAG_EXT_LOOKAHEAD: u32 = 1 << 16;

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct Config {
    version: u32,
    profile_guid: Guid,
    gop_length: u32,
    frame_interval_p: i32,
    mono_chrome_encoding: u32,
    frame_field_mode: u32,
    mv_precision: u32,
    rc_params: RcParams,
    codec_config: [u32; 448],
    reserved: [u32; 278],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<Config>() == 3584);

impl Config {
    fn disable_reordering(&mut self) {
        self.frame_interval_p = 1;
        self.rc_params.flags &= !(RC_FLAG_LOOKAHEAD | RC_FLAG_EXT_LOOKAHEAD);
        self.rc_params.flags |= RC_FLAG_ZERO_REORDER_DELAY;
        self.rc_params.tail[12] = 0;
    }
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct PresetConfig {
    version: u32,
    reserved: u32,
    preset_cfg: Config,
    reserved1: [u32; 256],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<PresetConfig>() == 5128);

#[allow(dead_code)]
#[repr(C)]
struct InitializeParams {
    version: u32,
    encode_guid: Guid,
    preset_guid: Guid,
    encode_width: u32,
    encode_height: u32,
    dar_width: u32,
    dar_height: u32,
    frame_rate_num: u32,
    frame_rate_den: u32,
    enable_encode_async: u32,
    enable_ptd: u32,
    flags: u32,
    priv_data_size: u32,
    reserved: u32,
    priv_data: *mut c_void,
    encode_config: *mut Config,
    max_encode_width: u32,
    max_encode_height: u32,
    max_me_hint_counts: [u32; 8],
    tuning_info: u32,
    buffer_format: u32,
    num_state_buffers: u32,
    output_stats_level: u32,
    reserved1: [u32; 284],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<InitializeParams>() == 1800);

#[allow(dead_code)]
#[repr(C)]
struct OpenSessionExParams {
    version: u32,
    device_type: u32,
    device: *mut c_void,
    reserved: *mut c_void,
    api_version: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<OpenSessionExParams>() == 1552);

#[allow(dead_code)]
#[repr(C)]
struct CreateInputBuffer {
    version: u32,
    width: u32,
    height: u32,
    memory_heap: u32,
    buffer_fmt: u32,
    reserved: u32,
    input_buffer: *mut c_void,
    sys_mem_buffer: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 63],
}
const _: () = assert!(size_of::<CreateInputBuffer>() == 776);

#[allow(dead_code)]
#[repr(C)]
struct CreateBitstreamBuffer {
    version: u32,
    size: u32,
    memory_heap: u32,
    reserved: u32,
    bitstream_buffer: *mut c_void,
    bitstream_buffer_ptr: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<CreateBitstreamBuffer>() == 776);

#[allow(dead_code)]
#[repr(C)]
struct LockInputBuffer {
    version: u32,
    flags: u32,
    input_buffer: *mut c_void,
    buffer_data_ptr: *mut c_void,
    pitch: u32,
    reserved1: [u32; 251],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<LockInputBuffer>() == 1544);

#[allow(dead_code)]
#[repr(C)]
struct LockBitstream {
    version: u32,
    flags: u32,
    output_bitstream: *mut c_void,
    slice_offsets: *mut u32,
    frame_idx: u32,
    hw_encode_status: u32,
    num_slices: u32,
    bitstream_size_in_bytes: u32,
    output_time_stamp: u64,
    output_duration: u64,
    bitstream_buffer_ptr: *mut c_void,
    tail: [u32; 370],
}
const _: () = assert!(size_of::<LockBitstream>() == 1544);

#[allow(dead_code)]
#[repr(C)]
struct PicParams {
    version: u32,
    input_width: u32,
    input_height: u32,
    input_pitch: u32,
    encode_pic_flags: u32,
    frame_idx: u32,
    input_time_stamp: u64,
    input_duration: u64,
    input_buffer: *mut c_void,
    output_bitstream: *mut c_void,
    completion_event: *mut c_void,
    buffer_fmt: u32,
    picture_struct: u32,
    picture_type: u32,
    tail: [u32; 821],
}
const _: () = assert!(size_of::<PicParams>() == 3360);

#[allow(dead_code)]
#[repr(C)]
struct SequenceParamPayload {
    version: u32,
    in_buffer_size: u32,
    sps_id: u32,
    pps_id: u32,
    sps_pps_buffer: *mut c_void,
    out_sps_pps_payload_size: *mut u32,
    reserved: [u32; 250],
    reserved2: [*mut c_void; 64],
}
const _: () = assert!(size_of::<SequenceParamPayload>() == 1544);

type Fn1<T> = Option<unsafe extern "C" fn(*mut c_void, *mut T) -> Status>;
type FnPtr = Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> Status>;

#[allow(dead_code)]
#[repr(C)]
struct FunctionList {
    version: u32,
    reserved: u32,
    open_encode_session: *mut c_void,
    get_encode_guid_count: *mut c_void,
    get_encode_profile_guid_count: *mut c_void,
    get_encode_profile_guids: *mut c_void,
    get_encode_guids: *mut c_void,
    get_input_format_count: *mut c_void,
    get_input_formats: *mut c_void,
    get_encode_caps: *mut c_void,
    get_encode_preset_count: *mut c_void,
    get_encode_preset_guids: *mut c_void,
    get_encode_preset_config: *mut c_void,
    initialize_encoder: Fn1<InitializeParams>,
    create_input_buffer: Fn1<CreateInputBuffer>,
    destroy_input_buffer: FnPtr,
    create_bitstream_buffer: Fn1<CreateBitstreamBuffer>,
    destroy_bitstream_buffer: FnPtr,
    encode_picture: Fn1<PicParams>,
    lock_bitstream: Fn1<LockBitstream>,
    unlock_bitstream: FnPtr,
    lock_input_buffer: Fn1<LockInputBuffer>,
    unlock_input_buffer: FnPtr,
    get_encode_stats: *mut c_void,
    get_sequence_params: Fn1<SequenceParamPayload>,
    register_async_event: *mut c_void,
    unregister_async_event: *mut c_void,
    map_input_resource: *mut c_void,
    unmap_input_resource: *mut c_void,
    destroy_encoder: Option<unsafe extern "C" fn(*mut c_void) -> Status>,
    invalidate_ref_frames: *mut c_void,
    open_encode_session_ex:
        Option<unsafe extern "C" fn(*mut OpenSessionExParams, *mut *mut c_void) -> Status>,
    register_resource: *mut c_void,
    unregister_resource: *mut c_void,
    reconfigure_encoder: *mut c_void,
    reserved1: *mut c_void,
    create_mv_buffer: *mut c_void,
    destroy_mv_buffer: *mut c_void,
    run_motion_estimation_only: *mut c_void,
    get_last_error_string: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
    set_io_cuda_streams: *mut c_void,
    get_encode_preset_config_ex:
        Option<unsafe extern "C" fn(*mut c_void, Guid, Guid, u32, *mut PresetConfig) -> Status>,
    get_sequence_param_ex: *mut c_void,
    restore_encoder_state: *mut c_void,
    lookahead_picture: *mut c_void,
    reserved2: [*mut c_void; 275],
}
const _: () = assert!(size_of::<FunctionList>() == 2552);

struct Api {
    #[allow(dead_code)]
    lib: Library,
    list: FunctionList,
    api_version: u32,
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

const SONAME: &str = "libnvidia-encode.so.1";

fn load() -> Result<&'static Api, String> {
    API.get_or_init(|| {
        // SAFETY: `dlopen` of a driver library by SONAME. It runs the
        // library's initialisers, which is the documented way to reach NVENC.
        let lib = unsafe { Library::new(SONAME) }.map_err(|e| {
            format!(
                "mp4-h264 needs the NVIDIA encoder: could not load {SONAME} ({e}). \
                 It ships with the driver; a container needs the `video` driver \
                 capability for it to be visible."
            )
        })?;

        // SAFETY: both symbols have the signatures the header declares, and
        // the pointers they are called through live as long as `lib`.
        let (max_version, create_instance) = unsafe {
            let get_max: libloading::Symbol<unsafe extern "C" fn(*mut u32) -> Status> = lib
                .get(b"NvEncodeAPIGetMaxSupportedVersion\0")
                .map_err(|e| format!("{SONAME} has no NvEncodeAPIGetMaxSupportedVersion: {e}"))?;
            let mut v = 0u32;
            let s = get_max(&mut v);
            if s != NV_ENC_SUCCESS {
                return Err(format!(
                    "NvEncodeAPIGetMaxSupportedVersion failed: {}",
                    status_name(s)
                ));
            }
            let create: libloading::Symbol<unsafe extern "C" fn(*mut FunctionList) -> Status> = lib
                .get(b"NvEncodeAPICreateInstance\0")
                .map_err(|e| format!("{SONAME} has no NvEncodeAPICreateInstance: {e}"))?;
            (v, *create)
        };

        let (drv_major, drv_minor) = ((max_version >> 4) & 0xff, max_version & 0xf);
        if drv_major < API_MAJOR {
            return Err(format!(
                "the installed NVIDIA driver offers NVENC API {drv_major}.{drv_minor} \
                 and this build speaks {API_MAJOR}.x; upgrade the driver"
            ));
        }
        let minor = if drv_major > API_MAJOR {
            API_MINOR_MAX
        } else {
            drv_minor.min(API_MINOR_MAX)
        };
        let api_version = api_version(API_MAJOR, minor);

        let mut list: FunctionList = unsafe { std::mem::zeroed() };
        list.version = struct_version(api_version, 2);
        // SAFETY: `list` is a correctly sized, zeroed NV_ENCODE_API_FUNCTION_LIST
        // with its version word set, which is the call's whole contract.
        let s = unsafe { create_instance(&mut list) };
        if s != NV_ENC_SUCCESS {
            return Err(format!(
                "NvEncodeAPICreateInstance({API_MAJOR}.{minor}) failed: {}",
                status_name(s)
            ));
        }
        Ok(Api {
            lib,
            list,
            api_version,
        })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

struct Session {
    api: &'static Api,
    encoder: *mut c_void,
    inputs: Vec<*mut c_void>,
    outputs: Vec<*mut c_void>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: every pointer here came from this session's own create
        // calls and is destroyed exactly once, before the encoder itself.
        unsafe {
            if let Some(f) = self.api.list.destroy_input_buffer {
                for p in self.inputs.drain(..) {
                    let _ = f(self.encoder, p);
                }
            }
            if let Some(f) = self.api.list.destroy_bitstream_buffer {
                for p in self.outputs.drain(..) {
                    let _ = f(self.encoder, p);
                }
            }
            if let Some(f) = self.api.list.destroy_encoder
                && !self.encoder.is_null()
            {
                let _ = f(self.encoder);
            }
        }
    }
}

impl Session {
    fn last_error(&self) -> String {
        let Some(f) = self.api.list.get_last_error_string else {
            return String::new();
        };
        // SAFETY: the driver owns the returned string and keeps it valid
        // until the next call on this encoder; it is copied here immediately.
        unsafe {
            let p = f(self.encoder);
            if p.is_null() {
                String::new()
            } else {
                format!(" ({})", CStr::from_ptr(p).to_string_lossy())
            }
        }
    }

    fn check(&self, s: Status, what: &str) -> Result<(), String> {
        if s == NV_ENC_SUCCESS {
            Ok(())
        } else {
            Err(format!(
                "{what} failed: {}{}",
                status_name(s),
                self.last_error()
            ))
        }
    }

    fn sequence_header(&self) -> Result<Vec<u8>, String> {
        let get = self
            .api
            .list
            .get_sequence_params
            .ok_or("NVENC exposes no nvEncGetSequenceParams")?;
        let mut buf = vec![0u8; 512];
        let mut written: u32 = 0;
        let mut p: SequenceParamPayload = unsafe { std::mem::zeroed() };
        p.version = struct_version(self.api.api_version, 1);
        p.in_buffer_size = buf.len() as u32;
        p.sps_pps_buffer = buf.as_mut_ptr() as *mut c_void;
        p.out_sps_pps_payload_size = &mut written;
        // SAFETY: the driver writes at most `in_buffer_size` bytes into
        // `buf` and the payload length into `written`, both of which outlive
        // the call.
        let s = unsafe { get(self.encoder, &mut p) };
        self.check(s, "nvEncGetSequenceParams")?;
        buf.truncate(written as usize);
        Ok(buf)
    }
}

const POOL: usize = 8;

pub fn encode_h264(
    rgb: &[u8],
    width: u32,
    height: u32,
    count: u32,
    fps: f32,
    device: usize,
) -> Result<Vec<Vec<u8>>, String> {
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!(
            "H.264 4:2:0 needs even dimensions; this handle is {width}x{height}"
        ));
    }
    if width < MIN_WIDTH || height < MIN_HEIGHT {
        return Err(format!(
            "mp4-h264: NVENC will not encode H.264 below {MIN_WIDTH}x{MIN_HEIGHT}              and this handle is {width}x{height}; encode `y4m` or a still instead"
        ));
    }
    if count == 0 {
        return Err("mp4-h264: nothing to encode".to_string());
    }
    let (w, h, n) = (width as usize, height as usize, count as usize);
    let frame_bytes = w * h * 3;
    if rgb.len() != frame_bytes * n {
        return Err(format!(
            "mp4-h264: {} bytes for {n} frames of {width}x{height} (expected {})",
            rgb.len(),
            frame_bytes * n
        ));
    }

    let api = load()?;

    let ctx = cudarc::driver::CudaContext::new(device)
        .map_err(|e| format!("mp4-h264: no CUDA context on device {device}: {e}"))?;
    ctx.bind_to_thread()
        .map_err(|e| format!("mp4-h264: could not bind CUDA context: {e}"))?;

    let open = api
        .list
        .open_encode_session_ex
        .ok_or("NVENC exposes no nvEncOpenEncodeSessionEx")?;
    let mut params: OpenSessionExParams = unsafe { std::mem::zeroed() };
    params.version = struct_version(api.api_version, 1);
    params.device_type = NV_ENC_DEVICE_TYPE_CUDA;
    params.device = ctx.cu_ctx() as *mut c_void;
    params.api_version = api.api_version;
    let mut encoder: *mut c_void = std::ptr::null_mut();
    // SAFETY: `params` is a fully initialised session-open record and
    // `encoder` an out-parameter the driver writes once.
    let s = unsafe { open(&mut params, &mut encoder) };
    if s != NV_ENC_SUCCESS {
        return Err(format!(
            "nvEncOpenEncodeSessionEx failed: {}",
            status_name(s)
        ));
    }
    let mut session = Session {
        api,
        encoder,
        inputs: Vec::new(),
        outputs: Vec::new(),
    };

    let mut preset: PresetConfig = unsafe { std::mem::zeroed() };
    preset.version = struct_version_ex(api.api_version, 5);
    preset.preset_cfg.version = struct_version_ex(api.api_version, 9);
    let get_preset = api
        .list
        .get_encode_preset_config_ex
        .ok_or("NVENC exposes no nvEncGetEncodePresetConfigEx")?;
    // SAFETY: an initialised out-parameter and two GUIDs passed by value, as
    // the header declares.
    let s = unsafe {
        get_preset(
            encoder,
            H264_GUID,
            PRESET_P4_GUID,
            NV_ENC_TUNING_INFO_HIGH_QUALITY,
            &mut preset,
        )
    };
    session.check(s, "nvEncGetEncodePresetConfigEx")?;

    let (rate_num, rate_den) = frame_rate(fps);
    let mut config = preset.preset_cfg;
    config.version = struct_version_ex(api.api_version, 9);
    config.profile_guid = H264_PROFILE_HIGH_GUID;
    config.gop_length = ((rate_num as u64 * 2 / rate_den.max(1) as u64) as u32).clamp(1, 250);
    config.disable_reordering();
    config.rc_params.version = struct_version(api.api_version, 1);
    config.rc_params.rate_control_mode = NV_ENC_PARAMS_RC_VBR;
    let bitrate = ((w as u64 * h as u64 * rate_num as u64 / rate_den.max(1) as u64) / 10)
        .clamp(200_000, 60_000_000) as u32;
    config.rc_params.average_bitrate = bitrate;
    config.rc_params.max_bitrate = bitrate * 2;

    let mut init: InitializeParams = unsafe { std::mem::zeroed() };
    init.version = struct_version_ex(api.api_version, 7);
    init.encode_guid = H264_GUID;
    init.preset_guid = PRESET_P4_GUID;
    init.encode_width = width;
    init.encode_height = height;
    init.dar_width = width;
    init.dar_height = height;
    init.frame_rate_num = rate_num;
    init.frame_rate_den = rate_den;
    init.enable_ptd = 1;
    init.encode_config = &mut config;
    init.max_encode_width = width;
    init.max_encode_height = height;
    init.tuning_info = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    let initialize = api
        .list
        .initialize_encoder
        .ok_or("NVENC exposes no nvEncInitializeEncoder")?;
    // SAFETY: `init` borrows `config` for the duration of this call only; the
    // driver copies both.
    let s = unsafe { initialize(encoder, &mut init) };
    session.check(s, "nvEncInitializeEncoder")?;

    let depth = POOL.min(n);
    let create_input = api
        .list
        .create_input_buffer
        .ok_or("no nvEncCreateInputBuffer")?;
    let create_output = api
        .list
        .create_bitstream_buffer
        .ok_or("no nvEncCreateBitstreamBuffer")?;
    for _ in 0..depth {
        let mut cib: CreateInputBuffer = unsafe { std::mem::zeroed() };
        cib.version = struct_version(api.api_version, 2);
        cib.width = width;
        cib.height = height;
        cib.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
        // SAFETY: an initialised creation record; the driver writes
        // `input_buffer` and the session owns it from here.
        let s = unsafe { create_input(encoder, &mut cib) };
        session.check(s, "nvEncCreateInputBuffer")?;
        session.inputs.push(cib.input_buffer);

        let mut cbb: CreateBitstreamBuffer = unsafe { std::mem::zeroed() };
        cbb.version = struct_version(api.api_version, 1);
        let s = unsafe { create_output(encoder, &mut cbb) };
        session.check(s, "nvEncCreateBitstreamBuffer")?;
        session.outputs.push(cbb.bitstream_buffer);
    }

    let lock_input = api
        .list
        .lock_input_buffer
        .ok_or("no nvEncLockInputBuffer")?;
    let unlock_input = api
        .list
        .unlock_input_buffer
        .ok_or("no nvEncUnlockInputBuffer")?;
    let encode = api.list.encode_picture.ok_or("no nvEncEncodePicture")?;
    let lock_bs = api.list.lock_bitstream.ok_or("no nvEncLockBitstream")?;
    let unlock_bs = api.list.unlock_bitstream.ok_or("no nvEncUnlockBitstream")?;

    fn drain(
        session: &Session,
        slot: usize,
        lock: unsafe extern "C" fn(*mut c_void, *mut LockBitstream) -> Status,
        unlock: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Status,
        out: &mut Vec<Vec<u8>>,
    ) -> Result<(), String> {
        let mut lb: LockBitstream = unsafe { std::mem::zeroed() };
        lb.version = struct_version_ex(session.api.api_version, 2);
        lb.output_bitstream = session.outputs[slot];
        // SAFETY: `lb` names a bitstream buffer this session created; the
        // driver fills `bitstream_buffer_ptr` and `bitstream_size_in_bytes`,
        // and the mapping stays valid until the matching unlock below.
        let s = unsafe { lock(session.encoder, &mut lb) };
        session.check(s, "nvEncLockBitstream")?;
        let bytes = if lb.bitstream_buffer_ptr.is_null() {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    lb.bitstream_buffer_ptr as *const u8,
                    lb.bitstream_size_in_bytes as usize,
                )
                .to_vec()
            }
        };
        // SAFETY: the same buffer, unlocked exactly once per lock.
        let s = unsafe { unlock(session.encoder, session.outputs[slot]) };
        session.check(s, "nvEncUnlockBitstream")?;
        out.push(bytes);
        Ok(())
    }

    let mut pictures: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut queued: Vec<usize> = Vec::with_capacity(depth);

    for frame in 0..n {
        if queued.len() == depth {
            return Err(format!(
                "mp4-h264: the encoder is holding {depth} pictures without \
                 returning one — picture reordering should be off"
            ));
        }
        let slot = frame % depth;

        let mut lib_: LockInputBuffer = unsafe { std::mem::zeroed() };
        lib_.version = struct_version(api.api_version, 1);
        lib_.input_buffer = session.inputs[slot];
        // SAFETY: the buffer belongs to this session and is not in flight
        // (its slot is not in `queued`).
        let s = unsafe { lock_input(encoder, &mut lib_) };
        session.check(s, "nvEncLockInputBuffer")?;
        let pitch = lib_.pitch as usize;
        // SAFETY: the driver guarantees `pitch * height * 3/2` writable bytes
        // behind `buffer_data_ptr` for an NV12 surface, and the write below
        // stays inside that.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                lib_.buffer_data_ptr as *mut u8,
                pitch * h + pitch * h / 2,
            )
        };
        rgb8_to_nv12_pitched(
            &rgb[frame * frame_bytes..(frame + 1) * frame_bytes],
            w,
            h,
            pitch,
            dst,
        );
        // SAFETY: the matching unlock for the lock above.
        let s = unsafe { unlock_input(encoder, session.inputs[slot]) };
        session.check(s, "nvEncUnlockInputBuffer")?;

        let mut pp: PicParams = unsafe { std::mem::zeroed() };
        pp.version = struct_version_ex(api.api_version, 7);
        pp.input_width = width;
        pp.input_height = height;
        pp.input_pitch = pitch as u32;
        pp.frame_idx = frame as u32;
        pp.input_time_stamp = frame as u64;
        pp.input_duration = 1;
        pp.input_buffer = session.inputs[slot];
        pp.output_bitstream = session.outputs[slot];
        pp.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
        pp.picture_struct = NV_ENC_PIC_STRUCT_FRAME;
        pp.picture_type = NV_ENC_PIC_TYPE_UNKNOWN;
        // SAFETY: every pointer in `pp` is a live buffer of this session.
        let s = unsafe { encode(encoder, &mut pp) };
        queued.push(slot);
        if s == NV_ENC_SUCCESS {
            for slot in queued.drain(..) {
                drain(&session, slot, lock_bs, unlock_bs, &mut pictures)?;
            }
        } else if s != NV_ENC_ERR_NEED_MORE_INPUT {
            return Err(format!(
                "nvEncEncodePicture(frame {frame}) failed: {}{}",
                status_name(s),
                session.last_error()
            ));
        }
    }

    let mut eos: PicParams = unsafe { std::mem::zeroed() };
    eos.version = struct_version_ex(api.api_version, 7);
    eos.encode_pic_flags = NV_ENC_PIC_FLAG_EOS;
    // SAFETY: the EOS submission carries no buffers, as the header requires.
    let s = unsafe { encode(encoder, &mut eos) };
    session.check(s, "nvEncEncodePicture(EOS)")?;
    for slot in queued.drain(..) {
        drain(&session, slot, lock_bs, unlock_bs, &mut pictures)?;
    }

    if pictures.len() != n {
        return Err(format!(
            "mp4-h264: submitted {n} frames and got {} coded pictures back",
            pictures.len()
        ));
    }

    let has_sps = super::mp4::nal_units(&pictures[0])
        .iter()
        .any(|nal| nal[0] & 0x1f == 7);
    if !has_sps {
        let mut first = session.sequence_header()?;
        first.extend_from_slice(&pictures[0]);
        pictures[0] = first;
    }
    Ok(pictures)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvenc_every_case() {
        the_struct_layouts_match_the_header();
        the_version_words_are_the_headers_macros();
        reordering_is_off_after_configuring();
    }

    fn the_struct_layouts_match_the_header() {
        assert_eq!(size_of::<Guid>(), 16);
        assert_eq!(size_of::<RcParams>(), 128);
        assert_eq!(size_of::<Config>(), 3584);
        assert_eq!(size_of::<PresetConfig>(), 5128);
        assert_eq!(size_of::<InitializeParams>(), 1800);
        assert_eq!(size_of::<PicParams>(), 3360);
        assert_eq!(size_of::<FunctionList>(), 2552);
        assert_eq!(std::mem::offset_of!(Config, rc_params), 40);
        assert_eq!(std::mem::offset_of!(Config, codec_config), 168);
        assert_eq!(std::mem::offset_of!(RcParams, tail), 40);
    }

    fn the_version_words_are_the_headers_macros() {
        let api = api_version(13, 1);
        assert_eq!(api, 0x0100_000d);
        assert_eq!(struct_version_ex(api, 9), 0xf109_000d, "NV_ENC_CONFIG_VER");
        assert_eq!(
            struct_version_ex(api, 7),
            0xf107_000d,
            "NV_ENC_INITIALIZE_PARAMS_VER"
        );
        assert_eq!(
            struct_version(api, 2),
            0x7102_000d,
            "NV_ENCODE_API_FUNCTION_LIST_VER"
        );
    }

    fn reordering_is_off_after_configuring() {
        let mut c: Config = unsafe { std::mem::zeroed() };
        c.frame_interval_p = 3;
        c.rc_params.flags = RC_FLAG_LOOKAHEAD | RC_FLAG_EXT_LOOKAHEAD | 0x8;
        c.rc_params.tail[12] = 0x0020_0010;
        c.disable_reordering();
        assert_eq!(c.frame_interval_p, 1);
        assert_eq!(c.rc_params.flags & RC_FLAG_LOOKAHEAD, 0);
        assert_eq!(c.rc_params.flags & RC_FLAG_EXT_LOOKAHEAD, 0);
        assert_ne!(c.rc_params.flags & RC_FLAG_ZERO_REORDER_DELAY, 0);
        assert_eq!(c.rc_params.tail[12], 0, "lookaheadDepth cleared");
        assert_eq!(
            c.rc_params.flags & 0x8,
            0x8,
            "enableAQ left as the preset had it"
        );
    }
}

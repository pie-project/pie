//! The JNI core of the Kotlin `PieServer` (`org.pieproject.server.NativeCore`):
//! a `worker::Server` behind a `jlong` handle. Failures throw
//! `org.pieproject.client.PieException$Server`.

use std::path::{Path, PathBuf};

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject, JString};
use jni::sys::{jint, jlong, jobjectArray};
use worker::Server;
use worker::embedded::Settings;

const EXCEPTION: &str = "org/pieproject/client/PieException$Server";

/// Runs `f`, throwing its error into the JVM and returning `failed` in its place.
fn throwing<'local, T>(
    env: &mut JNIEnv<'local>,
    failed: T,
    f: impl FnOnce(&mut JNIEnv<'local>) -> anyhow::Result<T>,
) -> T {
    match f(env) {
        Ok(value) => value,
        Err(error) => {
            if !env.exception_check().unwrap_or(false) {
                let _ = env.throw_new(EXCEPTION, format!("{error:#}"));
            }
            failed
        }
    }
}

fn string(env: &mut JNIEnv, value: &JString) -> anyhow::Result<String> {
    Ok(env.get_string(value)?.into())
}

fn optional_string(env: &mut JNIEnv, value: &JString) -> anyhow::Result<Option<String>> {
    if value.is_null() {
        Ok(None)
    } else {
        string(env, value).map(Some)
    }
}

/// The server a handle from `start` names; handles are never freed, since
/// the runtime boots once per process and `shutdown` releases what it holds.
fn server<'a>(handle: jlong) -> anyhow::Result<&'a Server> {
    unsafe { (handle as *const Server).as_ref() }.ok_or_else(|| anyhow::anyhow!("no server"))
}

fn boot(
    artifact: &Path,
    settings: Option<&str>,
    home: &Path,
    listen: Option<&str>,
) -> anyhow::Result<Server> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();

    let settings = settings.map_or_else(|| Ok(Settings::default()), Settings::parse)?;
    Server::embed(
        artifact,
        &settings,
        home,
        listen.map(str::parse).transpose()?,
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_start(
    mut env: JNIEnv,
    _: JClass,
    artifact: JString,
    config: JString,
    home: JString,
    listen: JString,
) -> jlong {
    throwing(&mut env, 0, |env| {
        let artifact = PathBuf::from(string(env, &artifact)?);
        let config = optional_string(env, &config)?;
        let home = PathBuf::from(string(env, &home)?);
        let listen = optional_string(env, &listen)?;
        let server = boot(&artifact, config.as_deref(), &home, listen.as_deref())?;
        Ok(Box::into_raw(Box::new(server)) as jlong)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_summary<'a>(
    mut env: JNIEnv<'a>,
    _: JClass,
    handle: jlong,
) -> JString<'a> {
    throwing(&mut env, JObject::null().into(), |env| {
        let summary = serde_json::to_string(server(handle)?.summary())?;
        Ok(env.new_string(summary)?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_listenAddr<'a>(
    mut env: JNIEnv<'a>,
    _: JClass,
    handle: jlong,
) -> JString<'a> {
    throwing(&mut env, JObject::null().into(), |env| {
        Ok(match server(handle)?.listen_addr() {
            Some(addr) => env.new_string(addr.to_string())?,
            None => JObject::null().into(),
        })
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_install<'a>(
    mut env: JNIEnv<'a>,
    _: JClass,
    handle: jlong,
    program: JByteArray,
    file: JString,
    version: JString,
) -> JString<'a> {
    throwing(&mut env, JObject::null().into(), |env| {
        let program = env.convert_byte_array(&program)?;
        let file = string(env, &file)?;
        let version = optional_string(env, &version)?;
        let name = server(handle)?
            .install(program, &file, version.as_deref())
            .map_err(|e| e.context("install"))?;
        Ok(env.new_string(name)?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_installLanguage(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
    language: JString,
    component: JByteArray,
) {
    throwing(&mut env, (), |env| {
        let language = string(env, &language)?;
        let component = env.convert_byte_array(&component)?;
        server(handle)?
            .install_language(&language, component)
            .map_err(|e| e.context("install language"))?;
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_openSession(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
) -> jint {
    throwing(&mut env, 0, |_| {
        let session = server(handle)?
            .open_session()
            .map_err(|e| e.context("open session"))?;
        Ok(session as jint)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_closeSession(
    _: JNIEnv,
    _: JClass,
    handle: jlong,
    session: jint,
) {
    if let Ok(server) = server(handle) {
        server.close_session(session as u32);
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_sendFrame(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
    session: jint,
    frame: JByteArray,
) {
    throwing(&mut env, (), |env| {
        let frame = env.convert_byte_array(&frame)?;
        server(handle)?.send_frame(session as u32, &frame)
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_recvFrames(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
    session: jint,
    max_wait_ms: jint,
    max: jint,
) -> jobjectArray {
    throwing(&mut env, std::ptr::null_mut(), |env| {
        let frames = server(handle)?.recv_frames(
            session as u32,
            max_wait_ms.max(0) as u64,
            max.max(1) as usize,
        )?;
        let array = env.new_object_array(frames.len() as i32, "[B", JObject::null())?;
        for (i, frame) in frames.iter().enumerate() {
            let bytes = env.byte_array_from_slice(frame)?;
            env.set_object_array_element(&array, i as i32, &bytes)?;
            env.delete_local_ref(bytes)?;
        }
        Ok(array.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_pieproject_server_NativeCore_shutdown(
    _: JNIEnv,
    _: JClass,
    handle: jlong,
) {
    if let Ok(server) = server(handle) {
        server.shutdown();
    }
}

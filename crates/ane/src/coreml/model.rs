use std::ffi::{CString, c_char, c_int, c_long, c_void};

unsafe extern "C" {
    fn pie_coreml_load(
        path: *const c_char,
        function: *const c_char,
        units: c_int,
        err: *mut c_char,
        cap: c_int,
    ) -> *mut c_void;
    fn pie_coreml_free(model: *mut c_void);
    fn pie_coreml_predict(
        model: *mut c_void,
        in_name: *const c_char,
        input: *mut c_void,
        rows: c_long,
        width_in: c_long,
        out_name: *const c_char,
        output: *mut c_void,
        width_out: c_long,
        err: *mut c_char,
        cap: c_int,
    ) -> c_int;
}

#[derive(Clone, Copy)]
pub enum Units {
    NeuralEngine = 0,
    Gpu = 2,
}

pub struct Model(*mut c_void);

unsafe impl Send for Model {}

fn text(err: &[c_char]) -> String {
    unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl Model {
    pub fn load(path: &std::path::Path, function: &str, units: Units) -> Result<Model, String> {
        let path = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let function = CString::new(function).map_err(|e| e.to_string())?;
        let mut err = [0 as c_char; 512];
        let model = unsafe {
            pie_coreml_load(
                path.as_ptr(),
                function.as_ptr(),
                units as c_int,
                err.as_mut_ptr(),
                512,
            )
        };
        if model.is_null() {
            return Err(text(&err));
        }
        Ok(Model(model))
    }

    pub unsafe fn predict(
        &self,
        names: (&CString, &CString),
        input: *mut c_void,
        output: *mut c_void,
        rows: u32,
        width: u32,
    ) -> Result<(), String> {
        let mut err = [0 as c_char; 512];
        let status = unsafe {
            pie_coreml_predict(
                self.0,
                names.0.as_ptr(),
                input,
                c_long::from(rows),
                c_long::from(width),
                names.1.as_ptr(),
                output,
                c_long::from(width),
                err.as_mut_ptr(),
                512,
            )
        };
        if status == 0 { Ok(()) } else { Err(text(&err)) }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { pie_coreml_free(self.0) };
    }
}

pub mod data;
pub mod games;
pub mod maths;
pub mod nn;
pub mod testing;

use serde::{Deserialize, Serialize};
use std::ffi::{CString, CStr};
use crate::testing::predict::predict_from_pos;
use libc::c_char;

#[derive(Deserialize, Serialize)]
pub struct PredictionRequest {
    pub x: Vec<i32>,
}

#[derive(Serialize)]
pub struct PredictionResponse {
    pub d: usize,
    pub a: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn predict(model_path: *const c_char, input_json: *const c_char) -> *mut c_char {
    let c_model_path = unsafe { CStr::from_ptr(model_path) }.to_str().unwrap();
    let c_input_json = unsafe { CStr::from_ptr(input_json) }.to_str().unwrap();

    let request: PredictionRequest = serde_json::from_str(c_input_json).unwrap();
    let (d, a) = predict_from_pos(c_model_path, request.x);

    let response = PredictionResponse { d, a };
    let output_json = serde_json::to_string(&response).unwrap();

    CString::new(output_json).unwrap().into_raw()
}
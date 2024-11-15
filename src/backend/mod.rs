#[cfg(not(feature = "gcu"))]
mod cache;
#[cfg(not(feature = "gcu"))]
pub mod gptq;
#[cfg(not(feature = "gcu"))]
mod paged_attention;

const COPY_BLOCKS_KERNEL_NAME: &str = "copy_blocks_kernel";

pub fn get_or_load_func(
    ptx_file: &'static str,
    kernel_base: &str,
    dtype: DType,
    suffix: Option<&str>,
    device: &CudaDevice,
) -> Result<CudaFunction, APIError> {
    let spec = match dtype {
        DType::U8 => "_u8",
        DType::I8 => "_i8",
        DType::U32 => "_u32",
        DType::I64 => "_i64",
        DType::BF16 => "_bf16",
        DType::F16 => "_f16",
        DType::F32 => "_f32",
        DType::F64 => "_f64",
    };
    let spec = if let Some(suffix) = suffix {
        spec.to_owned() + suffix
    } else {
        spec.to_owned()
    };
    let kernel = kernel_base.to_owned() + &spec;
    device
        .get_or_load_func(&kernel, ptx_file)
        .map_err(APIError::from)
}

#[repr(transparent)]
struct Conjoined<'a, T, R> {
    raw: *mut T,
    _ref: PhantomData<&'a mut R>,
}

impl<'a, T, R> Conjoined<'a, T, R> {
    fn new(raw: NonNull<T>, _ref: &'a mut R) -> Self {
        Self {
            raw: raw.as_ptr(),
            _ref: PhantomData,
        }
    }
}

/// According to the docs: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
/// Each of the kernel params (*mut c_void) "must point to a region of memory from which the actual kernel parameter will be copied".
/// This means that we must return a pointer to our pointer.
///
/// ## Safety
/// - The returned pointer **must not** outlive the &self reference. Otherwise, a dangling pointer is created.

#[cfg(not(feature = "gcu"))]
unsafe impl<'a, T, R> DeviceRepr for Conjoined<'a, T, R> {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        addr_of!(self.raw) as *mut _
    }
}

#[cfg(not(feature = "gcu"))]
pub use cache::*;

#[cfg(not(feature = "gcu"))]
use candle_core::{
    cuda_backend::cudarc::driver::{CudaFunction, DeviceRepr},
    CudaDevice, DType,
};

#[cfg(feature = "gcu")]
use candle_core::{
    gcu_backend::ubridge::prelude::GcuFunction as CudaFunction, DType, GcuDevice as CudaDevice,
};

#[cfg(not(feature = "gcu"))]
pub use gptq::*;
#[cfg(not(feature = "gcu"))]
pub use paged_attention::*;

#[cfg(feature = "gcu")]
pub use candle_paged_attention::*;

pub use std::ops::Deref;
use std::{
    marker::PhantomData,
    ptr::{addr_of, NonNull},
};

use crate::openai::responses::APIError;

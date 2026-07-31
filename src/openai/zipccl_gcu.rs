//! GCU ZipCCL collectives.
//!
//! The codec is the lossless top-seven exponent format described in
//! `zipccl.md`.  Metadata is synchronized once per collective because ECCL's
//! point-to-point API takes a fixed element count.  This module is selected
//! only for prefill BF16/F16 tensors above the configured threshold; decode
//! and unsupported shapes continue to use native ECCL.

use candle_core::gcu_backend::ubridge::eccl::Comm;
use candle_core::gcu_backend::ubridge::gcu_launch::{DeviceCopy, GcuLaunchAsync};
use candle_core::gcu_backend::ubridge::gcu_slice::GcuSlice;
use candle_core::gcu_backend::ubridge::{self, device_ptr::DevicePtr, device_ptr::DevicePtrMut};
use candle_core::gcu_backend::WrapErr;
use candle_core::{DType, Result};
use half::{bf16, f16};
use std::sync::Arc;

const BLOCK: usize = 256;
const HEADER: usize = 16;

#[derive(Clone, Copy, Debug)]
enum CodecDType {
    Bf16,
    F16,
}

impl CodecDType {
    fn id(self) -> i32 {
        match self {
            Self::Bf16 => 0,
            Self::F16 => 1,
        }
    }
}

fn codec_dtype(dtype: DType) -> Result<CodecDType> {
    match dtype {
        DType::BF16 => Ok(CodecDType::Bf16),
        DType::F16 => Ok(CodecDType::F16),
        other => candle_core::bail!("GCU ZipCCL supports BF16/F16 only, got {other:?}"),
    }
}

fn blocks(n: usize) -> usize {
    (n + BLOCK - 1) / BLOCK
}

fn bitplane_bytes(n: usize) -> usize {
    (n + 7) / 8
}

fn static_bytes(n: usize, dtype: CodecDType) -> usize {
    let planes = if matches!(dtype, CodecDType::F16) {
        6
    } else {
        3
    };
    (HEADER + n + planes * bitplane_bytes(n) + 15) & !15
}

fn transfer_capacity(n: usize, dtype: CodecDType) -> usize {
    (static_bytes(n, dtype) + blocks(n) * 4 + n + 15) & !15
}

fn check_count(n: usize, world: usize, dtype: DType) -> Result<CodecDType> {
    if n == 0 {
        candle_core::bail!("GCU ZipCCL does not support empty tensors")
    }
    if world <= 1 {
        candle_core::bail!("GCU ZipCCL is not needed for a single rank")
    }
    codec_dtype(dtype)
}

fn launch_config(raw: &ubridge::gcu_device::GcuDevice) -> ubridge::gcu_launch::GcuLaunchConfig {
    // Keep the device's native 2-block × 12-SIP configuration.  The kernels
    // stride over the complete tensor instead of manufacturing CUDA-like
    // grids, which would be invalid on this platform.
    raw.launch_cfg
}

type RawDevicePtr = ubridge::gcu_slice::driv::topsDeviceptr_t;

fn launch_count(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    counts: &GcuSlice<u32>,
    top7: &GcuSlice<u32>,
    n: usize,
    dtype: CodecDType,
) -> Result<()> {
    let func = raw.get_or_load_func("zipccl_count", ubridge::ZIPCCL).w()?;
    let cfg = launch_config(raw);
    unsafe {
        func.launch(
            &cfg,
            (
                input,
                counts.device_ptr(),
                top7.device_ptr(),
                n as i32,
                dtype.id(),
            ),
        )
    }
    .w()
}

fn select_top7(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    histograms: &GcuSlice<u32>,
    top7: &GcuSlice<u32>,
    n: usize,
    dtype: CodecDType,
) -> Result<()> {
    let histogram = raw
        .get_or_load_func("zipccl_histogram", ubridge::ZIPCCL)
        .w()?;
    let cfg = launch_config(raw);
    unsafe { histogram.launch(&cfg, (input, histograms.device_ptr(), n as i32, dtype.id())) }
        .w()?;

    let select = raw
        .get_or_load_func("zipccl_select_top7", ubridge::ZIPCCL)
        .w()?;
    unsafe {
        select.launch(
            &cfg,
            (histograms.device_ptr(), top7.device_ptr(), dtype.id()),
        )
    }
    .w()
}

fn compress(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    n: usize,
    dtype: CodecDType,
) -> Result<(GcuSlice<u8>, usize)> {
    let capacity = transfer_capacity(n, dtype);
    let output = raw.alloc::<u8>(capacity).w()?;
    // Store the selected exponents as 32 naturally aligned words.  This is a
    // 128-byte DTE transfer, but avoids the unsupported byte-vector ABI on
    // gcu300.
    let top7 = raw.alloc::<u32>(32).w()?;
    let histograms = raw.alloc::<u32>(24 * 256).w()?;
    select_top7(raw, input, &histograms, &top7, n, dtype)?;
    // The fixed packet still needs its header initialized, but does not need
    // dynamic per-block counts or prefix offsets.  The metadata kernel keeps
    // this single-SIP DTE operation out of the multi-SIP compressor.
    let metadata = raw.alloc::<u32>(1).w()?;
    let prefix = raw.get_or_load_func("zipccl_prefix", ubridge::ZIPCCL).w()?;
    let prefix_cfg = launch_config(raw);
    unsafe {
        prefix.launch(
            &prefix_cfg,
            (
                output.device_ptr(),
                metadata.device_ptr(),
                top7.device_ptr(),
                n as i32,
                dtype.id(),
                blocks(n) as i32,
            ),
        )
    }
    .w()?;
    let pack = raw
        .get_or_load_func("zipccl_compress", ubridge::ZIPCCL)
        .w()?;
    let pack_cfg = launch_config(raw);
    unsafe {
        pack.launch(
            &pack_cfg,
            (
                input,
                output.device_ptr(),
                top7.device_ptr(),
                n as i32,
                dtype.id(),
            ),
        )
    }
    .w()?;

    // The receiver can determine every fixed outlier slot from n and the
    // 256-value tile size.  ECCL receives the same Ubridge stream as the
    // codec launches, so stream ordering makes a host synchronization both
    // unnecessary and very expensive in the repeated prefill schedule.
    Ok((output, capacity))
}

fn exchange_sizes(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    local: &[usize],
) -> Result<Vec<usize>> {
    let local_u32: Vec<u32> = local
        .iter()
        .copied()
        .map(|v| u32::try_from(v).map_err(candle_core::Error::wrap))
        .collect::<Result<_>>()?;
    let send = raw.htod_copy(local_u32).w()?;
    let mut recv = raw.alloc::<u32>(local.len() * comm.world_size()).w()?;
    comm.all_gather(&send, &mut recv)
        .map_err(candle_core::Error::debug)?;
    raw.dtoh_sync_copy(&recv)
        .w()
        .map(|sizes| sizes.into_iter().map(|v| v as usize).collect())
}

fn exchange_fixed_payload(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    payload: &GcuSlice<u8>,
    payload_size: usize,
) -> Result<GcuSlice<u8>> {
    let world = comm.world_size();
    let mut received = raw.alloc::<u8>(payload_size * world).w()?;
    raw.dtod_copy(
        &payload.slice(..payload_size),
        &mut received.slice_mut(comm.rank() * payload_size..(comm.rank() + 1) * payload_size),
    )
    .w()?;

    ubridge::eccllib::group_start().map_err(candle_core::Error::debug)?;
    for peer in 0..world {
        if peer == comm.rank() {
            continue;
        }
        let send = payload.slice(..payload_size);
        let mut recv = received.slice_mut(peer * payload_size..(peer + 1) * payload_size);
        comm.send(&send, peer as i32)
            .map_err(candle_core::Error::debug)?;
        comm.recv(&mut recv, peer as i32)
            .map_err(candle_core::Error::debug)?;
    }
    ubridge::eccllib::group_end().map_err(candle_core::Error::debug)?;
    Ok(received)
}

fn exchange_fixed_chunks(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    send_pack: &GcuSlice<u8>,
    chunk_size: usize,
) -> Result<GcuSlice<u8>> {
    let world = comm.world_size();
    let mut received = raw.alloc::<u8>(chunk_size * (world - 1)).w()?;
    ubridge::eccllib::group_start().map_err(candle_core::Error::debug)?;
    let mut recv_index = 0;
    for peer in 0..world {
        if peer == comm.rank() {
            continue;
        }
        let send = send_pack.slice(peer * chunk_size..(peer + 1) * chunk_size);
        let mut recv = received.slice_mut(recv_index * chunk_size..(recv_index + 1) * chunk_size);
        comm.send(&send, peer as i32)
            .map_err(candle_core::Error::debug)?;
        comm.recv(&mut recv, peer as i32)
            .map_err(candle_core::Error::debug)?;
        recv_index += 1;
    }
    ubridge::eccllib::group_end().map_err(candle_core::Error::debug)?;
    Ok(received)
}

fn decompress<O: DeviceCopy, P: DevicePtr<u8>, R: DevicePtrMut<O>>(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    payload: &P,
    output: &mut R,
    n: usize,
    dtype: CodecDType,
) -> Result<()> {
    let func = raw
        .get_or_load_func("zipccl_decompress", ubridge::ZIPCCL)
        .w()?;
    let cfg = launch_config(raw);
    unsafe {
        func.launch(
            &cfg,
            (
                payload.device_ptr(),
                output.device_ptr_mut(),
                n as i32,
                dtype.id(),
            ),
        )
    }
    .w()
}

fn decompress_add_f32<P: DevicePtr<u8>>(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    payload: &P,
    accumulator: &mut GcuSlice<f32>,
    n: usize,
    dtype: CodecDType,
) -> Result<()> {
    let func = raw
        .get_or_load_func("zipccl_decompress_add_f32", ubridge::ZIPCCL)
        .w()?;
    let cfg = launch_config(raw);
    unsafe {
        func.launch(
            &cfg,
            (
                payload.device_ptr(),
                accumulator.device_ptr_mut(),
                n as i32,
                dtype.id(),
            ),
        )
    }
    .w()
}

fn add_f32(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: &GcuSlice<f32>,
    accumulator: &mut GcuSlice<f32>,
    n: usize,
) -> Result<()> {
    let func = raw
        .get_or_load_func("zipccl_add_f32", ubridge::ZIPCCL)
        .w()?;
    let cfg = launch_config(raw);
    unsafe {
        func.launch(
            &cfg,
            (input.device_ptr(), accumulator.device_ptr_mut(), n as i32),
        )
    }
    .w()
}

fn to_f32(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    output: &mut GcuSlice<f32>,
    n: usize,
    dtype: CodecDType,
) -> Result<()> {
    let func = raw.get_or_load_func("zipccl_to_f32", ubridge::ZIPCCL).w()?;
    let cfg = launch_config(raw);
    unsafe { func.launch(&cfg, (input, output.device_ptr_mut(), n as i32, dtype.id())) }.w()
}

fn from_f32<O: DeviceCopy, R: DevicePtrMut<O>>(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: &GcuSlice<f32>,
    output: &mut R,
    n: usize,
    dtype: CodecDType,
) -> Result<()> {
    let func = raw
        .get_or_load_func("zipccl_from_f32", ubridge::ZIPCCL)
        .w()?;
    let cfg = launch_config(raw);
    unsafe {
        func.launch(
            &cfg,
            (
                input.device_ptr(),
                output.device_ptr_mut(),
                n as i32,
                dtype.id(),
            ),
        )
    }
    .w()
}

fn all_gather_impl<O: DeviceCopy>(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    input: RawDevicePtr,
    n: usize,
    dtype: CodecDType,
) -> Result<GcuSlice<O>> {
    let (payload, exact) = compress(raw, input, n, dtype)?;
    let received = exchange_fixed_payload(raw, comm, &payload, exact)?;
    let mut output = raw.alloc::<O>(n * comm.world_size()).w()?;
    for rank in 0..comm.world_size() {
        let view = received.slice(rank * exact..(rank + 1) * exact);
        let mut out = output.slice_mut(rank * n..(rank + 1) * n);
        decompress::<O, _, _>(raw, &view, &mut out, n, dtype)?;
    }
    Ok(output)
}

fn all_reduce_impl<O: DeviceCopy>(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    input: RawDevicePtr,
    n: usize,
    dtype: CodecDType,
) -> Result<GcuSlice<O>> {
    let world = comm.world_size();
    if n % world != 0 {
        candle_core::bail!("GCU ZipCCL AllReduce requires n divisible by world size")
    }
    let chunk = n / world;
    let payload_size = transfer_capacity(chunk, dtype);
    let mut send_pack = raw.alloc::<u8>(world * payload_size).w()?;
    for peer in 0..world {
        let src = offset_ptr::<u16>(input, peer * chunk);
        let (payload, exact) = compress(raw, src, chunk, dtype)?;
        debug_assert_eq!(exact, payload_size);
        raw.dtod_copy(
            &payload.slice(..payload_size),
            &mut send_pack.slice_mut(peer * payload_size..(peer + 1) * payload_size),
        )
        .w()?;
    }
    let received = exchange_fixed_chunks(raw, comm, &send_pack, payload_size)?;

    let local = offset_ptr::<u16>(input, comm.rank() * chunk);
    let mut accumulator = raw.alloc_zeros::<f32>(chunk).w()?;
    to_f32(raw, local, &mut accumulator, chunk, dtype)?;
    let mut decoded_raw = raw.alloc::<u16>(chunk).w()?;
    let mut decoded = raw.alloc::<f32>(chunk).w()?;
    let mut recv_index = 0;
    for src in 0..world {
        if src == comm.rank() {
            continue;
        }
        let view = received.slice(recv_index * payload_size..(recv_index + 1) * payload_size);
        decompress(raw, &view, &mut decoded_raw, chunk, dtype)?;
        to_f32(raw, decoded_raw.device_ptr(), &mut decoded, chunk, dtype)?;
        add_f32(raw, &decoded, &mut accumulator, chunk)?;
        recv_index += 1;
    }

    let mut reduced = raw.alloc::<O>(chunk).w()?;
    from_f32(raw, &accumulator, &mut reduced, chunk, dtype)?;
    let (payload, exact) = compress(raw, reduced.device_ptr(), chunk, dtype)?;
    let received = exchange_fixed_payload(raw, comm, &payload, exact)?;
    let mut output = raw.alloc::<O>(n).w()?;
    for rank in 0..world {
        let view = received.slice(rank * exact..(rank + 1) * exact);
        let mut out = output.slice_mut(rank * chunk..(rank + 1) * chunk);
        decompress::<O, _, _>(raw, &view, &mut out, chunk, dtype)?;
    }
    Ok(output)
}

fn offset_ptr<T>(input: RawDevicePtr, elements: usize) -> RawDevicePtr {
    (input as usize + elements * std::mem::size_of::<T>()) as RawDevicePtr
}

pub fn active() -> bool {
    matches!(
        std::env::var("CANDLE_VLLM_ZIPCCL").as_deref(),
        Ok("1" | "true" | "TRUE")
    )
}

pub fn eligible(n: usize, world: usize, dtype: DType) -> bool {
    if !active() || world <= 1 || n < 131_072 || !matches!(dtype, DType::BF16 | DType::F16) {
        return false;
    }
    n % world == 0
}

pub fn all_reduce_bf16(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    input: &impl DevicePtr<bf16>,
    n: usize,
) -> Result<GcuSlice<bf16>> {
    check_count(n, comm.world_size(), DType::BF16)?;
    all_reduce_impl(raw, comm, input.device_ptr(), n, CodecDType::Bf16)
}

pub fn all_reduce_f16(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    input: &impl DevicePtr<f16>,
    n: usize,
) -> Result<GcuSlice<f16>> {
    check_count(n, comm.world_size(), DType::F16)?;
    all_reduce_impl(raw, comm, input.device_ptr(), n, CodecDType::F16)
}

pub fn all_gather_bf16(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    input: &impl DevicePtr<bf16>,
    n: usize,
) -> Result<GcuSlice<bf16>> {
    check_count(n, comm.world_size(), DType::BF16)?;
    all_gather_impl(raw, comm, input.device_ptr(), n, CodecDType::Bf16)
}

pub fn all_gather_f16(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    input: &impl DevicePtr<f16>,
    n: usize,
) -> Result<GcuSlice<f16>> {
    check_count(n, comm.world_size(), DType::F16)?;
    all_gather_impl(raw, comm, input.device_ptr(), n, CodecDType::F16)
}

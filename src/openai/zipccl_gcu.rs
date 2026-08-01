//! GCU ZipCCL collectives.
//!
//! The codec is the lossless top-seven exponent format described in
//! `zipccl.md`.  Metadata is synchronized once per collective because ECCL's
//! point-to-point API takes a fixed element count.  This module is selected
//! only for prefill BF16/F16 tensors above the configured threshold; decode
//! and unsupported shapes continue to use native ECCL.

use candle_core::gcu_backend::ubridge::eccl::{Comm, EcclType};
use candle_core::gcu_backend::ubridge::gcu_launch::{DeviceCopy, GcuLaunchAsync};
use candle_core::gcu_backend::ubridge::gcu_slice::GcuSlice;
use candle_core::gcu_backend::ubridge::{self, device_ptr::DevicePtr, device_ptr::DevicePtrMut};
use candle_core::gcu_backend::WrapErr;
use candle_core::{DType, Result};
use half::{bf16, f16};
use std::sync::{Arc, Mutex, OnceLock};

const BLOCK: usize = 256;
const HEADER: usize = 16;
const TOP7_REFRESH_CALLS: u32 = 24;

struct CachedTop7 {
    words: Vec<u32>,
    uses: u32,
}

static TOP7_CACHE: OnceLock<Mutex<[Option<CachedTop7>; 2]>> = OnceLock::new();

/// Device-side result of a batched compression launch.
///
/// Compression and compaction leave `sizes` on the same GCU stream as the
/// payload.  Keeping that buffer device-resident is important: callers can
/// enqueue more device work after compression without an accidental host
/// synchronization hidden inside the codec.  The host sizes are materialized
/// only immediately before ECCL variable-count transfers are posted.
struct CompressedChunks {
    payload: GcuSlice<u8>,
    sizes: GcuSlice<u32>,
    capacity: usize,
    num_chunks: usize,
}

impl CompressedChunks {
    /// Read the exact packet sizes for the next host-count ECCL operation.
    ///
    /// This synchronization cannot currently be removed completely: ECCL's
    /// send/recv ABI takes a host `usize` count, while the compaction kernel
    /// produces a data-dependent count on the device.  The readback is kept
    /// here, after all compression kernels have been enqueued, so there is one
    /// ordered metadata fence per collective rather than one in each codec
    /// helper or packet operation.
    fn exact_sizes(
        &self,
        raw: &Arc<ubridge::gcu_device::GcuDevice>,
    ) -> Result<Vec<usize>> {
        let words = raw.dtoh_sync_copy(&self.sizes).w()?;
        if words.len() != self.num_chunks * 4 {
            candle_core::bail!(
                "invalid ZipCCL size metadata length: got {}, expected {}",
                words.len(),
                self.num_chunks * 4
            )
        }
        let exact: Vec<usize> = words
            .chunks_exact(4)
            .map(|chunk| chunk[0] as usize)
            .collect();
        if exact.iter().any(|&size| size == 0 || size > self.capacity) {
            candle_core::bail!("invalid ZipCCL compressed packet size metadata")
        }
        Ok(exact)
    }
}

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

fn base_bytes(n: usize) -> usize {
    (blocks(n) * std::mem::size_of::<u32>() + 15) & !15
}

fn transfer_capacity(n: usize, dtype: CodecDType) -> usize {
    let fixed_base = static_bytes(n, dtype) + base_bytes(n);
    let scratch_base = (fixed_base + n + 15) & !15;
    let tile_count = (blocks(n) + 15) / 16;
    (scratch_base + tile_count * 4096 + 15) & !15
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

fn top7_for(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    n: usize,
    dtype: CodecDType,
) -> Result<GcuSlice<u32>> {
    let slot = dtype.id() as usize;
    let cache = TOP7_CACHE.get_or_init(|| Mutex::new([None, None]));
    let cached = {
        let mut guard = cache
            .lock()
            .map_err(|_| candle_core::Error::Msg("ZipCCL top7 cache lock poisoned".into()))?;
        if let Some(entry) = guard[slot].as_mut() {
            if entry.uses < TOP7_REFRESH_CALLS {
                entry.uses += 1;
                Some(entry.words.clone())
            } else {
                None
            }
        } else {
            None
        }
    };
    if let Some(words) = cached {
        return raw.htod_copy(words).w();
    }

    // Keep the device buffer large enough for the GCU DTE granule. Only the
    // first seven words are meaningful, but the kernel reads the full table.
    let top7 = raw.alloc::<u32>(256).w()?;
    let histograms = raw.alloc::<u32>(24 * 256).w()?;
    select_top7(raw, input, &histograms, &top7, n, dtype)?;
    let words = raw.dtoh_sync_copy(&top7).w()?;
    let mut guard = cache
        .lock()
        .map_err(|_| candle_core::Error::Msg("ZipCCL top7 cache lock poisoned".into()))?;
    guard[slot] = Some(CachedTop7 { words, uses: 0 });
    Ok(top7)
}

/// Select the top7 window and compress `num_chunks` contiguous chunks of
/// `n_chunk` values each into worst-case-capacity packets. A tiled DTE
/// compaction pass then returns the exact dynamic packet size for every chunk.
fn compress_chunks(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    n_chunk: usize,
    num_chunks: usize,
    dtype: CodecDType,
) -> Result<CompressedChunks> {
    let capacity = transfer_capacity(n_chunk, dtype);
    let mut output = raw.alloc::<u8>(capacity * num_chunks).w()?;
    let mut sizes = raw.alloc::<u32>(num_chunks * 4).w()?;
    let top7 = top7_for(raw, input, n_chunk, dtype)?;
    let pack_cfg = launch_config(raw);
    // The compressor writes vectorized sign/plane data and fixed exponent
    // slots. The compactor scans those tiles and emits the dynamic outlier
    // stream; its single-packet mapping uses all twelve SIPs in block zero.
    let pack = raw
        .get_or_load_func("zipccl_compress_all", ubridge::ZIPCCL)
        .w()?;
    unsafe {
        pack.launch(
            &pack_cfg,
            (
                input,
                output.device_ptr(),
                top7.device_ptr(),
                n_chunk as i32,
                num_chunks as i32,
                dtype.id(),
            ),
        )
    }
    .w()?;

    let compact = raw
        .get_or_load_func("zipccl_compact_all", ubridge::ZIPCCL)
        .w()?;
    unsafe {
        compact.launch(
            &pack_cfg,
            (
                output.device_ptr_mut(),
                sizes.device_ptr_mut(),
                n_chunk as i32,
                num_chunks as i32,
                dtype.id(),
            ),
        )
    }
    .w()?;

    Ok(CompressedChunks {
        payload: output,
        sizes,
        capacity,
        num_chunks,
    })
}

fn compress(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    input: RawDevicePtr,
    n: usize,
    dtype: CodecDType,
) -> Result<CompressedChunks> {
    compress_chunks(raw, input, n, 1, dtype)
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

fn exchange_variable_payload(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    payload: &GcuSlice<u8>,
    local_size: usize,
    all_sizes: &[usize],
) -> Result<GcuSlice<u8>> {
    let world = comm.world_size();
    if all_sizes.len() != world {
        candle_core::bail!("invalid ZipCCL AllGather size table")
    }
    let mut offsets = vec![0usize; world];
    for rank in 1..world {
        offsets[rank] = offsets[rank - 1] + all_sizes[rank - 1];
    }
    let total = offsets[world - 1] + all_sizes[world - 1];
    let mut received = raw.alloc::<u8>(total).w()?;
    raw.dtod_copy(
        &payload.slice(..local_size),
        &mut received.slice_mut(offsets[comm.rank()]..offsets[comm.rank()] + local_size),
    )
    .w()?;

    ubridge::eccllib::group_start().map_err(candle_core::Error::debug)?;
    for peer in 0..world {
        if peer == comm.rank() {
            continue;
        }
        let send = payload.slice(..local_size);
        let mut recv = received.slice_mut(offsets[peer]..offsets[peer] + all_sizes[peer]);
        comm.send(&send, peer as i32)
            .map_err(candle_core::Error::debug)?;
        comm.recv(&mut recv, peer as i32)
            .map_err(candle_core::Error::debug)?;
    }
    ubridge::eccllib::group_end().map_err(candle_core::Error::debug)?;
    Ok(received)
}

fn exchange_variable_chunks(
    raw: &Arc<ubridge::gcu_device::GcuDevice>,
    comm: &Comm,
    send_pack: &GcuSlice<u8>,
    packet_capacity: usize,
    local_sizes: &[usize],
    all_sizes: &[usize],
) -> Result<(GcuSlice<u8>, Vec<usize>, Vec<usize>)> {
    let world = comm.world_size();
    if local_sizes.len() != world || all_sizes.len() != world * world {
        candle_core::bail!("invalid ZipCCL AllReduce size table")
    }
    let mut recv_sizes = vec![0usize; world];
    for src in 0..world {
        recv_sizes[src] = if src == comm.rank() {
            local_sizes[comm.rank()]
        } else {
            all_sizes[src * world + comm.rank()]
        };
    }
    let mut offsets = vec![0usize; world];
    for src in 1..world {
        offsets[src] = offsets[src - 1] + recv_sizes[src - 1];
    }
    let total = offsets[world - 1] + recv_sizes[world - 1];
    let mut received = raw.alloc::<u8>(total).w()?;
    let local_packet = send_pack.slice(
        comm.rank() * packet_capacity..comm.rank() * packet_capacity + local_sizes[comm.rank()],
    );
    raw.dtod_copy(
        &local_packet,
        &mut received
            .slice_mut(offsets[comm.rank()]..offsets[comm.rank()] + recv_sizes[comm.rank()]),
    )
    .w()?;

    ubridge::eccllib::group_start().map_err(candle_core::Error::debug)?;
    for peer in 0..world {
        if peer == comm.rank() {
            continue;
        }
        let send =
            send_pack.slice(peer * packet_capacity..peer * packet_capacity + local_sizes[peer]);
        let mut recv = received.slice_mut(offsets[peer]..offsets[peer] + recv_sizes[peer]);
        comm.send(&send, peer as i32)
            .map_err(candle_core::Error::debug)?;
        comm.recv(&mut recv, peer as i32)
            .map_err(candle_core::Error::debug)?;
    }
    ubridge::eccllib::group_end().map_err(candle_core::Error::debug)?;
    Ok((received, offsets, recv_sizes))
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
    let compressed = compress(raw, input, n, dtype)?;
    // ECCL requires a host-side count.  Delay the only metadata fence until
    // after compression has returned the device buffers to this collective.
    let exact = compressed.exact_sizes(raw)?[0];
    let sizes = exchange_sizes(raw, comm, &[exact])?;
    let received = exchange_variable_payload(raw, comm, &compressed.payload, exact, &sizes)?;
    let mut output = raw.alloc::<O>(n * comm.world_size()).w()?;
    let mut offset = 0;
    for rank in 0..comm.world_size() {
        let size = sizes[rank];
        let view = received.slice(offset..offset + size);
        let mut out = output.slice_mut(rank * n..(rank + 1) * n);
        decompress::<O, _, _>(raw, &view, &mut out, n, dtype)?;
        offset += size;
    }
    Ok(output)
}

fn all_reduce_impl<O: DeviceCopy + EcclType>(
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
    // Compress all world chunks in a single launch: the chunks are contiguous
    // in `input`, and each chunk becomes one fixed-capacity packet.
    let compressed = compress_chunks(raw, input, chunk, world, dtype)?;
    // Materialize all packet lengths with one stream-ordered readback.  The
    // compact kernel already produced every length on-device; this is deferred
    // to the point where the host must post variable-count ECCL receives.
    let local_sizes = compressed.exact_sizes(raw)?;
    let all_sizes = exchange_sizes(raw, comm, &local_sizes)?;
    let packet_capacity = compressed.capacity;
    let (received, recv_offsets, recv_sizes) = exchange_variable_chunks(
        raw,
        comm,
        &compressed.payload,
        packet_capacity,
        &local_sizes,
        &all_sizes,
    )?;

    let local = offset_ptr::<u16>(input, comm.rank() * chunk);
    let mut accumulator = raw.alloc_zeros::<f32>(chunk).w()?;
    to_f32(raw, local, &mut accumulator, chunk, dtype)?;
    for src in 0..world {
        if src == comm.rank() {
            continue;
        }
        let view = received.slice(recv_offsets[src]..recv_offsets[src] + recv_sizes[src]);
        // Fused decompress + FP32 accumulation in one kernel launch.
        decompress_add_f32(raw, &view, &mut accumulator, chunk, dtype)?;
    }
    let mut reduced = raw.alloc::<O>(chunk).w()?;
    from_f32(raw, &accumulator, &mut reduced, chunk, dtype)?;
    // The reduce-scatter phase is where compression saves the expensive
    // cross-rank traffic.  Once each rank owns its reduced chunk, do not
    // launch a second compress -> all-gather -> decompress cycle: that added
    // nine codec launches per collective and dominated prefill latency.
    // ECCL can all-gather the native reduced BF16/F16 chunks directly.
    let mut output = raw.alloc::<O>(n).w()?;
    comm.all_gather(&reduced, &mut output)
        .map_err(candle_core::Error::debug)?;
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

mod graph_gcu {
    // Original implementation:
    // https://github.com/guoqingbao/vllm.rs/blob/main/src/utils/graph.rs
    #[cfg(feature = "flashinfer")]
    use crate::FlashInferKvParams;
    use std::collections::BTreeMap;
    use std::mem::MaybeUninit;
    use std::ptr;
    use std::sync::Arc;

    use attention_rs::InputMetadata;
    use candle_core::gcu_backend::ubridge::gcu_device::driv;
    use candle_core::gcu_backend::ubridge::prelude::tops::error::ToResult;
    use candle_core::gcu_backend::GcuDevice;
    use candle_core::{DType, Device, Result, Tensor};
    use parking_lot::RwLock;
    use tqdm::tqdm;

    /// Maximum context length for which the regular decode graph path is
    /// eligible.  The captured FA metadata uses the model's actual limit,
    /// while this public bound is used by the pipeline before replay.
    pub const GRAPH_FA_MAX_CONTEXT_LEN: usize = 262_144;

    #[allow(dead_code)]
    pub struct CudaGraph {
        cu_graph: driv::topsGraph_t,
        cu_graph_exec: driv::topsGraphExec_t,
        stream: driv::topsStream_t,
    }

    impl CudaGraph {
        pub fn begin_capture(
            stream: driv::topsStream_t,
            mode: driv::topsStreamCaptureMode,
        ) -> Result<()> {
            unsafe {
                driv::topsStreamBeginCapture(stream, mode)
                    .to_result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("topsStreamBeginCapture failed: {e:?}"))
                    })
            }
        }

        pub fn end_capture(
            stream: driv::topsStream_t,
            _flags: driv::topsGraphInstantiateFlags,
        ) -> Result<CudaGraph> {
            let mut graph = MaybeUninit::uninit();
            let cu_graph = unsafe {
                driv::topsStreamEndCapture(stream, graph.as_mut_ptr())
                    .to_result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("topsStreamEndCapture failed: {e:?}"))
                    })?;
                graph.assume_init()
            };

            let mut graph_exec = MaybeUninit::uninit();
            let cu_graph_exec = unsafe {
                driv::topsGraphInstantiateWithFlags(
                    graph_exec.as_mut_ptr(),
                    cu_graph,
                    0 as u32 as u64,
                )
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsGraphInstantiateWithFlags failed: {e:?}"))
                })?;
                graph_exec.assume_init()
            };
            Ok(CudaGraph {
                cu_graph,
                cu_graph_exec,
                stream,
            })
        }

        pub fn capture_status(stream: driv::topsStream_t) -> Result<driv::topsStreamCaptureStatus> {
            let mut status = driv::topsStreamCaptureStatus::topsStreamCaptureStatusNone;
            unsafe {
                driv::topsStreamIsCapturing(stream, &mut status)
                    .to_result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("topsStreamIsCapturing failed: {e:?}"))
                    })?;
            }
            Ok(status)
        }

        pub fn launch(&self) -> Result<()> {
            unsafe {
                driv::topsGraphLaunch(self.cu_graph_exec, self.stream)
                    .to_result()
                    .map_err(|e| candle_core::Error::Msg(format!("topsGraphLaunch failed: {e:?}")))
            }
        }
    }

    pub trait CudaGraphModule {
        fn start_capture(&mut self, bs: usize) -> Result<()>;
        fn end_capture(&mut self) -> Result<()>;
        fn replay(&self, bs: usize) -> Result<()>;
        fn forward(
            &self,
            input_ids: &Tensor,
            positions: &Tensor,
            kv_caches: Option<&Vec<(Tensor, Tensor)>>,
            input_metadata: &InputMetadata,
        ) -> Result<Tensor>;
        fn report_graph_pool_usage(&self) -> Result<()>;
    }

    pub struct CudaGraphHandle {
        graph: Arc<CudaGraph>,
    }

    impl CudaGraphHandle {
        pub fn new(graph: Arc<CudaGraph>) -> Self {
            Self { graph }
        }

        pub fn replay(&self) -> Result<()> {
            self.graph.launch().map_err(|e| {
                candle_core::Error::Msg(format!("CUDA Graph launch failed: {:?}", e))
            })?;
            Ok(())
        }
    }

    pub struct CudaGraphWrapper<M>
    where
        M: for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
        ) -> Result<Tensor>,
    {
        module: M,
        captured_graphs: BTreeMap<usize, CudaGraphHandle>,
        capturing: bool,
        current_bs: Option<usize>,
        device: Arc<GcuDevice>,
        pub pool_handle: RwLock<Option<i64>>,
        captured_bs: Vec<usize>,
    }

    impl<M> CudaGraphWrapper<M>
    where
        M: for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
        ) -> Result<Tensor>,
    {
        pub fn new(module: M, device: Arc<GcuDevice>) -> Self {
            Self {
                module,
                captured_graphs: BTreeMap::new(),
                capturing: false,
                current_bs: None,
                device,
                pool_handle: RwLock::new(None),
                captured_bs: Vec::new(),
            }
        }

        fn sync_stream(&self) -> Result<()> {
            let stream = self
                .device
                .stream_inner()
                .expect("unable to obtain stream!");
            unsafe {
                driv::topsStreamSynchronize(stream)
                    .to_result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("topsStreamSynchronize failed: {e:?}"))
                    })
            }
        }

        fn create_capture_pool(&self) -> Result<driv::topsMemPool_t> {
            let mut pool: driv::topsMemPool_t = ptr::null_mut();
            unsafe {
                driv::topsDeviceGetDefaultMemPool(&mut pool, self.device.gcu_device().id as i32)
                    .to_result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "topsDeviceGetDefaultMemPool failed: {e:?}"
                        ))
                    })?;

                let handle = pool as *mut std::ffi::c_void as usize as i64;
                *self.pool_handle.write() = Some(handle);

                let threshold: u64 = u64::MAX;
                driv::topsMemPoolSetAttribute(
                    pool,
                    driv::topsMemPoolAttr::topsMemPoolAttrReleaseThreshold,
                    &threshold as *const _ as _,
                )
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsMemPoolSetAttribute failed: {e:?}"))
                })?;
            }
            Ok(pool)
        }

        fn set_capture_mem_pool(&self) -> Result<()> {
            if self.pool_handle.read().is_some() {
                return Ok(());
            }
            let stream = self
                .device
                .stream_inner()
                .expect("unable to obtain stream!");
            unsafe {
                let status = CudaGraph::capture_status(stream)?;
                if status != driv::topsStreamCaptureStatus::topsStreamCaptureStatusActive {
                    let pool = self.create_capture_pool()?;
                    driv::topsDeviceSetMemPool(self.device.gcu_device().id as i32, pool)
                        .to_result()
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("topsDeviceSetMemPool failed: {e:?}"))
                        })?;
                }
            }

            Ok(())
        }

        /// Reads a usize attribute from the given GCU memory pool.
        fn get_mem_pool_attribute(
            pool: driv::topsMemPool_t,
            attr: driv::topsMemPoolAttr,
        ) -> Result<usize> {
            let mut value: usize = 0;
            unsafe {
                driv::topsMemPoolGetAttribute(
                    pool,
                    attr,
                    &mut value as *mut _ as *mut std::ffi::c_void,
                )
                .to_result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("topsMemPoolGetAttribute failed: {e:?}"))
                })?;
            }
            Ok(value)
        }

        /// Returns peak memory used (in bytes) from a given CUDA memory pool.
        pub fn get_peak_memory_usage(pool: driv::topsMemPool_t) -> Result<usize> {
            Self::get_mem_pool_attribute(
                pool,
                driv::topsMemPoolAttr::topsMemPoolAttrReservedMemHigh,
            )
        }

        /// Returns current memory usage (in bytes) from a given CUDA memory pool.
        pub fn get_current_memory_usage(pool: driv::topsMemPool_t) -> Result<usize> {
            Self::get_mem_pool_attribute(pool, driv::topsMemPoolAttr::topsMemPoolAttrUsedMemCurrent)
        }

        /// Retrieves the default GCU memory pool for a device.
        pub fn get_current_mem_pool(&self) -> Result<driv::topsMemPool_t> {
            if self.pool_handle.read().is_some() {
                let pool_handle = self.pool_handle.read().unwrap();
                let pool: driv::topsMemPool_t = pool_handle as usize as driv::topsMemPool_t;
                Ok(pool)
            } else {
                candle_core::bail!("Memory pool for graph is not init!")
            }
        }
    }

    impl<M> CudaGraphModule for CudaGraphWrapper<M>
    where
        M: for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
        ) -> Result<Tensor>,
    {
        fn start_capture(&mut self, bs: usize) -> Result<()> {
            self.capturing = true;
            self.current_bs = Some(bs);
            self.sync_stream()?;
            self.set_capture_mem_pool()?;
            let stream = self
                .device
                .stream_inner()
                .expect("unable to obtain stream!");
            CudaGraph::begin_capture(
                stream,
                driv::topsStreamCaptureMode::topsStreamCaptureModeRelaxed,
            )?;
            Ok(())
        }

        fn end_capture(&mut self) -> Result<()> {
            self.capturing = false;
            let bs = self.current_bs.take().unwrap();
            let stream = self
                .device
                .stream_inner()
                .expect("unable to obtain stream!");
            let graph = CudaGraph::end_capture(
                stream,
                driv::topsGraphInstantiateFlags::topsGraphInstantiateFlagAutoFreeOnLaunch,
            )?;
            self.captured_graphs
                .insert(bs, CudaGraphHandle::new(Arc::new(graph)));

            self.captured_bs.push(bs);
            self.captured_bs.sort_unstable(); // keep it sorted for binary search
            self.sync_stream()?;
            Ok(())
        }

        fn replay(&self, bs: usize) -> Result<()> {
            if let Some(&next_bs) = self.captured_bs.iter().find(|&&x| x >= bs) {
                if let Some(graph) = self.captured_graphs.get(&next_bs) {
                    self.sync_stream()?;
                    graph.replay()?;
                    self.sync_stream()
                } else {
                    candle_core::bail!("No suitable graph is found for batch size {}!", next_bs)
                }
            } else {
                candle_core::bail!("Batch size {} is not captured in graph!", bs)
            }
        }

        fn forward(
            &self,
            input_ids: &Tensor,
            positions: &Tensor,
            kv_caches: Option<&Vec<(Tensor, Tensor)>>,
            input_metadata: &InputMetadata,
        ) -> Result<Tensor> {
            (self.module)(input_ids, positions, kv_caches, input_metadata)
        }

        fn report_graph_pool_usage(&self) -> Result<()> {
            let pool = self.get_current_mem_pool()?;
            let peak = Self::get_peak_memory_usage(pool)?;
            let current = Self::get_current_memory_usage(pool)?;
            tracing::info!(
                "Default pool usage: {:.2} MB (current), {:.2} MB (peak)",
                current as f64 / 1e6,
                peak as f64 / 1e6
            );
            Ok(())
        }
    }

    pub struct GraphCaptureVars {
        pub input_ids: Tensor,
        pub positions: Tensor,
        pub mamba_slot_mapping: Tensor,
        pub slot_mapping: Tensor,
        pub context_lens: Tensor,
        pub block_tables: Tensor,
        pub cu_seqlens_q: Tensor,
        #[cfg(feature = "flashinfer")]
        pub flashinfer_indptr: Tensor,
        #[cfg(feature = "flashinfer")]
        pub flashinfer_indices: Tensor,
        #[cfg(feature = "flashinfer")]
        pub flashinfer_last_len: Tensor,
        pub outputs: BTreeMap<usize, Tensor>,
    }

    pub struct GraphCapturer<M: CudaGraphModule> {
        pub model: M,
        pub graph_bs: Vec<usize>,
        pub graph_vars: Option<GraphCaptureVars>,
        pub max_num_seqs: usize,
        pub max_model_len: usize,
        pub block_size: usize,
        pub hidden_size: usize,
        pub device: Option<Device>,
        #[cfg(feature = "flashinfer")]
        pub flashinfer_kv_params: Option<FlashInferKvParams>,
    }

    pub fn planned_graph_capture_batches(max_num_seqs: usize) -> Vec<usize> {
        let small_max = max_num_seqs.clamp(1, 15);
        let mut graph_bs = (1..=small_max).collect::<Vec<_>>();
        if max_num_seqs >= 16 {
            graph_bs.extend((16..=max_num_seqs.min(32)).step_by(16));
        }
        graph_bs
    }

    impl<M: CudaGraphModule> GraphCapturer<M> {
        pub fn new(
            model: M,
            max_num_seqs: usize,
            max_model_len: usize,
            block_size: usize,
            hidden_size: usize,
            #[cfg(feature = "flashinfer")] flashinfer_kv_params: &Option<FlashInferKvParams>,
        ) -> Self {
            let graph_bs = planned_graph_capture_batches(max_num_seqs);
            println!("The following batches for capture: {:?}", graph_bs);

            Self {
                model,
                graph_bs,
                graph_vars: None,
                max_num_seqs,
                max_model_len,
                block_size,
                hidden_size,
                device: None,
                #[cfg(feature = "flashinfer")]
                flashinfer_kv_params: *flashinfer_kv_params,
            }
        }

        pub fn capture(
            &mut self,
            device: &Device,
            kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        ) -> Result<()> {
            self.device = Some(device.clone());
            let max_bs = self.graph_bs[self.graph_bs.len() - 1];
            let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;
            let graph_context_len = self.max_model_len;

            let input_ids = Tensor::zeros((max_bs,), DType::U32, device)?;
            let positions = Tensor::zeros((max_bs,), DType::I64, device)?;
            let mamba_slot_mapping = Tensor::from_vec(
                (0..max_bs).map(|i| i as i64).collect::<Vec<_>>(),
                (max_bs,),
                device,
            )?;
            let slot_mapping = Tensor::zeros((max_bs,), DType::I64, device)?;
            let context_lens = Tensor::ones((max_bs,), DType::U32, device)?;
            let block_tables = Tensor::zeros((max_bs, max_num_blocks), DType::U32, device)?;
            let cu_seqlens_q = Tensor::from_vec(
                (0..=max_bs as u32).collect::<Vec<_>>(),
                (max_bs + 1,),
                device,
            )?;
            #[cfg(feature = "flashinfer")]
            let (flashinfer_indptr, flashinfer_indices, flashinfer_last_len, last_len_host) = {
                let mut indptr = Vec::with_capacity(max_bs + 1);
                indptr.push(0u32);
                let mut indices = Vec::with_capacity(max_bs * max_num_blocks);
                for _ in 0..max_bs {
                    for i in 0..max_num_blocks {
                        indices.push(i as u32);
                    }
                    indptr.push(indices.len() as u32);
                }
                let last = if self.max_model_len == 0 {
                    0u32
                } else {
                    ((self.max_model_len - 1) % self.block_size + 1) as u32
                };
                let last_len = vec![last; max_bs];
                (
                    Tensor::from_vec(indptr, (max_bs + 1,), device)?,
                    Tensor::from_vec(indices, (max_bs * max_num_blocks,), device)?,
                    Tensor::from_vec(last_len.clone(), (max_bs,), device)?,
                    last_len,
                )
            };
            // Enable the kernel parameter cache so that shape/stride metadata
            // uploaded via htod_copy gets a fixed device address across warmup
            // and capture phases.
            let _param_cache_guard = candle_core::cuda_param_cache_scope(true);

            let mut outputs = BTreeMap::<usize, Tensor>::new();

            // Helper closure to build InputMetadata for a given batch size.
            let build_input_metadata = |bs: usize| -> Result<InputMetadata> {
                #[cfg(feature = "flashinfer")]
                let flashinfer_metadata = {
                    let mut indptr_host = Vec::with_capacity(bs + 1);
                    indptr_host.push(0u32);
                    for i in 0..bs {
                        indptr_host.push(((i + 1) * max_num_blocks) as u32);
                    }

                    let (decode_plan_info, kv_len_arr_host) =
                        if let Some(params) = self.flashinfer_kv_params {
                            let mut kv_len_arr_host_bs = Vec::with_capacity(bs);
                            for i in 0..bs {
                                let num_pages = indptr_host[i + 1] - indptr_host[i];
                                if num_pages == 0 {
                                    kv_len_arr_host_bs.push(0);
                                } else {
                                    let full = (num_pages - 1) * params.page_size as u32;
                                    kv_len_arr_host_bs.push(full + last_len_host[i]);
                                }
                            }
                            let kv_len_arr_host = kv_len_arr_host_bs.clone();
                            (
                                Some(attention_rs::flashinfer::decode_plan(
                                    device,
                                    params.kv_dtype,
                                    params.out_dtype,
                                    &indptr_host,
                                    Some(&last_len_host[..bs]),
                                    Some(kv_len_arr_host_bs.as_slice()),
                                    bs,
                                    params.num_qo_heads,
                                    params.num_kv_heads,
                                    params.head_dim,
                                    params.page_size,
                                    true,
                                )?),
                                Some(kv_len_arr_host),
                            )
                        } else {
                            (None, None)
                        };

                    Some(attention_rs::FlashInferMetadata {
                        indptr: flashinfer_indptr.narrow(0, 0, bs + 1)?,
                        indptr_host,
                        indices: flashinfer_indices.narrow(0, 0, bs * max_num_blocks)?,
                        last_len: flashinfer_last_len.narrow(0, 0, bs)?,
                        last_len_host: Some(last_len_host[..bs].to_vec()),
                        kv_len_arr_host,
                        cu_seqlens_q_host: None,
                        total_num_rows: None,
                        batch_indices: None,
                        positions: None,
                        use_cuda_graph: true,
                        decode_plan_info,
                    })
                };
                #[cfg(not(feature = "flashinfer"))]
                let flashinfer_metadata = None;

                Ok(InputMetadata {
                    is_prefill: false,
                    is_mla: false,
                    sequence_ids: None,
                    mamba_slot_mapping: Some(mamba_slot_mapping.narrow(0, 0, bs)?),
                    slot_mapping: slot_mapping.narrow(0, 0, bs)?,
                    block_tables: Some(block_tables.narrow(0, 0, bs)?),
                    context_lens: Some(context_lens.narrow(0, 0, bs)?),
                    cu_seqlens_q: Some(cu_seqlens_q.narrow(0, 0, bs + 1)?),
                    cu_seqlens_k: None,
                    max_seqlen_q: 1,
                    max_seqlen_k: graph_context_len,
                    max_context_len: graph_context_len,
                    seqlens: None,
                    flashinfer_metadata,
                    is_mtp_verify: false,
                })
            };

            // Phase 1 – Warmup: run forward passes without capture to populate
            // the kernel parameter cache and exercise allocators.
            for i in 0..self.graph_bs.len() {
                let bs = self.graph_bs[self.graph_bs.len() - i - 1];
                let input_ids_bs = input_ids.narrow(0, 0, bs)?;
                let positions_bs = positions.narrow(0, 0, bs)?;
                let input_metadata = build_input_metadata(bs)?;
                let _ =
                    self.model
                        .forward(&input_ids_bs, &positions_bs, kv_caches, &input_metadata)?;
            }

            // Phase 2 – Capture: graph capture with stable kernel parameter
            // addresses.
            for i in tqdm(0..self.graph_bs.len()).desc(Some("Graph capturing")) {
                let bs = self.graph_bs[self.graph_bs.len() - i - 1];
                let input_ids_bs = input_ids.narrow(0, 0, bs)?;
                let positions_bs = positions.narrow(0, 0, bs)?;
                let input_metadata = build_input_metadata(bs)?;

                self.model.start_capture(bs)?;
                let out =
                    self.model
                        .forward(&input_ids_bs, &positions_bs, kv_caches, &input_metadata)?;
                self.model.end_capture()?;
                outputs.insert(bs, out);
            }

            let _ = self.model.report_graph_pool_usage();

            tracing::warn!("Captured batches {:?}", outputs.keys());

            self.graph_vars = Some(GraphCaptureVars {
                input_ids,
                positions,
                mamba_slot_mapping,
                slot_mapping,
                context_lens,
                block_tables,
                cu_seqlens_q,
                #[cfg(feature = "flashinfer")]
                flashinfer_indptr,
                #[cfg(feature = "flashinfer")]
                flashinfer_indices,
                #[cfg(feature = "flashinfer")]
                flashinfer_last_len,
                outputs,
            });

            Ok(())
        }

        pub fn is_captured(&self, batch: usize) -> bool {
            self.graph_vars.is_some()
                && self
                    .graph_vars
                    .as_ref()
                    .unwrap()
                    .outputs
                    .keys()
                    .find(|&&x| x >= batch)
                    .is_some()
        }

        pub fn is_exact_captured(&self, batch: usize) -> bool {
            self.graph_vars.is_some()
                && self
                    .graph_vars
                    .as_ref()
                    .unwrap()
                    .outputs
                    .contains_key(&batch)
        }

        pub fn replay(
            &self,
            input_ids: &Tensor,
            positions: &Tensor,
            input_metadata: &InputMetadata,
        ) -> Result<Tensor> {
            if input_metadata.is_prefill {
                candle_core::bail!("Graph replay is not used for prefill!")
            }
            let max_num_blocks = (self.max_model_len + self.block_size - 1) / self.block_size;
            let input_batch = input_ids.dim(0)?;
            let require_exact_batch = input_metadata.mamba_slot_mapping.is_some()
                || input_metadata.flashinfer_metadata.is_some();
            if let Some(graph_vars) = &self.graph_vars {
                let selected_batch = if require_exact_batch {
                    graph_vars
                        .outputs
                        .keys()
                        .find(|&&x| x == input_batch)
                        .copied()
                } else {
                    graph_vars
                        .outputs
                        .keys()
                        .find(|&&x| x >= input_batch)
                        .copied()
                };
                if let Some(batch) = selected_batch {
                    graph_vars.input_ids.zero_()?;
                    graph_vars.input_ids.copy_(&input_ids, 0)?;
                    graph_vars.positions.zero_()?;
                    graph_vars.positions.copy_(&positions, 0)?;

                    if let Some(ms_mapping) = input_metadata.mamba_slot_mapping.as_ref() {
                        graph_vars.mamba_slot_mapping.zero_()?;
                        graph_vars.mamba_slot_mapping.copy_(&ms_mapping, 0)?;
                    } else {
                        graph_vars.mamba_slot_mapping.zero_()?;
                    }

                    let s_mapping = input_metadata.slot_mapping.as_ref();
                    graph_vars.slot_mapping.zero_()?;
                    graph_vars.slot_mapping.copy_(&s_mapping, 0)?;

                    let c_lens = input_metadata.context_lens.as_ref().unwrap();
                    graph_vars.context_lens.zero_()?;
                    graph_vars.context_lens.copy_(&c_lens, 0)?;

                    let b_tables = input_metadata.block_tables.as_ref().unwrap();
                    let padded_table = b_tables
                        .pad_with_zeros(1, 0, max_num_blocks - b_tables.dim(1)?)?
                        .contiguous()?;

                    graph_vars.block_tables.zero_()?;
                    graph_vars.block_tables.copy_(&padded_table, 0)?;

                    #[cfg(feature = "flashinfer")]
                    if let Some(fm) = &input_metadata.flashinfer_metadata {
                        let mut indptr_host = fm.indptr_host.clone();
                        if input_batch == batch {
                            graph_vars.flashinfer_indptr.zero_()?;
                            graph_vars.flashinfer_indptr.copy_(&fm.indptr, 0)?;
                        } else {
                            let last = *indptr_host.last().unwrap_or(&0);
                            for _ in (input_batch + 1)..=batch {
                                indptr_host.push(last);
                            }
                            let indptr_padded = Tensor::from_vec(
                                indptr_host.clone(),
                                (batch + 1,),
                                graph_vars.input_ids.device(),
                            )?;
                            graph_vars.flashinfer_indptr.copy_(&indptr_padded, 0)?;
                        }

                        graph_vars.flashinfer_last_len.zero_()?;
                        graph_vars.flashinfer_last_len.copy_(&fm.last_len, 0)?;

                        graph_vars.flashinfer_indices.zero_()?;
                        graph_vars.flashinfer_indices.copy_(&fm.indices, 0)?;

                        if let Some(params) = self.flashinfer_kv_params {
                            let dev = self.device.as_ref().ok_or_else(|| {
                                candle_core::Error::msg("graph device is missing")
                            })?;
                            let _ = attention_rs::flashinfer::decode_plan(
                                dev,
                                params.kv_dtype,
                                params.out_dtype,
                                &indptr_host,
                                fm.last_len_host.as_deref(),
                                fm.kv_len_arr_host.as_deref(),
                                batch,
                                params.num_qo_heads,
                                params.num_kv_heads,
                                params.head_dim,
                                params.page_size,
                                fm.use_cuda_graph,
                            )?;
                        }
                    }

                    let result = self.model.replay(batch);
                    if result.is_err() {
                        eprintln!("Error when replaying graph {:?}", result);
                    }

                    graph_vars.outputs[&batch]
                        .narrow(0, 0, input_batch)?
                        .contiguous()
                } else {
                    candle_core::bail!("Input batch {} is not captured!", input_batch)
                }
            } else {
                candle_core::bail!("Graph is not captured!")
            }
        }
    }

    unsafe impl Send for CudaGraph {}
    unsafe impl Sync for CudaGraph {}

    pub type ModelFn = dyn for<'a> Fn(
            &'a Tensor,
            &'a Tensor,
            Option<&'a Vec<(Tensor, Tensor)>>,
            &'a InputMetadata,
        ) -> Result<Tensor>
        + Send
        + Sync;

    pub type CudaGraphFn = Box<
        dyn for<'a> Fn(
                &'a Tensor,
                &'a Tensor,
                Option<&'a Vec<(Tensor, Tensor)>>,
                &'a InputMetadata,
            ) -> Result<Tensor>
            + Send
            + Sync,
    >;
}

pub use graph_gcu::*;

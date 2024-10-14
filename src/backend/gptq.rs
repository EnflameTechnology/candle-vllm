use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use half::{bf16, f16};
use kernels::ffi::{gptq_marlin_repack, marlin_4bit_f16};

struct GPTQMatMul {
    qzeros: Option<Tensor>,
    g_idx: Option<Tensor>,
    perm: Option<Tensor>,
    workspace: Option<Tensor>,
    bits: i32,
}

impl GPTQMatMul {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let internal_type = match x.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            dtype => candle::bail!("dtype {dtype:?} is not supported"),
        };
        let dev = qweight.device();

        let x_shape = x_l.dims();
        let weight_shape = qweight_l.dims();
        // let zero_shape = self.qzeros.shape().dims();
        let scale_shape = scale_l.dims();

        let pack_factor: usize = 32 / self.bits as usize;
        let size_k = weight_shape[0] * pack_factor * 2; //marlin format
        let size_n = weight_shape[1] / 2; //marlin format

        let mut out_shape: Vec<usize> = x_shape.to_vec();
        out_shape[x_shape.len() - 1] = size_n;
        let oshape: Shape = out_shape.into();

        // Get cuda slices for all tensors
        let input = x.as_cuda_slice::<T>()?;
        let qw = qweight.as_cuda_slice::<u32>()?;
        let qs = scale.as_cuda_slice::<f16>()?;

        // Get cuda views for all tensors
        let input = input.slice(x_l.start_offset()..);
        let qw = qw.slice(qweight_l.start_offset()..);
        let qs = qs.slice(scale_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *const core::ffi::c_void;
        let in_ptr = *input.device_ptr() as *const core::ffi::c_void;
        let qw_ptr = *qw.device_ptr() as *const core::ffi::c_void;
        let qs_ptr = *qs.device_ptr() as *const core::ffi::c_void;
        let workspace_ptr = if self.workspace.is_some() {
            let (workspace, workspace_l) = self.workspace.as_ref().unwrap().storage_and_layout();
            let workspace = match &*workspace {
                Storage::Cuda(p) => p,
                _ => candle::bail!("workspace must be a cuda tensor"),
            };
            let workspace_ = workspace.as_cuda_slice::<u32>()?;
            let workspace_ = workspace_.slice(workspace_l.start_offset()..);
            *workspace_.device_ptr() as *const core::ffi::c_void
        } else {
            out_ptr
        };
        unsafe {
            let groupsize: i32 = if scale_shape[0] == 1 {
                -1i32
            } else {
                (size_k / scale_shape[0]) as i32
            };
            marlin_4bit_f16(
                in_ptr,
                qw_ptr as *const i32,
                qs_ptr,
                out_ptr,
                (x_shape[0] * x_shape[1]) as i32, //m
                size_k as i32,                    //k
                size_n as i32,                    //n
                workspace_ptr,
                groupsize as i32,
            );
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp3 for GPTQMatMul {
    fn name(&self) -> &'static str {
        "GPTQMatMul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for GPTQMatMul")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.cuda_fwd_t::<f16>(x, x_l, qweight, qweight_l, scale, scale_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(x, x_l, qweight, qweight_l, scale, scale_l),
            DType::F32 => self.cuda_fwd_t::<f32>(x, x_l, qweight, qweight_l, scale, scale_l),
            dt => candle::bail!("GPTQMatMul is only supported for f16, bf16 and f32 ({dt:?})"),
        }
    }
}

pub fn gptq_matmul(
    x: &Tensor,
    qweight: &Tensor,
    scale: &Tensor,
    qzeros: &Option<Tensor>,
    g_idx: &Option<Tensor>,
    perm: &Option<Tensor>,
    workspace: &Option<Tensor>,
    bits: i32,
) -> Result<Tensor> {
    let op = GPTQMatMul {
        qzeros: qzeros.to_owned(),
        g_idx: g_idx.to_owned(),
        perm: perm.to_owned(),
        workspace: workspace.to_owned(),
        bits,
    };
    x.apply_op3(qweight, scale, op)
}

struct GPTQRepack {
    k: i32,
    bits: i32,
}

impl GPTQRepack {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        perm: &CudaStorage,
        perm_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = qweight.device();
        let q_shape = qweight_l.dims();
        let mut out_shape: Vec<usize> = q_shape.to_vec();
        // out_shape[0] = (q_shape[0] / 2) as usize;
        // out_shape[1] = (q_shape[1] * 2) as usize;

        let oshape: Shape = out_shape.into();

        // Get cuda slices for all tensors
        let q = qweight.as_cuda_slice::<u32>()?;
        let perm_ = perm.as_cuda_slice::<u32>()?;

        // Get cuda views for all tensors
        let q = q.slice(qweight_l.start_offset()..);
        let perm_ = perm_.slice(perm_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<u32>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *const core::ffi::c_void;
        let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
        let q_perm = *perm_.device_ptr() as *const core::ffi::c_void;

        unsafe { gptq_marlin_repack(q_ptr, q_perm, out_ptr, self.k, q_shape[1] as i32, self.bits) }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp2 for GPTQRepack {
    fn name(&self) -> &'static str {
        "GPTQRepack"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for GPTQRepack")
    }

    fn cuda_fwd(
        &self,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        perm: &CudaStorage,
        perm_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match qweight.dtype() {
            DType::U32 => self.cuda_fwd_t::<u32>(qweight, qweight_l, perm, perm_l),
            dt => candle::bail!("GPTQRepack is only supported for i32/u32 weight ({dt:?})"),
        }
    }
}

pub fn gptq_weight_repack(qweight: &Tensor, perm: &Tensor, size_k: usize) -> Result<Tensor> {
    let op = GPTQRepack {
        bits: 4,
        k: size_k as i32,
    };
    qweight.apply_op2(perm, op)
}

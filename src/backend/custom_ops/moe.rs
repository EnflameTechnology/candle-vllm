#[cfg(feature = "cuda")]
use attention_rs::sort::ArgSortOp; //use custom argsort which fixed the bugs on A100
use candle::shape::Dim;
use candle::{CpuStorage, CustomOp1, Error, Layout, Shape, WithDType};
use candle::{Result, Tensor, D};
use candle_core as candle;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
struct NonZero {
    v: u32,
}

impl NonZero {
    // Sequential version
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout, v: T) -> Vec<u32> {
        let dims = layout.dims();
        let n = dims.len();
        let chunk_size = if vs.len() > 4096 { 512 } else { 128 };
        use rayon::iter::IndexedParallelIterator;
        use rayon::prelude::ParallelSlice;
        vs.par_chunks(chunk_size)
            .enumerate()
            .fold(
                || Vec::with_capacity(chunk_size * n), // Pre-allocate per-thread
                |mut acc, (chunk_idx, chunk)| {
                    let start_index = chunk_idx * chunk_size;

                    // Process each element in chunk
                    chunk.iter().enumerate().for_each(|(i, value)| {
                        if *value == v {
                            let mut idx = start_index + i;
                            let mut indices = Vec::with_capacity(n);

                            // Calculate indices in reverse dimension order
                            for &dim in dims.iter().rev() {
                                let d = idx % dim;
                                indices.push(u32::try_from(d).unwrap());
                                idx /= dim;
                            }

                            // Store in original dimension order
                            indices.reverse();
                            acc.extend(indices);
                        }
                    });

                    acc
                },
            )
            .reduce(
                || Vec::new(),
                |mut a, mut b| {
                    a.append(&mut b);
                    a
                },
            )
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "nonzero" });
        }
        let result = match storage {
            CpuStorage::U8(vs) => self.nonzero(vs, layout, self.v as u8),
            CpuStorage::U32(vs) => self.nonzero(vs, layout, self.v),
            CpuStorage::I64(vs) => self.nonzero(vs, layout, self.v as i64),
            CpuStorage::BF16(vs) => self.nonzero(vs, layout, half::bf16::from_f32(self.v as f32)),
            CpuStorage::F16(vs) => self.nonzero(vs, layout, half::f16::from_f32(self.v as f32)),
            CpuStorage::F32(vs) => self.nonzero(vs, layout, self.v as f32),
            CpuStorage::F64(vs) => self.nonzero(vs, layout, self.v as f64),
            CpuStorage::I8(vs) => self.nonzero(vs, layout, self.v as i8),
            CpuStorage::I32(vs) => self.nonzero(vs, layout, self.v as i32),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        let result = CpuStorage::U32(result);
        let shape = Shape::from_dims(&[result_len, index_len]);
        Ok((result, shape))
    }
}

pub trait NonZeroOp {
    fn nonzero(&self, v: u32) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self, v: u32) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        let original_device = self.device();
        if original_device.is_cpu() {
            self.apply_op1_no_bwd(&NonZero { v })
        } else {
            self.to_device(&candle_core::Device::Cpu)?
                .apply_op1_no_bwd(&NonZero { v })?
                .to_device(original_device)
        }
    }
}

pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=True.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;

    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=False.
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let (topk_indices, topk_values) = if self.device().is_gcu() {
            let (topk_values, topk_indices) = candle_nn::ops::topk(self, topk)?;
            (topk_indices, topk_values)
        } else {
            #[cfg(feature = "cuda")]
            let (values, sorted_indices) = self.sort(false)?;
            #[cfg(not(feature = "cuda"))]
            let (values, sorted_indices) = self.sort_last_dim(false)?;
            let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
            let topk_values = values.narrow(D::Minus1, 0, topk)?.contiguous()?;
            (topk_indices, topk_values)
        };
        Ok(TopKOutput {
            values: topk_values,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        if self.device().is_gcu() {
            return self.topk(topk);
        }
        // Sorted descending
        // let TopKOutput { values, indices } = self.topk(topk)?;
        let (values, indices) = self.sort_last_dim(false)?;
        // Reorder the indices ascending
        #[cfg(feature = "cuda")]
        let reorder_indices = indices.arg_sort(true)?;
        #[cfg(not(feature = "cuda"))]
        let reorder_indices = indices.arg_sort_last_dim(true)?;
        let topk_indices_unsorted = indices.gather(&reorder_indices, D::Minus1)?;
        let topk_values_unsorted = values.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
    fn split2<D: Dim>(&self, splits: &[usize], dim: D) -> Result<(Tensor, Tensor)>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }

    fn split2<D: Dim>(&self, splits: &[usize], dim: D) -> Result<(Tensor, Tensor)> {
        assert!(splits.len() == 2, "splits must be 2");
        let dim = dim.to_index(self.shape(), "split")?;
        Ok((
            self.narrow(dim, 0, splits[0])?,
            self.narrow(dim, splits[0], splits[1])?,
        ))
    }
}

pub trait BincountOp {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>>;
}

fn bincount(values: &[u32], minlength: u32) -> Vec<u32> {
    // Find the maximum value in `values` (or zero if empty)
    let max_val = values.par_iter().max().copied().unwrap_or(0);

    // The final size of the bin counts must be at least `minlength`
    // and large enough to include the largest value in `values`.
    let result_len = (max_val + 1).max(minlength);

    // Each thread creates a local histogram (`fold`),
    // and then they are merged together (`reduce`).
    values
        .par_iter()
        .fold(
            // Create a local histogram
            || vec![0u32; result_len as usize],
            // Update the local histogram
            |mut local_counts, &val| {
                local_counts[val as usize] += 1;
                local_counts
            },
        )
        // Merge histograms from all threads
        .reduce(
            // Identity (empty histogram)
            || vec![0u32; result_len as usize],
            // Combine two histograms
            |mut global_counts, local_counts| {
                for (g, l) in global_counts.iter_mut().zip(local_counts) {
                    *g += l;
                }
                global_counts
            },
        )
}

impl BincountOp for Tensor {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>> {
        let values = self.to_vec1::<u32>()?;

        Ok(bincount(&values, minlength))
    }
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

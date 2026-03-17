use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::path::Path;

fn main() {
    let in_file_path = match &std::env::args().collect::<Vec<_>>()[..] {
        [_, in_file] => Path::new(in_file).to_path_buf(),
        args => {
            eprintln!("Usage: {} IN_FILE", args[0]);
            std::process::exit(1);
        }
    };

    let (ptx, ptx_entry_param_sources) = spirt_ptx::spv_file_to_ptx(&in_file_path);

    let dump_ptx = || {
        // FIXME(eddyb) this is different from `with_extension`, which replaces
        // the extension - `spirt` should settle on one of the two, everywhere.
        let ptx_out = in_file_path.with_added_extension("ptx");
        std::fs::write(&ptx_out, &ptx).unwrap();
        ptx_out
    };

    let always_dump = true;
    // HACK(eddyb) make the PTX available for e.g. separate testing via `ptxas`.
    if always_dump {
        eprintln!("loading PTX kernel `{}`...", dump_ptx().display());
    }

    // TODO(eddyb) make this depend on the input SPIR-V.
    let ptx_kernel_name = "kernel";

    let attempt_loading_cuda = unsafe { cudarc::driver::sys::is_culib_present() }
        .then_some(())
        .ok_or_else(|| "can't load CUDA libraries".to_string());

    let (stream, f) = attempt_loading_cuda
        .and_then(|()| {
            CudaContext::new(0)
                .and_then(|ctx| {
                    let module = ctx.load_module(Ptx::from_src(&ptx))?;
                    Ok((ctx.default_stream(), module.load_function(ptx_kernel_name)?))
                })
                .map_err(|e| e.to_string())
        })
        .unwrap_or_else(|e| {
            eprint!("loading CUDA kernel failed ({e})");
            if !always_dump {
                eprint!(", dumped PTX to `{}`", dump_ptx().display());
            }
            eprintln!();
            std::process::exit(1);
        });

    // TODO(eddyb) move this into lodestar demo!
    let mut lodestar_heap = vec![0; 1024];
    lodestar_heap[0] = u32::try_from(lodestar_heap.len() * 4).unwrap();

    let mut buf_dev = stream.clone_htod(&lodestar_heap).unwrap();

    // FIXME(eddyb) pull this from the SPIR-V execution mode details!
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    {
        use cudarc::driver::CudaViewMut;
        use std::num::NonZeroU32;

        enum Binding<'a> {
            BufferMutU32(CudaViewMut<'a, u32>),
        }
        enum DescriptorMeta {
            BufferLen { stride: NonZeroU32, len: u32 },
        }
        enum DescriptorPtr<'a, 'b> {
            BufferMutU32(&'a mut CudaViewMut<'b, u32>),
        }

        let mut descriptor_sets = [[Binding::BufferMutU32(buf_dev.as_view_mut())]];
        let mut descriptor_sets = descriptor_sets.each_mut().map(|set| {
            set.each_mut().map(|binding| {
                let meta = match binding {
                    Binding::BufferMutU32(buf) => DescriptorMeta::BufferLen {
                        stride: NonZeroU32::new(4).unwrap(),
                        len: buf.len().try_into().unwrap(),
                    },
                };
                let ptr = match binding {
                    Binding::BufferMutU32(buf) => DescriptorPtr::BufferMutU32(buf),
                };
                (Some(ptr), meta)
            })
        });

        let mut launch_args = stream.launch_builder(&f);

        for param in ptx_entry_param_sources {
            match param {
                spirt_ptx::PtxEntryParamSource::BufferDataPtrGeneric64 { buffer: _ } => {
                    // FIXME(eddyb) extract these from `buffer`.
                    let (descriptor_set, binding) = (0, 0);

                    match descriptor_sets[descriptor_set][binding].0.take().unwrap() {
                        DescriptorPtr::BufferMutU32(view) => {
                            launch_args.arg(view);
                        }
                    }
                }
                spirt_ptx::PtxEntryParamSource::BufferLen {
                    buffer: _,
                    fixed_base_size,
                    dyn_unit_stride: expected_stride,
                } => {
                    // FIXME(eddyb) extract these from `buffer`.
                    let (descriptor_set, binding) = (0, 0);

                    assert_eq!(fixed_base_size, 0);
                    match &descriptor_sets[descriptor_set][binding].1 {
                        DescriptorMeta::BufferLen { stride, len } => {
                            // HACK(eddyb) allow the desired stride to be larger,
                            // but this indicates `spirt::mem::analyze` caused
                            // `spirt::qptr::lift` to "distort" the buffer shape.
                            if expected_stride.get().is_multiple_of(stride.get()) {
                                launch_args.arg(&*Box::leak(Box::new(
                                    len / (expected_stride.get() / *stride),
                                )));
                                continue;
                            }

                            assert_eq!(*stride, expected_stride);
                            launch_args.arg(len);
                        }
                    }
                }
            }
        }

        unsafe { launch_args.launch(cfg).unwrap() };
    }

    let buf_host = stream.clone_dtoh(&buf_dev).unwrap();

    // TODO(eddyb) move this into lodestar demo!
    {
        let heap_unused_range_end = buf_host[0];
        let heap_contents = &buf_host[(heap_unused_range_end / 4) as usize..];

        // FIXME(eddyb) is this the best spot to do this decoding in?
        let mut heap_allocs = vec![];
        let mut next_alloc_footer_end = heap_contents.len() * 4;
        while next_alloc_footer_end > 0 {
            let alloc_end = next_alloc_footer_end - 4;
            let alloc_start = (heap_contents[alloc_end / 4] - heap_unused_range_end) as usize;
            next_alloc_footer_end = alloc_start;

            heap_allocs.push(&heap_contents[(alloc_start / 4)..(alloc_end / 4)]);
        }

        for alloc in heap_allocs {
            for (chunk_idx, chunk) in alloc.chunks(8).enumerate() {
                print!(" {}", if chunk_idx == 0 { '-' } else { ' ' });
                for &word in chunk {
                    print!(" {word:08x}");
                }
                println!();
            }
        }
    }
}

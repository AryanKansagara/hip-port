# HIP Port

A CUDA-to-HIP porting assistant that runs in the browser. Paste or upload CUDA sources and get translated HIP code, a list of unsupported or risky APIs, warp-size-64 hazards, library mappings (cuBLAS to hipBLAS, cuDNN to MIOpen, CUB to hipCUB and more) and a ROCm readiness checklist for your PyTorch stack.

Live: https://hip-port.vercel.app

## The pain

hipify gets most names right. The time goes into everything after: reductions that assume 32-lane warps and silently return half the sum on 64-lane Instinct wavefronts, 32-bit ballot masks, inline PTX, deprecated texture references, and a Python environment pinned to `+cu121` wheels, flash-attn and bitsandbytes builds that need ROCm forks. HIP Port finds all of that in one pass and turns it into a worklist.

## Features

- **Translator**: comment- and string-aware, over an explicit mapping table plus library families (runtime, driver, cuBLAS/cuBLASLt, cuFFT, cuRAND, cuSPARSE, cuSOLVER, cuDNN to MIOpen, CUB, Thrust, NCCL, NVTX, fp16/bf16/fp8 types, headers). Project-defined macros are never renamed; unknown CUDA names are reported, not guessed. Optional `hipLaunchKernelGGL` launch rewriting, coverage report per file.
- **Hazard detectors**: hard-coded warp size, shuffle reductions starting at 16, shuffles without width, 32-bit masks and ballots, lane bit shifts, warp-synchronous tails, inline PTX, texture references, cooperative groups, device-side launches, sm_80-only intrinsics, WMMA, `__CUDA_ARCH__` guards. Severity follows the gfx target (wave64 CDNA vs wave32 RDNA).
- **Python stack check** for requirements.txt and environment.yml against a dated package table, with a rewritten `requirements-rocm.txt`.
- **Targets**: gfx942, gfx950, gfx90a, gfx1201, gfx1100, gfx1030 with wave size, matrix cores and FP8 format (FNUZ vs OCP).
- **Actions**: zip of translated files, worklist CSV with status, printable readiness report, side-by-side diff per file.
- **Workspace**: projects in IndexedDB, scan history with compare and progress chart, import validation report, optional account sync (Supabase), export and delete.
- Samples are synthetic and labelled.

## Run locally

```sh
npm install
npm run dev      # http://localhost:5173
npm test         # engine tests
npm run build
```

Optional sync: copy `.env.example` to `.env` with `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY`.

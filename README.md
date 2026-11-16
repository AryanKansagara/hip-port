# HIP Port

A CUDA-to-HIP porting assistant that runs in the browser. Paste or upload CUDA sources and get translated HIP code, a list of unsupported or risky APIs, warp-size-64 hazards, library mappings (cuBLAS to hipBLAS, cuDNN to MIOpen, CUB to hipCUB and more) and a ROCm readiness checklist for your PyTorch stack.

Live: https://hip-port.vercel.app

## The pain

hipify gets most names right. The time goes into everything after: reductions that assume 32-lane warps and silently return half the sum on 64-lane Instinct wavefronts, 32-bit ballot masks, inline PTX, deprecated texture references, and a Python environment pinned to `+cu121` wheels, flash-attn and bitsandbytes builds that need ROCm forks. HIP Port finds all of that in one pass and turns it into a worklist.

## Features

- **Translator**: comment- and string-aware, over the rename and unsupported tables generated from AMD's `hipify-perl` (about 5,200 renames and 3,600 unsupported identifiers) plus a curated table of porting notes (runtime, driver, cuBLAS/cuBLASLt, cuFFT, cuRAND, cuSPARSE, cuSOLVER, cuTENSOR, NVRTC, cuDNN to MIOpen, CUB, Thrust, NCCL, NVTX, fp16/bf16/fp8 types, headers). Applies hipify's argument fix-ups (`HIP_SYMBOL`, `reinterpret_cast<const void*>`), de-duplicates collapsed includes, never renames project-defined names, and reports unknown CUDA names instead of guessing. Optional `hipLaunchKernelGGL` launch rewriting, coverage report per file.
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

## Mapping data

`src/engine/hipifyTable.ts` is generated from [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) `bin/hipify-perl` (MIT licence). To refresh it:

```sh
curl -sLO https://raw.githubusercontent.com/ROCm/HIPIFY/amd-staging/bin/hipify-perl
node scripts/gen-hipify-table.mjs hipify-perl <commit> <yyyy-mm-dd>
npm test
```

The script only reads the file as text. `npm test` then checks that the curated notes in `mapping.ts` still agree with the official names.

The tests run the translator over real sources in `src/engine/fixtures/` (HIPIFY's own unit tests with their expected output, and NVIDIA's vectorAdd sample); see the README there for sources and licences.

## Known gaps

- Text-based, not a compiler front end: it cannot tell that a plain function name is a template (hipify-clang adds `HIP_KERNEL_NAME` there), does not expand macros that hide a `<<<...>>>` launch, and does not reorder arguments for the few calls where hipify-clang does.
- The hipify table tracks the newest HIP release, so a handful of names may be missing from an older ROCm install.
- cuDNN to MIOpen covers handle, descriptor and the common forward calls only; everything else is flagged for a hand port.
- Generic device helpers hipify lists as unsupported (`umin`, `saturate`, `mul24`) are not flagged, because a text pass cannot tell them from your own functions.
- Python package statuses are a dated, curated table.

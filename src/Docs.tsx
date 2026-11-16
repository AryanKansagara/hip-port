import { RULES } from './engine/hazards.ts'
import { LIBS, TABLE_SIZE } from './engine/mapping.ts'
import { HIPIFY_SOURCE } from './engine/hipifyTable.ts'
import { TARGETS } from './engine/gfx.ts'
import { ROCM_INDEX } from './engine/pydeps.ts'
import { Code, Link } from './ui.tsx'

const SECTIONS = [['input', 'Getting your code in'], ['python', 'Python dependencies'], ['translate', 'What gets translated'], ['hazards', 'Hazard rules'], ['targets', 'Targets'], ['outputs', 'Outputs'], ['data', 'Your data'], ['assumptions', 'Dated assumptions']] as const

export default function Docs() {
  return (
    <section className="page docs">
      <aside className="docs-nav" aria-label="On this page">
        <div className="eyebrow">Docs</div>
        {SECTIONS.map(([id, l]) => <a key={id} href={`#${id}`}>{l}</a>)}
      </aside>
      <div className="docs-body">
        <h1>Using HIP Port</h1>
        <p className="lede">HIP Port translates CUDA C++ to HIP, flags what will not work on AMD GPUs, and checks your Python stack for ROCm. Everything runs in your browser.</p>

        <h2 id="input">Getting your code in</h2>
        <p>Create a project on <Link href="/app">Projects</Link>, open <b>Sources</b>, and drop files, a folder, or a zip. HIP Port reads <code>.cu .cuh .cpp .cc .cxx .c .h .hpp .hh .hxx .inl</code> plus <code>requirements*.txt</code> and <code>environment*.yml</code>. Anything else is listed in the validation report with the reason it was skipped. Limits: 1 MB per file (bigger files are almost always generated), 25 MB per project.</p>
        <h3>Zip the tracked sources of a git repo</h3>
        <p>This includes only committed files, which keeps build output and checkpoints out:</p>
        <Code label="shell" code={'git archive --format=zip -o my-repo.zip HEAD'} />
        <h3>Only the CUDA parts of a big repo</h3>
        <p>For a large Python project with a few extensions, zip just the kernel sources:</p>
        <Code label="shell" code={"git ls-files '*.cu' '*.cuh' 'csrc/*' 'requirements*.txt' | zip my-kernels.zip -@"} />
        <h3>Single kernels</h3>
        <p>Paste into <b>Paste code</b> on the Sources tab and give it a file name ending in <code>.cu</code>. Re-importing a path replaces the old copy, and each import saves a dated scan to History.</p>
        <p>Zips are read with the browser&rsquo;s built-in decompressor. Stored and deflate entries work; Zip64 (archives over 4 GB) and encrypted zips do not.</p>

        <h2 id="python">Python dependencies</h2>
        <p>Upload the file your environment is actually built from. If you only have a working environment, export it:</p>
        <Code label="shell" code={'# pip / venv\npip freeze > requirements.txt\n\n# conda (no build strings, so names match across platforms)\nconda env export --no-builds > environment.yml'} />
        <p>Each package is matched against a curated table of ROCm status: <b>works</b> (pure Python, or builds itself on ROCm), <b>ROCm build</b> (a ROCm wheel or index exists), <b>fork or source build</b> (AMD maintains a fork, for example ROCm/flash-attention), <b>partly supported</b>, and <b>CUDA only</b>. Pins with a <code>+cu121</code>-style local version and CUDA package indexes are flagged. Direct wheel URLs are read too: the package name comes from the wheel file name, and a <code>cu128</code> path segment or local version marks it as a CUDA build. The Python tab offers a rewritten <code>requirements-rocm.txt</code> that points at <code>{ROCM_INDEX}</code>.</p>

        <h2 id="translate">What gets translated</h2>
        <p>The source is scanned once to blank out comments and string literals, so <code>"cudaMalloc failed"</code> in a log message stays as written. Header names in <code>#include</code> lines are kept visible and mapped. Then each identifier is looked up in {TABLE_SIZE.toLocaleString('en-US')} entries: a curated table with porting notes first, then the rename and unsupported lists generated from AMD&rsquo;s own <code>hipify-perl</code>. A name hipify lists as unsupported is reported as such; it is never given an invented <code>hip</code> name.</p>
        <div className="table-wrap"><table className="data">
          <thead><tr><th>CUDA</th><th>ROCm</th><th>Notes</th></tr></thead>
          <tbody>{Object.values(LIBS).map(l => <tr key={l.cuda}><td>{l.cuda}</td><td><b>{l.rocm}</b></td><td className="small">{l.note ?? ''}</td></tr>)}</tbody>
        </table></div>
        <ul>
          <li>Names your project defines itself (a <code>CUDA_CHECK</code> macro, a <code>cudaCheck()</code> helper) are never renamed.</li>
          <li>Member access (<code>cfg.cudaGraph</code>) is left alone.</li>
          <li>Calls, <code>_t</code> types and upper-case constants with a CUDA prefix that are in neither table are reported as <b>unmapped</b> and left untouched rather than guessed. Your own variables (<code>cublasHandle</code>, <code>cuda_ctx</code>) are not.</li>
          <li>The argument fix-ups hipify applies are applied here too: <code>HIP_SYMBOL()</code> around the symbol in <code>hipMemcpyToSymbol</code>, <code>hipMemcpyFromSymbol</code> and <code>hipGetSymbolAddress</code>, <code>reinterpret_cast&lt;const void*&gt;</code> around the kernel in <code>hipFuncSetCacheConfig</code> and <code>hipLaunchKernel</code>, and the dropped callback argument of the module occupancy calls.</li>
          <li>When several CUDA headers map to one HIP header, only the first include is kept.</li>
          <li>cuDNN is mapped to MIOpen from the curated table only. hipify&rsquo;s cuDNN rows target hipDNN, which is no longer maintained, so they are not used.</li>
          <li>Library headers use the installed ROCm 6+ paths (<code>hipblas/hipblas.h</code>, <code>hipsolver/hipsolver.h</code>, <code>hipsparse/hipsparse.h</code>) where hipify still writes the flat names.</li>
          <li><code>kernel&lt;&lt;&lt;g, b, smem, stream&gt;&gt;&gt;(args)</code> is kept by default (hipcc accepts it). Tick <b>hipLaunchKernelGGL</b> to rewrite launches, including templated kernels (<code>HIP_KERNEL_NAME</code>).</li>
          <li><code>.cu</code> files are written as <code>.hip</code>; headers keep their names so includes still resolve. <code>#include &lt;hip/hip_runtime.h&gt;</code> is added where a file uses CUDA but includes nothing.</li>
        </ul>

        <h2 id="hazards">Hazard rules</h2>
        <p>These detectors find code that compiles under HIP but misbehaves, or code that will not compile. Severity depends on the target: warp-size rules are critical on wave64 Instinct GPUs and low on wave32 Radeon GPUs, where they still matter for portability.</p>
        <div className="table-wrap"><table className="data">
          <thead><tr><th>Rule</th><th>Wave64</th><th>Wave32</th><th>Why and how to fix</th></tr></thead>
          <tbody>{RULES.map(r => <tr key={r.id}><td><b>{r.title}</b><div className="mono tiny muted">{r.id}</div></td><td><span className={`sev ${r.sev[0]}`}>{r.sev[0]}</span></td><td><span className={`sev ${r.sev[1]}`}>{r.sev[1]}</span></td><td className="small">{r.message} <em>{r.fix}</em></td></tr>)}</tbody>
        </table></div>

        <h2 id="targets">Targets</h2>
        <div className="table-wrap"><table className="data">
          <thead><tr><th>gfx</th><th>GPUs</th><th>Wave</th><th>FP8</th><th>Support</th></tr></thead>
          <tbody>{TARGETS.map(t => <tr key={t.id}><td className="mono">{t.id}</td><td>{t.gpus}</td><td>{t.wave}</td><td>{t.fp8 === 'none' ? 'none' : t.fp8.toUpperCase()}</td><td className="small">{t.rocm === 'supported' ? 'ROCm data center' : t.rocm === 'consumer' ? 'ROCm Radeon' : 'community only'}</td></tr>)}</tbody>
        </table></div>
        <p>Check your GPU with <code>rocminfo | grep gfx</code>, then set <code>PYTORCH_ROCM_ARCH</code> to that value before building extensions.</p>

        <h2 id="outputs">Outputs</h2>
        <ul>
          <li><b>Download .zip</b>: every translated file, <code>requirements-rocm.txt</code> and <code>HIP_PORT_WORKLIST.csv</code>.</li>
          <li><b>Worklist CSV</b>: columns <code>severity, category, file, line, title, detail, fix, status</code>. Status is what you set in the Worklist tab (open, done, wontfix).</li>
          <li><b>Report</b>: a print-friendly readiness report; use your browser&rsquo;s Save as PDF.</li>
          <li><b>Files &amp; diff</b>: side-by-side diff per file with hazards in the gutter, copy and single-file download.</li>
        </ul>

        <h2 id="data">Your data</h2>
        <p>Sources are read and translated in your browser and stored in IndexedDB on this device. Accounts are optional: signing in syncs project settings, worklist status, scan history and, for projects under 1.5 MB of source, the sources themselves. Bigger trees stay on the device and the project shows a notice on other devices. The account menu can export everything as JSON, delete your cloud data, or delete the account.</p>

        <h2 id="assumptions">Dated assumptions</h2>
        <ul>
          <li>Renames, unsupported identifiers and CUDA deprecation versions are generated from <a href={`${HIPIFY_SOURCE.repo}/blob/${HIPIFY_SOURCE.commit}/${HIPIFY_SOURCE.file}`} target="_blank" rel="noreferrer">ROCm/HIPIFY <code>{HIPIFY_SOURCE.file}</code></a> (branch {HIPIFY_SOURCE.branch}, commit <code>{HIPIFY_SOURCE.commit.slice(0, 8)}</code>, fetched {HIPIFY_SOURCE.fetched}, MIT licence). That table tracks the newest HIP; an older ROCm install can lack a few of the names. hipBLAS 2.x semantics are assumed for <code>GemmEx</code> compute types.</li>
          <li>The translator is checked against hipify&rsquo;s own unit tests (headers, runtime, driver, cuBLAS, cuSPARSE, cuSOLVER and cuRAND: 2,524 of 2,527 expected lines match; the rest need a compiler front end) and against NVIDIA cuda-samples, Rodinia, darknet, faiss and llama.cpp sources.</li>
          <li>Target facts (wave size, FP8 format, support tier) from the ROCm 7.0 compatibility matrix and LLVM AMDGPU usage docs, November 2026.</li>
          <li>Python package statuses checked October 2026. vLLM, bitsandbytes and flash-attention move fast on ROCm; re-check before a big migration.</li>
          <li>Readiness score: 100 &times; e<sup>&minus;cost/100</sup>, where each open item costs 12 (critical), 5 (high), 1.5 (medium) or 0.4 (low). Resolved items cost nothing.</li>
        </ul>
      </div>
    </section>
  )
}

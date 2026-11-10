import { useEffect, useMemo, useState } from 'react'
import { detect, RULE_BY_ID } from './engine/hazards.ts'
import { mask } from './engine/lex.ts'
import { target, TARGETS } from './engine/gfx.ts'
import { diffLines } from './engine/scan.ts'
import { HERO_KERNEL } from './engine/sample.ts'
import { translate } from './engine/translate.ts'
import { TABLE_SIZE } from './engine/mapping.ts'
import { byLine, Segs, worst } from './CodeView.tsx'
import { highlight } from './hl.ts'
import { Link } from './ui.tsx'

const reduced = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

function HeroDemo() {
  const [src, setSrc] = useState(HERO_KERNEL)
  const [tid, setTid] = useState('gfx942')
  const [beam, setBeam] = useState(() => reduced() ? 999 : 0)
  const [pick, setPick] = useState<number | null>(null)
  const t = target(tid)
  const tr = useMemo(() => translate(src), [src])
  const hazards = useMemo(() => detect(src, mask(src), t), [src, t])
  const hz = useMemo(() => byLine(hazards), [hazards])
  const L = useMemo(() => highlight(src), [src])
  const R = useMemo(() => highlight(tr.out), [tr.out])
  const changed = useMemo(() => new Set(diffLines(src, tr.out).filter(r => r.type !== 'same' && r.b).map(r => r.b!)), [src, tr.out])
  const nL = L.length

  // The scan beam sweeps down once, translating and flagging as it goes.
  useEffect(() => {
    if (beam >= nL) return
    const id = setTimeout(() => setBeam(b => b + 1), beam === 0 ? 700 : 85)
    return () => clearTimeout(id)
  }, [beam, nL])

  const shown = hazards.filter(h => h.line <= beam)
  const active = pick ? hz.get(pick) ?? [] : []
  const apis = tr.uses.filter(u => u.kind === 'api' && u.hip && u.hip !== u.cuda).reduce((s, u) => s + u.lines.length, 0)
  const maxLen = Math.max(...src.split('\n').map(l => l.length))

  return (
    <div className="hero-demo" aria-label="Live CUDA to HIP translation">
      <div className="demo-bar">
        <span className="dots" aria-hidden="true"><i /><i /><i /></span>
        <span className="demo-file">dot.cu <span aria-hidden="true">→</span> dot.hip</span>
        <label className="demo-target">
          <span className="sr-only">Target GPU</span>
          <select value={tid} onChange={e => setTid(e.target.value)}>
            {TARGETS.filter(x => ['gfx942', 'gfx950', 'gfx1201', 'gfx1100'].includes(x.id)).map(x => <option key={x.id} value={x.id}>{x.id} · wave{x.wave}</option>)}
          </select>
        </label>
      </div>
      <div className="demo-panes">
        <div className="pane">
          <div className="pane-label">CUDA <small>edit me</small></div>
          <div className="editor">
            <div className="ed-gutter" aria-hidden="true">
              {L.map((_, i) => {
                const h = hz.get(i + 1)
                const w = h && i + 1 <= beam ? worst(h) : null
                return <div key={i} className={`gl${pick === i + 1 ? ' on' : ''}`}>{w ? <button type="button" tabIndex={-1} className={`mark ${w}`} onClick={() => setPick(pick === i + 1 ? null : i + 1)} /> : null}<i>{i + 1}</i></div>
              })}
            </div>
            <div className="ed-body">
              <pre className="ed-hl" aria-hidden="true">{L.map((segs, i) => <div key={i} className={`cl${hz.has(i + 1) && i + 1 <= beam ? ` hz ${worst(hz.get(i + 1)!)}` : ''}${i + 1 === beam ? ' beam' : ''}${pick === i + 1 ? ' picked' : ''}`}><Segs segs={segs} />{'\n'}</div>)}</pre>
              <textarea
                className="ed-input" spellCheck={false} wrap="off" aria-label="CUDA source (editable)"
                value={src} style={{ height: `${nL * 1.6 + 1.2}em`, width: `max(100%, ${maxLen + 3}ch)` }}
                onChange={e => { setSrc(e.target.value); setBeam(999) }}
              />
            </div>
          </div>
        </div>
        <div className="pane out">
          <div className="pane-label">HIP <small>translated by the engine</small></div>
          <pre className="ed-hl static">{R.map((segs, i) => <div key={i} className={`cl${changed.has(i + 1) && i < beam ? ' chg' : ''}${i < beam ? '' : ' pending'}`}><span className="ln">{i + 1}</span><Segs segs={segs} />{'\n'}</div>)}</pre>
        </div>
      </div>
      <div className="demo-foot">
        <div className="demo-stats" aria-live="polite">
          <span><b>{apis}</b> API calls renamed</span>
          <span><b>{shown.length}</b> hazards on {t.id}</span>
          <span><b>{shown.filter(h => h.severity === 'critical').length}</b> critical</span>
        </div>
        <div className="demo-detail">
          {active.length ? active.map((h, i) => <p key={i}><span className={`sev ${h.severity}`}>{h.severity}</span> <b>Line {h.line}: {RULE_BY_ID[h.rule].title}.</b> {h.message} <em>{h.fix}</em></p>)
            : <p className="muted">{beam < nL ? 'Scanning…' : `Click a marker in the gutter. Switch to ${t.wave === 64 ? 'gfx1201 (wave32)' : 'gfx942 (wave64)'} to see severities change.`}</p>}
        </div>
      </div>
    </div>
  )
}

function Wavefront() {
  const [wave, setWave] = useState<32 | 64>(64)
  const [step, setStep] = useState(0)
  // Offsets a reduction written for 32 lanes uses: 16, 8, 4, 2, 1.
  const offsets = [16, 8, 4, 2, 1]
  useEffect(() => {
    if (reduced()) { setStep(5); return }
    const id = setInterval(() => setStep(s => (s + 1) % 7), 900)
    return () => clearInterval(id)
  }, [])
  const s = Math.min(step, 5)
  // After the first s butterfly steps, lane 0 holds lanes whose index uses only the offsets applied so far.
  const used = offsets.slice(0, s).reduce((a, o) => a | o, 0)
  const lanes = Array.from({ length: 64 }, (_, i) => i)
  const reached = (i: number) => i < 32 && (i & ~used) === 0
  const lost = wave === 64 && s === 5 ? 32 : 0
  return (
    <div className="wavefront">
      <div className="seg-control" role="radiogroup" aria-label="Wavefront size">
        <button type="button" role="radio" aria-checked={wave === 32} className={wave === 32 ? 'on' : ''} onClick={() => setWave(32)}>NVIDIA warp · 32</button>
        <button type="button" role="radio" aria-checked={wave === 64} className={wave === 64 ? 'on' : ''} onClick={() => setWave(64)}>AMD Instinct wavefront · 64</button>
      </div>
      <div className="lanes" role="img" aria-label={`for (o = 16; o > 0; o >>= 1) on a ${wave}-lane ${wave === 32 ? 'warp' : 'wavefront'}: ${wave === 64 ? 'lanes 32 to 63 never reach lane 0' : 'all 32 lanes reach lane 0'}`}>
        {lanes.map(i => <i key={i} className={`${i >= wave ? 'off' : reached(i) ? 'in' : s === 5 && i >= 32 ? 'lost' : ''}`} />)}
      </div>
      <div className="wf-cap">
        <code>for (o = 16; o &gt; 0; o &gt;&gt;= 1)</code>
        <span>step {s}/5{s ? `, offset ${offsets[s - 1]}` : ''}</span>
        <span className={lost ? 'bad' : 'good'}>{s < 5 ? 'folding…' : lost ? `${lost} lanes never reach lane 0: the sum is half the data` : 'all lanes reach lane 0'}</span>
      </div>
    </div>
  )
}

const QUOTES = [
  { src: 'r/ROCm', text: 'What is the actual state of PyTorch and AI coding on AMD cards right now? Every guide seems to be out of date.' },
  { src: 'r/ROCm', text: 'Finally got PyTorch running smoothly on AMD with ROCm, after a week of matching wheel indexes and environment variables.' },
  { src: 'r/ROCm', text: 'A full ROCm PyTorch setup guide: tuning plus a long troubleshooting section, because that is where all the time goes.' },
  { src: 'r/LocalLLaMA', text: 'Spent forever getting R9700s working in vLLM, only to find the int4 and fp8 kernels for that card are not written yet.' },
]

export function Landing() {
  return (
    <>
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-copy">
            <div className="eyebrow">CUDA → HIP · ROCm readiness</div>
            <h1>Port CUDA to <span className="accent">AMD GPUs</span> without guessing what breaks.</h1>
            <p className="lede">Paste or drop your CUDA sources. HIP Port translates them with the hipify mapping table, flags the code that compiles but computes the wrong answer on 64-lane wavefronts, maps every library to its ROCm twin and checks your Python stack, in your browser.</p>
            <div className="row gap wrap">
              <Link href="/app?sample=mini-repo" className="btn primary lg">Try the sample repo</Link>
              <Link href="/app" className="btn outline lg">Import your code</Link>
            </div>
            <p className="muted small">No upload, no account needed. Sources stay on this device.</p>
          </div>
          <HeroDemo />
        </div>
      </section>

      <section className="band">
        <div className="wrap-lg">
          <div className="section-head">
            <div className="eyebrow">Why this exists</div>
            <h2>hipify gets the names right. The hard part is everything after.</h2>
            <p className="muted">Paraphrased from engineers moving real workloads to AMD:</p>
          </div>
          <div className="quotes">
            {QUOTES.map((q, i) => <figure key={i} className="quote"><blockquote>{q.text}</blockquote><figcaption>{q.src}</figcaption></figure>)}
          </div>
        </div>
      </section>

      <section className="wrap-lg features">
        <article className="feature">
          <div className="f-num">01</div>
          <h3>Translate</h3>
          <p>A comment- and string-aware translator over {TABLE_SIZE}+ explicit mappings plus library families: runtime and driver API, cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, cuDNN to MIOpen, CUB to hipCUB, Thrust, NCCL, NVTX. Optional hipLaunchKernelGGL launches. Coverage report per file.</p>
        </article>
        <article className="feature">
          <div className="f-num">02</div>
          <h3>Find the silent bugs</h3>
          <p>Hard-coded warp size 32, shuffles without width, 32-bit ballot masks, warp-synchronous tails, inline PTX, texture references, cooperative groups, device-side launches and sm_80-only intrinsics. Severity follows your gfx target.</p>
        </article>
        <article className="feature">
          <div className="f-num">03</div>
          <h3>Check the Python stack</h3>
          <p>requirements.txt or environment.yml against a dated table: ROCm wheels, AMD forks (flash-attention, apex), partial support (bitsandbytes, vLLM on RDNA4) and CUDA-only packages. Get a rewritten requirements file.</p>
        </article>
        <article className="feature">
          <div className="f-num">04</div>
          <h3>Ship the port</h3>
          <p>Download the translated tree as a zip, a severity-ranked worklist CSV, a printable readiness report, and track every scan so you can show progress week over week.</p>
        </article>
      </section>

      <section className="band alt">
        <div className="wrap-lg split">
          <div>
            <div className="eyebrow">The 64-lane problem</div>
            <h2>Code that compiles is not code that is correct.</h2>
            <p>A warp reduction that starts at offset 16 folds 32 lanes. On Instinct GPUs a wavefront has 64, so lanes 32 to 63 never reach lane 0 and the result is quietly wrong. hipcc will not warn you. HIP Port does, on the exact line, with the fix.</p>
            <ul className="ticks">
              <li>Critical on CDNA (gfx90a, gfx942, gfx950), low on RDNA wave32 targets</li>
              <li>Ballot masks and lane shifts checked for 64-bit width</li>
              <li>FP8 encodings checked per target: FNUZ on MI300, OCP on MI350 and RDNA4</li>
            </ul>
          </div>
          <Wavefront />
        </div>
      </section>

      <section className="wrap-lg targets-sec">
        <div className="section-head">
          <div className="eyebrow">Targets</div>
          <h2>Pick the GPU you are porting to</h2>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th>gfx</th><th>GPUs</th><th>Wave</th><th>Matrix</th><th>FP8</th><th>What changes</th></tr></thead>
            <tbody>{TARGETS.map(x => <tr key={x.id}><td className="mono">{x.id}</td><td>{x.gpus}</td><td>{x.wave}</td><td>{x.matrix}</td><td>{x.fp8 === 'none' ? '—' : x.fp8.toUpperCase()}</td><td className="small">{x.note}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="muted small">Checked against the ROCm 7.0 compatibility matrix and LLVM AMDGPU docs, November 2026.</p>
      </section>

      <section className="cta-band">
        <div className="wrap-lg row between wrap gap">
          <h2>Bring a repo. Leave with a worklist.</h2>
          <div className="row gap wrap">
            <Link href="/app?sample=mini-repo" className="btn primary lg">Open the sample</Link>
            <Link href="/docs" className="btn outline lg">Read the docs</Link>
          </div>
        </div>
      </section>
    </>
  )
}

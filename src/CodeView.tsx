import { Fragment, useMemo } from 'react'
import type { Hazard, Severity } from './engine/hazards.ts'
import { SEVERITIES } from './engine/hazards.ts'
import type { DiffRow } from './engine/scan.ts'
import { highlight, type Seg } from './hl.ts'

export const worst = (hs: Hazard[]): Severity | null => hs.length ? SEVERITIES.find(s => hs.some(h => h.severity === s)) ?? null : null

export function Segs({ segs }: { segs: Seg[] }) {
  return <>{segs.map((s, i) => s[0] ? <span key={i} className={`t-${s[0]}`}>{s[1]}</span> : <Fragment key={i}>{s[1]}</Fragment>)}</>
}

export function byLine(hazards: Hazard[]) {
  const m = new Map<number, Hazard[]>()
  for (const h of hazards) m.set(h.line, [...(m.get(h.line) ?? []), h])
  return m
}

/** Side-by-side diff of one file with hazard markers in the left gutter. */
export function DiffView({ rows, left, right, hazards, focusLine, onPick, wrap }: { rows: DiffRow[]; left: string; right: string; hazards: Hazard[]; focusLine?: number; onPick?(line: number): void; wrap?: boolean }) {
  const L = useMemo(() => highlight(left), [left])
  const R = useMemo(() => highlight(right), [right])
  const hz = useMemo(() => byLine(hazards), [hazards])
  return (
    <div className={`diff${wrap ? ' wrap' : ''}`} role="table" aria-label="Original CUDA and translated HIP, side by side">
      <div className="diff-head" role="row"><span role="columnheader">CUDA (original)</span><span role="columnheader">HIP (translated)</span></div>
      {rows.map((r, i) => {
        const h = r.a ? hz.get(r.a) : undefined
        const w = h ? worst(h) : null
        return (
          <div key={i} role="row" className={`drow ${r.type}${r.a && r.a === focusLine ? ' focus' : ''}`} id={r.a ? `L${r.a}` : undefined}>
            <span className="gut" role="cell">
              {w ? <button type="button" className={`mark ${w}`} title={h!.map(x => x.message).join('\n')} aria-label={`${h!.length} ${w} hazard${h!.length > 1 ? 's' : ''} on line ${r.a}`} onClick={() => onPick?.(r.a!)} /> : null}
              <i>{r.a ?? ''}</i>
            </span>
            <code className="cell l" role="cell">{r.a ? <Segs segs={L[r.a - 1] ?? []} /> : ''}</code>
            <span className="gut r" role="cell"><i>{r.b ?? ''}</i></span>
            <code className="cell r" role="cell">{r.b ? <Segs segs={R[r.b - 1] ?? []} /> : ''}</code>
          </div>
        )
      })}
    </div>
  )
}

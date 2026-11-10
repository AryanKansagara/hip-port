// Display-only syntax highlighting for CUDA / HIP C++. Returns one segment list per line.
export type Seg = [kind: '' | 'c' | 's' | 'k' | 'n' | 'p' | 'a' | 'q' | 'h', text: string]

const KW = new Set('auto bool break case char const constexpr continue default do double else enum extern float for if inline int long namespace nullptr return short signed sizeof static struct switch template typename typedef unsigned using void volatile while true false class public private dim3 size_t'.split(' '))
const QUAL = /^(?:__global__|__device__|__host__|__shared__|__constant__|__forceinline__|__restrict__|__launch_bounds__|threadIdx|blockIdx|blockDim|gridDim|warpSize)$/
const CUDA = /^(?:cuda|cu[A-Z]|CU_|CUDA_|cublas|CUBLAS|cufft|CUFFT|curand|CURAND|cusparse|cudnn|CUDNN|nvtx|__shfl|__ballot|__popc|__syncwarp|__activemask|__nv_|__reduce_)/
const HIP = /^(?:hip|HIP|miopen|roctx|rocwmma|hipcub|__hip_)/

export function highlight(src: string): Seg[][] {
  const lines: Seg[][] = [[]]
  const push = (k: Seg[0], t: string) => {
    const parts = t.split('\n')
    parts.forEach((p, i) => {
      if (i > 0) lines.push([])
      if (p) lines[lines.length - 1].push([k, p])
    })
  }
  const re = /(\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|$))|("(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|(^[ \t]*#[ \t]*\w+)|([A-Za-z_]\w*)|(\b\d[\w.]*\b)|(\s+|.)/gm
  for (const m of src.matchAll(re)) {
    if (m[1]) push('c', m[1])
    else if (m[2]) push('s', m[2])
    else if (m[3]) push('p', m[3])
    else if (m[4]) push(KW.has(m[4]) ? 'k' : QUAL.test(m[4]) ? 'q' : CUDA.test(m[4]) ? 'a' : HIP.test(m[4]) ? 'h' : '', m[4])
    else if (m[5]) push('n', m[5])
    else push('', m[6])
  }
  // Merge adjacent plain segments to keep the DOM small.
  return lines.map(l => l.reduce<Seg[]>((acc, s) => { const last = acc[acc.length - 1]; if (last && last[0] === s[0]) last[1] += s[1]; else acc.push([s[0], s[1]]); return acc }, []))
}

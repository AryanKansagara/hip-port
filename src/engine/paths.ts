// File paths come from archives other people made, and go into archives unpacked on other machines.

/** Forward slashes, no drive, leading slash, `.` or resolved `..`; null when the path climbs out of its folder. */
export function cleanPath(path: string): string | null {
  const out: string[] = []
  for (const part of path.replace(/^(?:[\\/]*[A-Za-z]:)+/, '').split(/[\\/]/)) {
    if (part === '' || part === '.') continue
    if (part !== '..') out.push(part)
    else if (out.pop() === undefined) return null
  }
  return out.join('/')
}

/** A path safe to unpack that no earlier call with the same `used` returned. Sources stored before paths were checked can still climb. */
export function exportPath(path: string, used: Set<string>): string {
  const safe = cleanPath(path) ?? cleanPath(path.split(/[\\/]/).filter(p => p !== '..').join('/'))
  let name = safe || 'unnamed'
  for (let n = 2; used.has(name); n++) name = (safe || 'unnamed').replace(/(\.[^./]*)?$/, ext => `-${n}${ext}`)
  used.add(name)
  return name
}

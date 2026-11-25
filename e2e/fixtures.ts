import { test as base, expect, type Page } from '@playwright/test'
import { deflateRawSync, inflateRawSync, crc32 } from 'node:zlib'
import { readFile } from 'node:fs/promises'

/** The only other hosts the page asks for: web fonts in index.html. They are answered with an empty stylesheet. */
const FONT_HOSTS = new Set(['fonts.googleapis.com', 'fonts.gstatic.com'])

export const test = base.extend<{ external: string[] }>({
  external: async ({}, use) => { await use([]) },

  // Only the app is reachable: any other host fails the test, and so does any uncaught page error.
  context: async ({ context, baseURL, external }, use) => {
    const errors: string[] = []
    const origin = new URL(baseURL!).origin
    await context.route('**/*', (route) => {
      const url = new URL(route.request().url())
      if (url.origin === origin) return route.continue()
      if (FONT_HOSTS.has(url.host)) return route.fulfill({ status: 200, contentType: 'text/css', body: '' })
      external.push(url.host)
      return route.abort('blockedbyclient')
    })
    context.on('weberror', (e) => errors.push(String(e.error())))
    await use(context)
    expect(errors, 'uncaught page errors').toEqual([])
    expect(external, 'requests to hosts other than the app').toEqual([])
  },
})

export { expect }

// ---------- Driving the app ----------

export type Upload = { name: string; data: string | Uint8Array }

/** Starts an empty project from the workspace and lands on its Sources tab. Returns the project path. */
export async function newProject(page: Page): Promise<string> {
  await page.goto('/app')
  await page.getByRole('button', { name: /^(Import your code|New project)$/ }).click()
  await expect(page.getByRole('heading', { name: 'Paste code' })).toBeVisible()
  return new URL(page.url()).pathname.replace(/\/sources$/, '')
}

export const tab = (page: Page, name: 'Overview' | 'Files & diff' | 'Worklist' | 'Python stack' | 'History' | 'Sources') =>
  page.getByRole('navigation', { name: 'Project sections' }).getByRole('link', { name }).click()

/** Feeds files to the import control. */
export async function upload(page: Page, files: Upload[]) {
  await page.locator('input[type=file]:not([webkitdirectory])').setInputFiles(
    files.map((f) => ({ name: f.name, mimeType: 'application/octet-stream', buffer: Buffer.from(f.data) })),
  )
}

/** Imports files and waits for the validation report to count them. */
export async function importFiles(page: Page, files: Upload[], imported: number, skipped = 0) {
  await upload(page, files)
  await expect(page.getByText(`${imported} imported · ${skipped} skipped of ${imported + skipped} files`)).toBeVisible()
}

export async function paste(page: Page, name: string, text: string) {
  await page.getByLabel('File name').fill(name)
  await page.getByLabel('Contents').fill(text)
  await page.getByRole('button', { name: 'Add to project' }).click()
}

/** A new project with the given files, on the Overview tab. */
export async function project(page: Page, files: Upload[]): Promise<string> {
  const path = await newProject(page)
  await importFiles(page, files, files.length)
  await tab(page, 'Overview')
  await expect(page.locator('.kpi').first()).toBeVisible()
  return path
}

export const kpi = (page: Page, label: string) => page.locator('.kpi').filter({ has: page.locator(`> span:text-is("${label}")`) })
export const check = (page: Page, title: string | RegExp) => page.locator('.checklist li').filter({ has: page.locator('b', { hasText: title }) })
export async function setTarget(page: Page, gfx: string) {
  await page.getByLabel('Target GPU').selectOption(gfx)
  await expect(page.getByLabel('Target GPU')).toHaveValue(gfx)
}
export const worklistRow = (page: Page, title: string | RegExp) => page.locator('table.worklist tbody tr').filter({ has: page.locator('td b', { hasText: title }) })

/** The validation report's line for one file. */
export const reportRow = (page: Page, path: string) =>
  page.locator('.panel').filter({ has: page.getByRole('heading', { name: 'Validation report' }) }).locator('tbody tr').filter({ has: page.locator(`td:first-child:text-is("${path}")`) })

/** The translated side of the open file, joined back into text. */
export async function translated(page: Page): Promise<string> {
  const [, text] = await download(page, page.getByRole('button', { name: 'Download file' }))
  return text
}

export async function download(page: Page, button: ReturnType<Page['locator']>): Promise<[string, string]> {
  const [d] = await Promise.all([page.waitForEvent('download'), button.click()])
  return [d.suggestedFilename(), await readFile(await d.path(), 'utf8')]
}

export async function downloadBytes(page: Page, button: ReturnType<Page['locator']>): Promise<[string, Buffer]> {
  const [d] = await Promise.all([page.waitForEvent('download'), button.click()])
  return [d.suggestedFilename(), await readFile(await d.path())]
}

// ---------- Zip, written and read independently of the app's own code ----------

export interface ZipFile { path: string; data: string | Uint8Array; stored?: boolean; method?: number; flags?: number; crc?: number }

export function zip(files: ZipFile[]): Buffer {
  const parts: Buffer[] = [], central: Buffer[] = []
  let offset = 0
  for (const f of files) {
    const raw = Buffer.from(f.data), name = Buffer.from(f.path)
    const body = f.stored ? raw : deflateRawSync(raw)
    const head = Buffer.alloc(26)
    head.writeUInt16LE(20, 0); head.writeUInt16LE(f.flags ?? 0, 2); head.writeUInt16LE(f.method ?? (f.stored ? 0 : 8), 4)
    head.writeUInt32LE(f.crc ?? crc32(raw), 10); head.writeUInt32LE(body.length, 14); head.writeUInt32LE(raw.length, 18); head.writeUInt16LE(name.length, 22)
    const local = Buffer.concat([Buffer.from('PK\x03\x04'), head, name, body])
    central.push(Buffer.concat([Buffer.from('PK\x01\x02'), Buffer.from([20, 0]), head, Buffer.alloc(10), u32(offset), name]))
    parts.push(local)
    offset += local.length
  }
  const dir = Buffer.concat(central)
  const end = Buffer.alloc(18)
  end.writeUInt16LE(files.length, 4); end.writeUInt16LE(files.length, 6); end.writeUInt32LE(dir.length, 8); end.writeUInt32LE(offset, 12)
  return Buffer.concat([...parts, dir, Buffer.from('PK\x05\x06'), end])
}
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b }

/** Reads a zip by its central directory, checking every CRC. */
export function unzip(b: Buffer): Record<string, string> {
  const eocd = b.lastIndexOf('PK\x05\x06')
  expect(eocd, 'end of central directory').toBeGreaterThanOrEqual(0)
  let p = b.readUInt32LE(eocd + 16)
  const out: Record<string, string> = {}
  for (let n = b.readUInt16LE(eocd + 10); n > 0; n--) {
    const size = b.readUInt32LE(p + 20), nameLen = b.readUInt16LE(p + 28), local = b.readUInt32LE(p + 42)
    const name = b.toString('utf8', p + 46, p + 46 + nameLen)
    const start = local + 30 + b.readUInt16LE(local + 26) + b.readUInt16LE(local + 28)
    const body = b.subarray(start, start + size)
    const data = b.readUInt16LE(p + 10) === 8 ? inflateRawSync(body) : body
    expect(crc32(data), `CRC of ${name}`).toBe(b.readUInt32LE(p + 16))
    expect(Object.keys(out), 'each path once').not.toContain(name)
    out[name] = data.toString('utf8')
    p += 46 + nameLen + b.readUInt16LE(p + 30) + b.readUInt16LE(p + 32)
  }
  return out
}

/** The launch-style box is written to IndexedDB before it repaints, so it is clicked and then waited for. */
export async function useGgl(page: Page) {
  await page.getByLabel('hipLaunchKernelGGL').click()
  await expect(page.getByLabel('hipLaunchKernelGGL')).toBeChecked()
}

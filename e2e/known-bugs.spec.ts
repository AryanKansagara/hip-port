// Real defects found while writing this suite. Each is pinned with test.fail until its issue is fixed.
import { test, expect, newProject, project, importFiles, upload, paste, reportRow, tab, zip, kpi, check, setTarget, useGgl, worklistRow, translated, download, downloadBytes, unzip } from './fixtures.ts'

const RT = '#include <hip/hip_runtime.h>\n'
const rocm = async (page: import('@playwright/test').Page, name: string, text: string) => {
  await project(page, [{ name, data: text }])
  await tab(page, 'Python stack')
  const [, out] = await download(page, page.getByRole('button', { name: 'Download requirements-rocm.txt' }))
  return out.split('\n').slice(2).filter(Boolean)
}

test('#2 a comma inside template arguments does not split a call argument', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: `${RT}cudaFuncSetAttribute(kern<float, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, n);\nk<<<div_up<int, 8>(n), 256>>>(x);\n` }])
  await useGgl(page)
  await tab(page, 'Files & diff')
  expect((await translated(page)).split('\n').slice(1, 3)).toEqual([
    'hipFuncSetAttribute(reinterpret_cast<const void*>(kern<float, 4>), hipFuncAttributeMaxDynamicSharedMemorySize, n);',
    'hipLaunchKernelGGL(k, dim3(div_up<int, 8>(n)), dim3(256), 0, 0, x);',
  ])
})

test.fail('#3 FP8 types follow the target: OCP on gfx950 and gfx1201, FNUZ on gfx942', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: `${RT}__nv_fp8_e4m3 a; __nv_fp8x2_e5m2 b;\n` }])
  await tab(page, 'Files & diff')
  const line = async () => (await translated(page)).split('\n')[1]
  expect(await line()).toBe('__hip_fp8_e4m3_fnuz a; __hip_fp8x2_e5m2_fnuz b;')
  for (const t of ['gfx950', 'gfx1201']) {
    await setTarget(page, t)
    await expect(page.locator('.drow.mod .cell.r')).toHaveText('__hip_fp8_e4m3 a; __hip_fp8x2_e5m2 b;')
  }
})

test.fail('#4 the usual spellings of a 32-lane assumption are found', async ({ page }) => {
  const lines = ['const int WARP_SIZE = 32;', 'static constexpr uint32_t kWarpSize = 32u;', '#define WARPSIZE (32)', 'enum { WARP_LANES = 32 };', 'int a = threadIdx.x & 0x1f;', 'int b = tid % 32u;', 'for (unsigned int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(m, v, o, 32);']
  await project(page, [{ name: 'k.cu', data: lines.join('\n') + '\n' }])
  await tab(page, 'Files & diff')
  await expect(page.locator('.diff .mark.critical')).toHaveCount(lines.length, { timeout: 5000 })
})

test.fail('#5 requirements-rocm.txt keeps markers, git URLs, hashes and other indexes, and stays installable', async ({ page }) => {
  expect(await rocm(page, 'requirements.txt', `--extra-index-url https://pypi.example.test/simple
peft @ git+https://github.com/huggingface/peft.git@v0.13.0
git+https://github.com/x/y.git@v1#egg=mylib
torch==2.1.0 ; sys_platform == "darwin"
torch==2.3.1+cu121 ; sys_platform == "linux"
einops==0.8.0 \\
    --hash=sha256:aaaa \\
    --hash=sha256:bbbb
jax[cuda12]==0.4.30
`)).toEqual([
    '--extra-index-url https://pypi.example.test/simple',
    'peft @ git+https://github.com/huggingface/peft.git@v0.13.0',
    'git+https://github.com/x/y.git@v1#egg=mylib',
    'torch==2.1.0 ; sys_platform == "darwin"',
    'torch==2.3.1 ; sys_platform == "linux"',
    'einops==0.8.0 --hash=sha256:aaaa --hash=sha256:bbbb',
    'jax==0.4.30',
  ])
})

test.fail('#5 a conda environment becomes lines pip accepts', async ({ page }) => {
  expect(await rocm(page, 'environment.yml', `dependencies:
  - python=3.11
  - pytorch=2.3.0=py3.11_cuda12.1_cudnn8.9.2_0
  - numpy 1.26.*
  - scipy=1.11
  - pip
`)).toEqual([
    '# removed python: set by the environment, not installed with pip.',
    'torch==2.3.0',
    'numpy==1.26.*',
    'scipy==1.11.*',
    'pip',
  ])
})

test.fail('#6 a dependency file that cannot be read does not pass its checks', async ({ page }) => {
  await project(page, [{ name: 'requirements.txt', data: 'Please sign in to continue\n' }])
  await expect(check(page, 'Remove CUDA-only packages')).not.toHaveClass('pass', { timeout: 5000 })
  await tab(page, 'Python stack')
  await expect(page.getByText('Line 1: Could not read “Please sign in to continue”')).toBeVisible({ timeout: 5000 })
})

test.fail('#6 conda flow lists, quoted entries and nested pip problems are read or reported', async ({ page }) => {
  await project(page, [{ name: 'environment.yml', data: 'channels: [nvidia]\ndependencies:\n  - "pytorch-cuda=12.1"\n  - pip:\n    - "torch==2.3.1+cu121"\n    - -r base.txt\n' }])
  await expect(check(page, 'Remove CUDA-only packages')).toHaveClass('todo', { timeout: 5000 })
  await tab(page, 'Python stack')
  await expect(page.locator('table.data tbody tr')).toHaveCount(2, { timeout: 5000 })
  await expect(page.getByText('Line 6: Includes another file (-r base.txt)')).toBeVisible({ timeout: 5000 })
  await expect(page.getByText('Conda channel nvidia only serves CUDA packages.')).toBeVisible({ timeout: 5000 })
})

test.fail('#7 a package pinned twice keeps its worst severity', async ({ page }) => {
  // torch+cu121 is high (5): 100*e^-0.05 = 95.12 -> 95. The darwin line alone would be low (0.4).
  await project(page, [{ name: 'requirements.txt', data: 'torch==2.1.0 ; sys_platform == "darwin"\ntorch==2.3.1+cu121 ; sys_platform == "linux"\n' }])
  await expect(kpi(page, 'Readiness').locator('b')).toHaveText('95/100', { timeout: 5000 })
  await expect(check(page, 'Install torch from the ROCm wheel index')).toHaveClass('todo', { timeout: 5000 })
})

for (const [what, file, message] of [
  ['a damaged entry', { path: 'a/k.cu', data: 'cudaMalloc(&p, 4);\n', stored: true, crc: 1 }, /k\.cu.*damaged/],
  ['an encrypted entry', { path: 'a/k.cu', data: 'cudaMalloc(&p, 4);\n', stored: true, flags: 1 }, /k\.cu.*encrypted/],
] as const) {
  test.fail(`#8 a zip with ${what} is refused`, async ({ page }) => {
    await newProject(page)
    await upload(page, [{ name: 'x.zip', data: zip([file]) }])
    await expect(page.getByRole('alert')).toHaveText(message, { timeout: 5000 })
    await expect(page.locator('.src-list li')).toHaveCount(0)
  })
}

test.fail('#8 an unreadable entry the import would skip anyway does not block the rest', async ({ page }) => {
  await newProject(page)
  await upload(page, [{ name: 'x.zip', data: zip([{ path: 'a/k.cu', data: 'cudaMalloc(&p, 4);\n' }, { path: 'a/weights.bin', data: 'x', stored: true, method: 93 }]) }])
  await expect(page.locator('.src-list li .mono')).toHaveText(['k.cu'], { timeout: 5000 })
})

test.fail('#9 undoing a saved scan does not reopen an item resolved since', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: 'int lane = threadIdx.x % 32;\n' }])
  await tab(page, 'Worklist')
  await page.getByRole('button', { name: 'Save scan' }).first().click()
  const saved = page.locator('.toast', { hasText: 'Saved' })
  await expect(saved).toBeVisible()
  await worklistRow(page, 'warp size').getByRole('combobox').selectOption('done')
  await expect(page.getByText('1 of 1 resolved')).toBeVisible()
  await saved.getByRole('button', { name: 'Undo' }).click()
  await tab(page, 'History')
  await expect(page.locator('table.data tbody tr')).toHaveCount(1)
  await tab(page, 'Worklist')
  await expect(page.getByText('1 of 1 resolved')).toBeVisible({ timeout: 5000 })
})

test.fail('#10 pasting a file with a folder keeps the folder', async ({ page }) => {
  await project(page, [{ name: 'util.h', data: 'int root;\n' }])
  await tab(page, 'Sources')
  await paste(page, 'src/util.h', 'int nested;\n')
  await expect(page.locator('.src-list li .mono')).toHaveText(['src/util.h', 'util.h'], { timeout: 5000 })
})

test.fail('#11 a project without any CUDA shows no coverage and no clean-kernel checks', async ({ page }) => {
  await project(page, [{ name: 'requirements.txt', data: 'numpy\n' }])
  await expect(kpi(page, 'API coverage').locator('b')).toHaveText('—', { timeout: 5000 })
  await expect(check(page, /wavefront-size clean/)).not.toHaveClass('pass', { timeout: 5000 })
})

test.fail('#11 an HTML page or a cut-off source saved as .cu is called out', async ({ page }) => {
  await newProject(page)
  await importFiles(page, [{ name: 'page.cu', data: '<!doctype html>\n<html><body>Sign in</body></html>\n' }, { name: 'cut.cu', data: 'cudaMalloc(&p, 4);\n/* the rest of this file was lost\n' }], 2)
  await expect(reportRow(page, 'page.cu')).toContainText(/HTML/, { timeout: 5000 })
  await expect(reportRow(page, 'cut.cu')).toContainText(/comment/, { timeout: 5000 })
})

test.fail('#12 a file name cannot become a spreadsheet formula', async ({ page }) => {
  await project(page, [{ name: '=HYPERLINK("http:evil","x").cu', data: 'int lane = threadIdx.x % 32;\n' }])
  await tab(page, 'Worklist')
  const [, csv] = await download(page, page.getByRole('button', { name: 'Export CSV' }))
  expect(csv).not.toMatch(/(^|,)"?=/m)
})

test.fail('#12 zip paths cannot climb out of the folder they are unpacked in', async ({ page }) => {
  await project(page, [{ name: 'ok.cu', data: 'int lane = threadIdx.x % 32;\n' }])
  await tab(page, 'Sources')
  await upload(page, [{ name: 'x.zip', data: zip([{ path: 'a/b.cu', data: 'cudaFree(p);\n' }, { path: '../../evil.cu', data: 'cudaFree(p);\n' }]) }])
  await expect(page.getByText(/of 2 files/)).toBeVisible()
  const [, bytes] = await downloadBytes(page, page.getByRole('button', { name: 'Download .zip' }))
  expect(Object.keys(unzip(bytes)).filter((p) => p.split('/').includes('..'))).toEqual([])
})

test.fail('#13 an import the browser refuses to store says so', async ({ page }) => {
  await newProject(page)
  await page.evaluate(() => { IDBObjectStore.prototype.put = () => { throw new DOMException('full', 'QuotaExceededError') } })
  await upload(page, [{ name: 'k.cu', data: 'int lane = threadIdx.x % 32;\n' }])
  await expect(page.getByRole('alert')).toContainText(/stor/i, { timeout: 5000 })
  await expect(page.locator('.toast', { hasText: 'Imported 1 file' })).toHaveCount(0)
})

test.fail('#14 a Latin-1 source is not exported with its bytes replaced', async ({ page }) => {
  await newProject(page)
  await importFiles(page, [{ name: 'k.cu', data: new Uint8Array(Buffer.from('// José\ncudaFree(p);\n', 'latin1')) }], 1)
  await tab(page, 'Files & diff')
  expect(await translated(page)).toContain('// José')
})

test.fail('#15 replacing a large file does not count it twice towards the project limit', async ({ page }) => {
  test.slow()
  await newProject(page)
  // 14 files of 0.9 MB = 12.6 MB, imported twice: 12.6 MB of the 25 MB limit, not 25.2 MB.
  const body = `cudaFree(p);\n${' '.repeat(899_000)}\n`
  const files = Array.from({ length: 14 }, (_, i) => ({ path: `r/f${i}.cu`, data: body + ' '.repeat(1000) }))
  const data = zip(files)
  await importFiles(page, [{ name: 'r.zip', data }], 14)
  await expect(page.locator('.src-list li')).toHaveCount(14)
  await upload(page, [{ name: 'r.zip', data }])
  await expect(page.getByText('14 imported · 0 skipped of 14 files')).toBeVisible()
  await expect(page.locator('.log li')).toHaveCount(2)
  await expect(page.getByText('Project size limit (25 MB) reached')).toHaveCount(0)
})

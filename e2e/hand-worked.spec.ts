// Every number and every output line here was worked out on paper from the documented rules, not copied from the app.
import { test, expect, project, kpi, check, setTarget, tab, worklistRow, download, downloadBytes, translated, unzip, useGgl } from './fixtures.ts'

/*
  Readiness = round(100 * e^(-cost/100)); critical 12, high 5, medium 1.5, low 0.4.
  line 2  #define WARP_SIZE 32   warp-size-32      critical on wave64, low on wave32
  line 4  threadIdx.x % 32       warp-size-32      critical on wave64, low on wave32
  line 5  asm                    inline-ptx        critical on both
  line 9  cudaFrobnicate         not mapped        medium
  gfx942:  3*12 + 1.5 = 37.5  -> 100*e^-0.375 = 68.73 -> 69
  gfx1100: 12 + 1.5 + 2*0.4 = 14.3 -> 100*e^-0.143 = 86.68 -> 87
  gfx942 with the PTX item done: 25.5 -> 100*e^-0.255 = 77.49 -> 77
  API coverage: cuda_runtime.h, cudaMalloc convert; cudaFrobnicate does not -> 2 of 3 = 67%
*/
const KERNEL = `#include <cuda_runtime.h>
#define WARP_SIZE 32
__global__ void k(float* x) {
  int lane = threadIdx.x % 32;
  asm volatile("nop");
}
void run(float* p, int n) {
  cudaMalloc(&p, n); // cudaMalloc in a comment stays
  cudaFrobnicate(p);
  puts("cudaFree in a string stays");
  k<<<1, 64>>>(p);
}
`
const HIP = `#include <hip/hip_runtime.h>
#define WARP_SIZE 32
__global__ void k(float* x) {
  int lane = threadIdx.x % 32;
  asm volatile("nop");
}
void run(float* p, int n) {
  hipMalloc(&p, n); // cudaMalloc in a comment stays
  cudaFrobnicate(p);
  puts("cudaFree in a string stays");
  k<<<1, 64>>>(p);
}
`

test('readiness, coverage and severities of a twelve-line kernel on wave64 and wave32', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: KERNEL }])
  await expect(kpi(page, 'Readiness').locator('b')).toHaveText('69/100')
  await expect(kpi(page, 'API coverage').locator('b')).toHaveText('67%')
  await expect(kpi(page, 'API coverage').locator('em')).toHaveText('2 of 3 references mechanical')
  await expect(kpi(page, 'Open items').locator('b')).toHaveText('4')
  await expect(page.getByRole('img', { name: '3 critical, 0 high, 1 medium, 0 low' })).toBeVisible()
  await expect(check(page, 'Kernels are wavefront-size clean (gfx942 runs wave64)')).toContainText('2 warp-size assumptions found.')

  await setTarget(page, 'gfx1100')
  await expect(kpi(page, 'Readiness').locator('b')).toHaveText('87/100')
  await expect(page.getByRole('img', { name: '1 critical, 0 high, 1 medium, 2 low' })).toBeVisible()

  await setTarget(page, 'gfx942')
  await tab(page, 'Worklist')
  await worklistRow(page, 'Inline PTX assembly').getByRole('combobox').selectOption('done')
  await expect(page.getByText('1 of 4 resolved')).toBeVisible()
  await tab(page, 'Overview')
  await expect(kpi(page, 'Readiness').locator('b')).toHaveText('77/100')
})

test('the translated file, in both launch styles', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: KERNEL }])
  await tab(page, 'Files & diff')
  await expect(page.locator('.file-bar')).toContainText('k.cu → k.hip')
  expect(await translated(page)).toBe(HIP)
  await useGgl(page)
  await expect(page.getByRole('table', { name: /side by side/ })).toContainText('hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, p);')
  expect(await translated(page)).toBe(HIP.replace('k<<<1, 64>>>(p);', 'hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, p);'))
})

/*
  torch +cu121 pin   high 5
  CUDA index         high 5
  nvidia-ml-py       high 5 (CUDA only)
  vllm               medium 1.5 (fork)
  numpy, einops      nothing
  cost 16.5 -> 100*e^-0.165 = 84.79 -> 85
*/
const REQ = `--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.1+cu121
numpy>=1.26,<3
nvidia-ml-py
vllm==0.5.3
einops
`

test('a requirements file: statuses, readiness and the rewritten file', async ({ page }) => {
  await project(page, [{ name: 'requirements.txt', data: REQ }, { name: 'k.cu', data: 'int x;\n' }])
  await expect(kpi(page, 'Readiness').locator('b')).toHaveText('85/100')
  await expect(check(page, 'Install torch from the ROCm wheel index')).toHaveClass('todo')
  await expect(check(page, 'Install torch from the ROCm wheel index')).toContainText('Line 2 pins a CUDA build (+cu121)')
  await tab(page, 'Python stack')
  await expect(kpi(page, 'Packages').locator('b')).toHaveText('5')
  await expect(kpi(page, 'Ready on ROCm').locator('b')).toHaveText('2') // torch, numpy
  await expect(kpi(page, 'Forks or partial').locator('b')).toHaveText('1') // vllm
  await expect(kpi(page, 'CUDA only').locator('b')).toHaveText('1') // nvidia-ml-py
  await expect(page.getByText('Line 1: package index https://download.pytorch.org/whl/cu121 serves CUDA wheels.')).toBeVisible()
  const [name, text] = await download(page, page.getByRole('button', { name: 'Download requirements-rocm.txt' }))
  expect(name).toBe('requirements-rocm.txt')
  expect(text).toBe(`# Rewritten for ROCm by HIP Port. Package table checked 2026-10; verify versions against your ROCm release.
--extra-index-url https://download.pytorch.org/whl/rocm7.0
torch==2.3.1
numpy>=1.26,<3
# amdsmi ships with ROCm (pip install /opt/rocm/share/amd_smi)
# vllm: use the rocm/vllm Docker image or build with VLLM_TARGET_DEVICE=rocm
einops
`)
})

test('the downloaded zip holds the translated tree, the requirements and the worklist', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: KERNEL }, { name: 'plain.cpp', data: 'int add(int a, int b) { return a + b; }\n' }, { name: 'requirements.txt', data: REQ }])
  const [name, bytes] = await downloadBytes(page, page.getByRole('button', { name: 'Download .zip' }))
  expect(name).toBe('port-1-hip.zip')
  const files = unzip(bytes)
  expect(Object.keys(files).sort()).toEqual(['HIP_PORT_WORKLIST.csv', 'k.hip', 'plain.cpp', 'requirements-rocm.txt'])
  expect(files['k.hip']).toBe(HIP)
  expect(files['plain.cpp']).toBe('int add(int a, int b) { return a + b; }\n')
  expect(files['requirements-rocm.txt']).toContain('torch==2.3.1\n')
  // 3 critical + (torch, index, nvidia-ml-py) high + (cudaFrobnicate, vllm) medium, worst first.
  const rows = files['HIP_PORT_WORKLIST.csv'].trimEnd().split('\n')
  expect(rows[0]).toBe('severity,category,file,line,title,detail,fix,status')
  expect(rows.slice(1).map((r) => r.split(',').slice(0, 4).join(','))).toEqual([
    'critical,hazard,k.cu,2', 'critical,hazard,k.cu,4', 'critical,hazard,k.cu,5',
    'high,dependency,requirements.txt,1', 'high,dependency,requirements.txt,2', 'high,dependency,requirements.txt,4',
    'medium,api,k.cu,9', 'medium,dependency,requirements.txt,5',
  ])
  expect(rows.slice(1).every((r) => r.endsWith(',open'))).toBe(true)
})

test('two scans compare by item: one fixed, one new, three still open', async ({ page }) => {
  await project(page, [{ name: 'k.cu', data: KERNEL }])
  await tab(page, 'Sources')
  // The PTX goes, a texture reference (high 5) arrives: 24 + 1.5 + 5 = 30.5 -> 100*e^-0.305 = 73.71 -> 74
  const changed = KERNEL.replace('  asm volatile("nop");\n', '').replace('#define', 'texture<float, 2> tex;\n#define')
  await page.getByLabel('File name').fill('k.cu')
  await page.getByLabel('Contents').fill(changed)
  await page.getByRole('button', { name: 'Add to project' }).click()
  await expect(page.locator('.toast').last()).toContainText('Readiness 74.')
  await tab(page, 'History')
  await expect(page.getByRole('heading', { name: '1 fixed' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '1 new' })).toBeVisible()
  await expect(page.getByText('3 still open · readiness +5')).toBeVisible()
})

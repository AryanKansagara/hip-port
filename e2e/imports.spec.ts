import { test, expect, newProject, importFiles, upload, paste, reportRow, tab, zip, project } from './fixtures.ts'

const CU = '#include <cuda_runtime.h>\n__global__ void k(float* x) { x[0] = 1.f; }\n'

test.describe('accepted', () => {
  test('several files, with a reason for each one skipped', async ({ page }) => {
    await newProject(page)
    await importFiles(page, [
      { name: 'k.cu', data: CU }, { name: 'h.hpp', data: '﻿#pragma once\r\nint x;\r\n' }, { name: 'environment.yml', data: 'dependencies:\n  - numpy\n' },
      { name: 'notes.md', data: 'hi' }, { name: 'empty.cu', data: '' }, { name: 'bin.cu', data: new Uint8Array([1, 0, 2]) }, { name: 'big.h', data: 'x'.repeat(1_000_001) },
    ], 3, 4)
    await expect(reportRow(page, 'k.cu')).toContainText('added')
    await expect(reportRow(page, 'h.hpp')).toContainText('3 lines')
    await expect(reportRow(page, 'notes.md')).toContainText('Not a CUDA/C++ source or dependency file (.md)')
    await expect(reportRow(page, 'empty.cu')).toContainText('Empty file')
    await expect(reportRow(page, 'bin.cu')).toContainText('Binary content')
    await expect(reportRow(page, 'big.h')).toContainText('Larger than 1 MB (1.0 MB); likely generated')
    await expect(page.locator('.src-list li')).toHaveCount(3)
    await expect(page.locator('.log')).toContainText('files · 7 files · 3 imported, 4 skipped')
  })

  test('a folder-prefixed zip, deflated and stored, without its VCS and build folders', async ({ page }) => {
    await newProject(page)
    await importFiles(page, [{ name: 'repo.zip', data: zip([
      { path: 'repo-main/src/k.cu', data: CU }, { path: 'repo-main/src/h.h', data: 'int y;\n', stored: true }, { path: 'repo-main/requirements.txt', data: 'numpy\n' },
      { path: 'repo-main/.git/x.h', data: 'int z;\n' }, { path: 'repo-main/build/gen.cu', data: CU }, { path: 'repo-main/README', data: 'hi' },
    ]) }], 3, 3)
    await expect(page.locator('.src-list li .mono')).toHaveText(['src/h.h', 'src/k.cu', 'requirements.txt'])
    await expect(reportRow(page, '.git/x.h')).toContainText('Build, VCS or environment folder')
    await expect(reportRow(page, 'README')).toContainText('no extension')
  })

  test('pasting code and a dependency file, replacing and removing with undo', async ({ page }) => {
    await newProject(page)
    await paste(page, 'k.cu', CU)
    await expect(reportRow(page, 'k.cu')).toContainText('added')
    await paste(page, 'k.cu', CU + 'int q;\n')
    await expect(reportRow(page, 'k.cu')).toContainText('replaced')
    await paste(page, 'requirements.txt', 'torch==2.3.1+cu121\n')
    await expect(page.locator('.src-list')).toContainText('Python dependencies')
    await page.getByRole('button', { name: 'Remove k.cu' }).click()
    await expect(page.locator('.src-list li')).toHaveCount(1)
    await page.locator('.toast', { hasText: 'Removed k.cu' }).getByRole('button', { name: 'Undo' }).click()
    await expect(page.locator('.src-list li')).toHaveCount(2)
  })

  test('a second dependency file says which one it replaces', async ({ page }) => {
    await project(page, [{ name: 'requirements.txt', data: 'numpy\n' }])
    await tab(page, 'Sources')
    await importFiles(page, [{ name: 'environment.yml', data: 'dependencies:\n  - numpy\n' }], 1)
    await expect(reportRow(page, 'environment.yml')).toContainText('Replaces requirements.txt')
  })
})

test.describe('refused with a reason, and nothing imported', () => {
  const alert = (page: import('@playwright/test').Page) => page.getByRole('alert')
  const good = zip([{ path: 'a/k.cu', data: CU }])
  const cases: [string, Buffer, RegExp][] = [
    ['an empty zip file', Buffer.alloc(0), /Not a zip file/],
    ['a text file named .zip', Buffer.from('hello, not a zip at all, just words'), /Not a zip file/],
    ['a truncated zip', good.subarray(0, good.length - 30), /Not a zip file/],
    ['a zip whose directory was cut out', Buffer.concat([good.subarray(0, 40), good.subarray(-22)]), /Corrupt zip central directory/],
    ['a Zip64 archive', (() => { const b = Buffer.from(good); b.writeUInt32LE(0xffffffff, b.length - 6); return b })(), /Zip64 archives are not supported/],
    ['an unknown compression method', zip([{ path: 'a/k.cu', data: CU, stored: true, method: 12 }]), /unsupported compression method 12/],
  ]
  for (const [name, data, message] of cases) {
    test(name, async ({ page }) => {
      await newProject(page)
      await upload(page, [{ name: 'x.zip', data }])
      await expect(alert(page)).toHaveText(message)
      await expect(page.getByText('Add your sources to start.')).toBeVisible()
      await expect(page.locator('.log')).toHaveCount(0)
    })
  }

  test('only unsupported files', async ({ page }) => {
    await newProject(page)
    await upload(page, [{ name: 'a.py', data: 'x=1' }])
    await expect(alert(page)).toHaveText('Nothing was imported. See the reasons below.')
    await expect(reportRow(page, 'a.py')).toContainText('(.py)')
  })

  test('an empty paste', async ({ page }) => {
    await newProject(page)
    await paste(page, 'k.cu', '  \n')
    await expect(alert(page)).toHaveText('Paste some code first.')
  })
})

test('unreadable requirement lines are listed with their line numbers', async ({ page }) => {
  await project(page, [{ name: 'requirements.txt', data: 'numpy\n<!doctype html>\n-r base.txt\n' }])
  await tab(page, 'Python stack')
  await expect(page.getByText('Line 2: Could not read “<!doctype html>”')).toBeVisible()
  await expect(page.getByText('Line 3: Includes another file (-r base.txt); upload it too to check its packages.')).toBeVisible()
})

test('markup in file names, sources and packages is shown as text', async ({ page }) => {
  const evil = '<img src=x onerror=document.title=1>'
  await newProject(page)
  await importFiles(page, [
    { name: `${evil}.cu`, data: `// ${evil}\ncudaMalloc("${evil}");\nint lane = threadIdx.x % 32; /* ${evil} */\n` },
    { name: 'requirements.txt', data: `${evil}\n` },
  ], 2)
  for (const t of ['Overview', 'Files & diff', 'Worklist', 'Python stack', 'History'] as const) {
    await tab(page, t)
    await expect(page.locator('nav.tabs a[aria-current=page]')).toHaveText(t)
    await expect(page.locator('main img')).toHaveCount(0)
  }
  await expect(page).toHaveTitle(/HIP Port/)
})

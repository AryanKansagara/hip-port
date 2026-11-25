import test from 'node:test'
import assert from 'node:assert/strict'
import { ingest } from './ingest.ts'
import { coveragePct, scan } from './scan.ts'

const run = (path: string, text: string) => scan([{ path, text }], null, { target: 'gfx942', ggl: false })
const states = (r: ReturnType<typeof scan>) => ['wave', 'compile'].map(id => r.checklist.find(c => c.id === id)!.state)
const row = (path: string, text: string) => ingest({ files: [], deps: null }, [{ path, data: new TextEncoder().encode(text) }], false).rows[0]

test('no references is not 100% coverage', () => {
  assert.ok(Number.isNaN(coveragePct(scan([], { path: 'requirements.txt', text: 'numpy\n' }, { target: 'gfx942', ggl: false }).coverage)))
  assert.equal(coveragePct(run('k.cu', 'cudaFree(p);\n').coverage), 1)
})

test('kernel checks pass only when a kernel was read to its end', () => {
  assert.deepEqual(states(scan([], { path: 'requirements.txt', text: 'numpy\n' }, { target: 'gfx942', ggl: false })), ['info', 'info'])
  for (const t of ['int x;\n', 'cudaFree(p);\n', '// __global__ void k();\n', 'cudaFree(p);\n/* lost\n__global__ void k() {}\n']) assert.deepEqual(states(run('k.cu', t)), ['info', 'info'], t)
  for (const t of ['__global__ void k() {}\n', 'auto f = [=] __device__ (int i) {};\n', 'int i = threadIdx.x;\n']) assert.deepEqual(states(run('k.cu', t)), ['pass', 'pass'], t)
  for (const t of ['__global__ void k() {}\n/* lost\n', '__global__ void k() {}\nauto s = R"(lost\n']) assert.deepEqual(states(run('k.cu', t)), ['info', 'info'], t)
  assert.equal(states(run('k.cu', 'int a = threadIdx.x % 32;\n/* lost'))[0], 'todo')
})

test('HTML pages are refused', () => {
  for (const t of ['<!doctype html>\n<p>x', '\n\n<!DOCTYPE HTML PUBLIC "x">', '<html lang="en">', '<!-- c -->\n<html>', '<!-- a --><!-- b -->\n<!doctype html>', '<?xml version="1.0"?>\n<html>', '﻿<html>']) {
    assert.match(row('k.cu', t).reason!, /HTML/, t)
    assert.equal(row('requirements.txt', t).status, 'skipped', t)
  }
  assert.equal(row('k.cu', '<!-- x -->'.repeat(50_000)).status, 'added')
  for (const t of ['const char* s = "<html>";\n', '// <!doctype html>\n', 'a <html> b;\n', '#include <html.h>\n']) assert.equal(row('k.cu', t).status, 'added', t)
})

test('an unterminated comment or raw string is called out, a closed one is not', () => {
  assert.match(row('k.cu', 'int x;\n/* lost\nint y;\n').reason!, /comment opened on line 2 is never closed/)
  assert.match(row('k.cu', 'int x; /*/').reason!, /line 1/)
  assert.match(row('k.cu', 'auto s = R"(x\n').reason!, /raw string opened on line 1/)
  for (const t of ['/* a */ int x;', '// /* not a comment\nint x;', 'const char* s = "/*";\n', 'auto s = R"(/*)";\n', "char c = '/'; int y = 1 / *p;\n", '/* a **/', '/**/']) assert.equal(row('k.cu', t).reason, undefined, t)
})

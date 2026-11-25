import test from 'node:test'
import assert from 'node:assert/strict'
import { changeProject, changeSources, getData, getProject, putData, saveProject, type Project } from './store.ts'

const make = async (id: string) => {
  const p: Project = { id, name: id, createdAt: 0, updatedAt: 0, target: 'gfx942', ggl: false, status: {}, snapshots: [], imports: [] }
  await saveProject(p)
  return p
}

test('changes made at the same time all stick', async () => {
  await make('a')
  await Promise.all([
    changeProject('a', p => ({ status: { ...p.status, x: 'done' } })),
    changeProject('a', p => ({ status: { ...p.status, y: 'wontfix' } })),
    changeProject('a', () => ({ name: 'renamed' })),
    changeSources('a', (ds, p) => ({ ds: { ...ds, files: [...ds.files, { path: '1.cu', text: '' }] }, patch: { target: p.target + '!' } })),
    changeSources('a', ds => ({ ds: { ...ds, files: [...ds.files, { path: '2.cu', text: '' }] }, patch: {} })),
  ])
  const p = getProject('a')!
  assert.deepEqual([p.status, p.name, p.target], [{ x: 'done', y: 'wontfix' }, 'renamed', 'gfx942!'])
  assert.deepEqual((await getData('a'))!.files.map(f => f.path), ['1.cu', '2.cu'])
})

test('a change that fails does not stop the next one', async () => {
  await make('b')
  const failed = changeProject('b', () => { throw new Error('no') })
  await changeProject('b', () => ({ name: 'after' }))
  await assert.rejects(failed, /no/)
  assert.equal(getProject('b')!.name, 'after')
})

test('a change to a project that is gone does nothing', async () => {
  await changeProject('gone', () => ({ name: 'back' }))
  await changeSources('gone', ds => ({ ds, patch: {} }))
  assert.equal(getProject('gone'), undefined)
  assert.equal(await getData('gone'), undefined)
})

test('null leaves the sources and the record alone', async () => {
  const p = await make('c')
  await putData('c', { files: [{ path: 'k.cu', text: 'x' }], deps: null })
  await changeSources('c', () => null)
  assert.equal(getProject('c')!.name, p.name)
  assert.equal((await getData('c'))!.files.length, 1)
})

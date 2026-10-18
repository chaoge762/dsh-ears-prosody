/**
 * 自检：合成本次音频（每个参数都是我们已知的），再让分析器把它读回来。
 *
 * 这是"可验证"的做法——不是看输出像不像，而是：
 * 我构造了 3 段说话、2 处停顿、一段上扬、一段下沉，
 * 分析器必须把这几件事说回来。
 *
 *   node test/selfcheck.mjs
 */
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { decodeWav } from '../lib/wav.js'
import { analyze } from '../lib/hearing.js'

const SR = 16000
const here = dirname(fileURLToPath(import.meta.url))

/** 生成一段带谐波的浊音（比纯正弦更接近真实嗓音，自相关也更稳）。 */
function voiced(seconds, f0Start, f0End, ampStart, ampEnd, tremoloHz = 4) {
  const n = Math.round(seconds * SR)
  const out = new Float32Array(n)
  let phase = 0
  for (let i = 0; i < n; i++) {
    const t = i / SR
    const f0 = f0Start + (f0End - f0Start) * (t / seconds)
    phase += (2 * Math.PI * f0) / SR
    let s = 0
    for (let h = 1; h <= 5; h++) s += Math.sin(phase * h) / h
    s /= 1.6
    // 15ms 淡入淡出，避免爆音
    const ramp = Math.min(1, t / 0.015, (seconds - t) / 0.015)
    const amp = (ampStart + (ampEnd - ampStart) * (t / seconds)) * Math.max(0, ramp)
    const trem = 0.75 + 0.25 * Math.sin(2 * Math.PI * tremoloHz * t) // 模拟音节起伏
    out[i] = s * amp * trem + (Math.random() - 0.5) * 0.004 // 一点底噪
  }
  return out
}

const silence = (seconds) => new Float32Array(Math.round(seconds * SR))

function concat(chunks) {
  const total = chunks.reduce((s, c) => s + c.length, 0)
  const out = new Float32Array(total)
  let o = 0
  for (const c of chunks) {
    out.set(c, o)
    o += c.length
  }
  return out
}

function encodeWav(samples, sampleRate) {
  const n = samples.length
  const buf = Buffer.alloc(44 + n * 2)
  buf.write('RIFF', 0)
  buf.writeUInt32LE(36 + n * 2, 4)
  buf.write('WAVE', 8)
  buf.write('fmt ', 12)
  buf.writeUInt32LE(16, 16)
  buf.writeUInt16LE(1, 20) // PCM
  buf.writeUInt16LE(1, 22) // mono
  buf.writeUInt32LE(sampleRate, 24)
  buf.writeUInt32LE(sampleRate * 2, 28)
  buf.writeUInt16LE(2, 32)
  buf.writeUInt16LE(16, 34)
  buf.write('data', 36)
  buf.writeUInt32LE(n * 2, 40)
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]))
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2)
  }
  return buf
}

// ── 构造：已知结构 ─────────────────────────────────────────────────────
const A = voiced(0.9, 170, 172, 0.3, 0.3) // 平稳
const B = voiced(1.1, 170, 260, 0.34, 0.34) // 上扬
const C = voiced(0.9, 250, 140, 0.34, 0.08) // 下沉 + 句尾变弱
const samples = concat([silence(0.3), A, silence(0.38), B, silence(0.2), C, silence(0.2)])
const expectedDuration = samples.length / SR

// ── 全程内存：合成 → 编码成真实 WAV 字节 → 解码回来 ────────────────────
// 不落盘。一是不给自检引入 I/O 副作用，二是本环境里子进程写工作区会 ENOENT。
const wavBytes = encodeWav(samples, SR)
const decoded = decodeWav(new Uint8Array(wavBytes))
const report = analyze(decoded.samples, decoded.sampleRate)

let failed = 0
const check = (cond, label, extra = '') => {
  console.log(`${cond ? '  ✓' : '  ✗'} ${label}${extra ? ` — ${extra}` : ''}`)
  if (!cond) failed++
}

console.log('fixture: 内存合成（不落盘）· 16000Hz mono 16bit PCM')
console.log(`期望时长 ${expectedDuration.toFixed(2)}s / 解码得到 ${decoded.durationSec.toFixed(2)}s`)
console.log('')

console.log('[1] 容器与时长')
check(Math.abs(decoded.durationSec - expectedDuration) < 0.02, 'WAV 解码时长与构造一致')
check(report.ok === true, '分析器返回 ok')

console.log('[2] 说话段与停顿（构造值：3 段 / 2 处停顿 / 最长 380ms）')
check(report.speech?.segmentCount === 3, '检测到 3 段说话', `实际 ${report.speech?.segmentCount}`)
check(report.pause?.count === 2, '检测到 2 处停顿', `实际 ${report.pause?.count} · ${JSON.stringify(report.pause?.durationsMs)}`)
check(
  Math.abs((report.pause?.longestMs ?? 0) - 380) <= 80,
  '最长停顿 ≈380ms',
  `实际 ${report.pause?.longestMs}ms`,
)

console.log('[3] 音高走向（构造值：第 2 段上扬、第 3 段下沉）')
const contours = report.pitch?.contours ?? []
check(contours[0]?.contour === '平稳', '第 1 段判为平稳', `实际 ${contours[0]?.contour} (斜率 ${contours[0]?.slopeHzPerSec})`)
check(contours[1]?.contour === '上扬', '第 2 段判为上扬', `实际 ${contours[1]?.contour} (斜率 ${contours[1]?.slopeHzPerSec})`)
check(contours[2]?.contour === '下沉', '第 3 段判为下沉', `实际 ${contours[2]?.contour} (斜率 ${contours[2]?.slopeHzPerSec})`)
check(
  (report.pitch?.meanHz ?? 0) > 140 && (report.pitch?.meanHz ?? 0) < 270,
  '基频均值落在构造区间内',
  `实际 ${report.pitch?.meanHz}Hz`,
)
check(report.pitch?.ending === '句尾下沉', '末段句尾判为下沉', `实际 ${report.pitch?.ending}`)

console.log('[4] 力度与节奏')
check(report.energy?.endingLabel === '句尾明显变弱', '末段力度收尾判为变弱', `实际 ${report.energy?.endingLabel}`)
check((report.tempo?.syllablesPerSec ?? 0) > 1, '节奏估出音节率 >1/s', `实际 ${report.tempo?.syllablesPerSec}/s（${report.tempo?.label}）`)

console.log('[5] 纪律：声学层不转写')
check(!('words' in report) || report.words === undefined, '声学层报告里没有 words 字段')
check(report.uncertainty?.some((u) => u.includes('不做转写')), '明确声明「不做转写」')
check(report.uncertainty?.some((u) => u.includes('valence')), '明确声明不猜正负情绪')
check(report.impression?.valence === '无法判断（不猜）', 'valence 输出为「无法判断」')
check(report.impression?.confidence?.startsWith('低'), '印象标签标注为低置信')
check(Array.isArray(report.impression?.candidates) && report.impression.candidates.length > 0, '给出候选听感词')

console.log('[6] 边界：静音不装听见，连读不误判成静音，过短直说太短')
const quiet = analyze(new Float32Array(SR * 2), SR)
check(quiet.ok === false && quiet.reason === 'no_speech', '纯静音输入被拒，不假装听见', `实际 ${quiet.reason}`)

// 反向边界：阈值被钳在可听下限后，一段"全程连续说话"不能被判成整段静音
const nonstop = analyze(voiced(2.0, 190, 195, 0.3, 0.3), SR)
check(
  nonstop.ok === true && nonstop.speech?.segmentCount === 1,
  '连续说话判为 1 段（阈值钳位没有造成反向误判）',
  `实际 ok=${nonstop.ok} segments=${nonstop.speech?.segmentCount ?? '–'}${nonstop.reason ? ' reason=' + nonstop.reason : ''}`,
)

const tiny = analyze(new Float32Array(64), SR)
check(tiny.ok === false && tiny.reason === 'too_short', '过短音频直说太短', `实际 ${tiny.reason}`)

console.log('')
console.log('—— 实际读数 ——')
console.log(JSON.stringify({
  delivery: report.delivery,
  speech: report.speech,
  pause: report.pause,
  tempo: report.tempo,
  pitch: { meanHz: report.pitch?.meanHz, stdHz: report.pitch?.stdHz, contours: report.pitch?.contours, ending: report.pitch?.ending },
  energy: report.energy,
  impression: report.impression,
}, null, 2))

console.log('')
console.log(failed === 0 ? '全部通过 ✓' : `有 ${failed} 项未通过 ✗`)
process.exit(failed === 0 ? 0 : 1)

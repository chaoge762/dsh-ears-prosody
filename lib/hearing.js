/**
 * 声学层：从波形里读出「怎么说的」，**不做任何转写**。
 *
 * 这是路线 C 的地基。日常的语音转文字丢掉了说话方式——语速、停顿结构、
 * 音高起伏、句尾走向、力度变化——而这些恰恰是"我在听你说话"和"我读了一段
 * 文字"的区别。这些量全部可以从波形直接算出来，不需要模型、不需要联网、
 * 不需要密钥。
 *
 * 纪律：
 *  · 只报告测到的数字与由数字推得的**低置信**印象，绝不编造内容；
 *  · 正面/负面情绪（valence）不从声学硬猜 —— 明确写进 uncertainty；
 *  · 语料是"某人说话"，检测不到人声就直说是静音/音乐/噪声，不假装听见了。
 */

const FRAME_MS = 25
const HOP_MS = 10
const MIN_SEG_MS = 100 // 短于此的响声不算一段说话
const MERGE_GAP_MS = 120 // 短于此的静音不切断同一句
const F0_MIN_HZ = 70
const F0_MAX_HZ = 400
const SILENCE_DB = -58
const DIGITAL_SILENCE_DB = -100 // 低于此值视为"数字静音"（纯 0 采样算出 -180dB）

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v)
const round = (v, d = 2) => (v === null || !Number.isFinite(v) ? v : Number(v.toFixed(d)))
const mean = (a) => (a.length === 0 ? null : a.reduce((x, y) => x + y, 0) / a.length)
const std = (a) => {
  if (a.length < 2) return null
  const m = mean(a)
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1))
}

function percentile(sorted, p) {
  if (sorted.length === 0) return null
  const i = clamp(Math.round((sorted.length - 1) * p), 0, sorted.length - 1)
  return sorted[i]
}

/** 自相关基频估计（含抛物线插值）。不成器就返回 null，不硬凑一个数。 */
function estimateF0(frame, sampleRate) {
  const n = frame.length
  let energy = 0
  for (let i = 0; i < n; i++) energy += frame[i] * frame[i]
  if (energy < 1e-9) return null

  const minLag = Math.max(2, Math.floor(sampleRate / F0_MAX_HZ))
  const maxLag = Math.min(n - 2, Math.floor(sampleRate / F0_MIN_HZ))
  if (maxLag <= minLag + 1) return null

  let bestLag = -1
  let bestVal = 0
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0
    for (let i = 0; i < n - lag; i++) sum += frame[i] * frame[i + lag]
    const norm = sum / energy
    if (norm > bestVal) {
      bestVal = norm
      bestLag = lag
    }
  }
  // 归一化自相关低于此值 → 当成非浊音（清音/摩擦音/噪声），宁缺勿滥
  if (bestLag < 0 || bestVal < 0.35) return null

  // 抛物线插值，把整数 lag 精修到亚采样级
  const y0 = bestVal
  let refined = bestLag
  if (bestLag > minLag && bestLag < maxLag) {
    let prev = 0
    let next = 0
    for (let i = 0; i < n - (bestLag - 1); i++) prev += frame[i] * frame[i + bestLag - 1]
    for (let i = 0; i < n - (bestLag + 1); i++) next += frame[i] * frame[i + bestLag + 1]
    prev /= energy
    next /= energy
    const denom = 2 * (2 * y0 - prev - next)
    if (denom !== 0) {
      const shift = (next - prev) / denom
      if (Math.abs(shift) < 1) refined = bestLag + shift
    }
  }
  const f0 = sampleRate / refined
  if (f0 < F0_MIN_HZ || f0 > F0_MAX_HZ) return null
  return f0
}

/** 最小二乘斜率（每秒多少 Hz），用来判断句内音高走向。 */
function f0Slope(samples) {
  const n = samples.length
  if (n < 6) return null
  const mx = mean(samples.map((s) => s.t))
  const my = mean(samples.map((s) => s.f0))
  let num = 0
  let den = 0
  for (const s of samples) {
    num += (s.t - mx) * (s.f0 - my)
    den += (s.t - mx) ** 2
  }
  if (den === 0) return null
  return num / den
}

/** dB 包络的最小二乘斜率（dB/秒）—— 用来判断句尾是在变弱还是撑住。 */
function envelopeSlopeDbPerSec(frames, hopMs) {
  const n = frames.length
  if (n < 4) return null
  const step = hopMs / 1000
  const mx = ((n - 1) * step) / 2
  const my = mean(frames)
  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    const x = i * step - mx
    num += x * (frames[i] - my)
    den += x * x
  }
  return den === 0 ? null : num / den
}

function contourLabel(slope) {
  if (slope === null) return '样本不足'
  if (slope > 8) return '上扬'
  if (slope < -8) return '下沉'
  return '平稳'
}

function tempoLabel(rate) {
  if (rate === null) return '无法判断'
  if (rate < 3.2) return '偏慢'
  if (rate > 4.8) return '偏快'
  return '中等'
}

/**
 * @param {Float32Array} samples 单声道样本
 * @param {number} sampleRate 采样率
 */
export function analyze(samples, sampleRate) {
  const frameLen = Math.max(64, Math.round((sampleRate * FRAME_MS) / 1000))
  const hop = Math.max(16, Math.round((sampleRate * HOP_MS) / 1000))
  const frameCount = samples.length >= frameLen ? 1 + Math.floor((samples.length - frameLen) / hop) : 0

  const uncertainty = []
  if (frameCount < 5) {
    return {
      ok: false,
      reason: 'too_short',
      detail: `有效帧只有 ${frameCount} 个，音频太短（不足 ${FRAME_MS * 2}ms）`,
      uncertainty: ['音频过短，任何声学特征都不可靠'],
    }
  }

  // ── 逐帧能量与过零率 ────────────────────────────────────────────────
  const db = new Float64Array(frameCount)
  const zcr = new Float64Array(frameCount)
  for (let f = 0; f < frameCount; f++) {
    const start = f * hop
    let sum = 0
    let crossings = 0
    let prevNeg = samples[start] < 0
    const frame = samples.subarray(start, start + frameLen)
    for (let i = 0; i < frame.length; i++) {
      const s = frame[i]
      sum += s * s
      const neg = s < 0
      if (neg !== prevNeg) crossings++
      prevNeg = neg
    }
    db[f] = 20 * Math.log10(Math.sqrt(sum / frameLen) + 1e-9)
    zcr[f] = crossings / frameLen
  }

  const sortedDb = Float64Array.from(db).sort()
  const peakDb = percentile(sortedDb, 0.95)

  // 绝对闸门：整段连可听电平都够不着，就是没有声音，不装听见。
  if (!(peakDb > SILENCE_DB)) {
    return {
      ok: false,
      reason: 'no_speech',
      detail: `整段峰值 ${round(peakDb, 1)}dB 低于可听阈值 ${SILENCE_DB}dB：近乎数字静音`,
      durationSec: round(samples.length / sampleRate, 2),
      uncertainty: ['整段静音 —— 本工具不会假装听见了内容'],
    }
  }

  // ── 两遍法门限 ──────────────────────────────────────────────────────
  // 第一版的错误：拿"可听帧"的 p10 当噪声底。当停顿是**数字静音**时（录屏、
  // 后期静音、浏览器录音的静音段都是）这些帧被排除在"可听"之外，剩下的全
  // 是语音，于是 p10 等于"最弱的语音"，噪声底被高估 → 阈值顶到语音之上 →
  // 衰减的句尾被整段切掉（自检抓到的第三个真问题：rangeDb 只剩 6dB）。
  //
  // 正确做法：先用一个粗门限找出"明显不是说话"的帧，用它们估噪声底；
  // 没有这类帧（例如全程连续说话）时，退回纯相对门限。
  const roughGateDb = Math.max(peakDb - 18, SILENCE_DB + 6)
  const belowGate = sortedDb.filter((v) => v > DIGITAL_SILENCE_DB && v <= roughGateDb)
  const noiseFloorDb = belowGate.length >= 8 ? percentile(belowGate, 0.9) : null
  const thresholdDb =
    noiseFloorDb === null
      ? Math.max(peakDb - 25, SILENCE_DB)
      : clamp(noiseFloorDb + 8, SILENCE_DB, peakDb - 6)

  // ── 切出说话段（合并短间隙、过滤毛刺）──────────────────────────────
  const raw = []
  for (let f = 0; f < frameCount; f++) {
    if (db[f] > thresholdDb) raw.push([f, f])
  }
  const merged = []
  for (const seg of raw) {
    const last = merged[merged.length - 1]
    if (last && (seg[0] - last[1] - 1) * HOP_MS < MERGE_GAP_MS) last[1] = seg[1]
    else merged.push([seg[0], seg[1]])
  }
  const segments = merged.filter(([a, b]) => (b - a + 1) * HOP_MS >= MIN_SEG_MS)

  const durationSec = samples.length / sampleRate
  if (segments.length === 0) {
    return {
      ok: false,
      reason: 'no_speech',
      detail: '没有检测到说话段：整段更接近静音、音乐或稳态噪声',
      durationSec: round(durationSec, 2),
      thresholdDb: round(thresholdDb, 1),
      uncertainty: ['检测不到人声 —— 本工具不会假装听见了内容'],
    }
  }

  // ── 停顿结构 ────────────────────────────────────────────────────────
  const pauses = []
  for (let i = 1; i < segments.length; i++) {
    pauses.push((segments[i][0] - segments[i - 1][1] - 1) * HOP_MS)
  }
  const speechMs = segments.reduce((s, [a, b]) => s + (b - a + 1) * HOP_MS, 0)
  const speechSec = speechMs / 1000
  const longestPauseMs = pauses.length ? Math.max(...pauses) : 0

  // ── 音高（只对浊音帧估 F0）─────────────────────────────────────────
  const f0Series = []
  for (const [a, b] of segments) {
    for (let f = a; f <= b; f++) {
      if (zcr[f] > 0.3) continue // 清音/摩擦音不估音高
      const f0 = estimateF0(samples.subarray(f * hop, f * hop + frameLen), sampleRate)
      if (f0 !== null) f0Series.push({ t: (f * hop) / sampleRate, f0, seg: segments.indexOf([a, b]) })
    }
  }
  const f0Values = f0Series.map((s) => s.f0)
  const f0Sorted = [...f0Values].sort((x, y) => x - y)
  const f0Mean = mean(f0Values)
  const f0Std = std(f0Values)

  if (f0Values.length < 10) {
    uncertainty.push('浊音帧太少，音高类指标（起伏/走向）不可靠')
  }

  // ── 逐段音高走向：整段斜率 + 末段"句尾" ─────────────────────────────
  const segContours = segments.map(([a, b], idx) => {
    const pts = f0Series.filter((s) => s.t * sampleRate >= a * hop && s.t * sampleRate <= (b + 1) * hop)
    const slope = f0Slope(pts)
    return { index: idx + 1, durationMs: (b - a + 1) * HOP_MS, slopeHzPerSec: round(slope, 1), contour: contourLabel(slope) }
  })
  const firstContour = segContours[0]?.contour ?? '样本不足'

  const lastSeg = segments[segments.length - 1]
  const lastPts = f0Series.filter((s) => s.t * sampleRate >= lastSeg[0] * hop && s.t * sampleRate <= (lastSeg[1] + 1) * hop)
  let endingPitch = '样本不足'
  if (lastPts.length >= 6) {
    const half = Math.floor(lastPts.length / 2)
    const head = mean(lastPts.slice(0, half).map((s) => s.f0))
    const tail = mean(lastPts.slice(half).map((s) => s.f0))
    const delta = tail - head
    endingPitch = delta > 6 ? '句尾上扬' : delta < -6 ? '句尾下沉' : '句尾平稳'
  }

  // ── 力度变化与收尾 ──────────────────────────────────────────────────
  const segDb = []
  for (const [a, b] of segments) for (let f = a; f <= b; f++) segDb.push(db[f])
  const segSorted = [...segDb].sort((x, y) => x - y)
  const rangeDb = (percentile(segSorted, 0.95) ?? 0) - (percentile(segSorted, 0.1) ?? 0)
  const lastSegFrames = []
  for (let f = lastSeg[0]; f <= lastSeg[1]; f++) lastSegFrames.push(db[f])
  const tailFrames = lastSegFrames.slice(-Math.max(4, Math.round(300 / HOP_MS)))
  // 用 dB 包络的**斜率**判定收尾，而不是"段末均值 − 整段均值"：
  // VAD 恰好会在语音衰减到阈值以下时切断，段末那段最弱的尾音落在段外，
  // 差值法必然把"变弱"算小（自检抓到的第二个真问题）。斜率只看段内可见的
  // 下降趋势，与阈值切在哪里无关。
  const endingSlope = envelopeSlopeDbPerSec(tailFrames, HOP_MS)
  const endingEnergy =
    endingSlope === null
      ? '收尾样本不足'
      : endingSlope < -10
        ? '句尾明显变弱'
        : endingSlope > 8
          ? '句尾加强'
          : '句尾力度平稳'

  // ── 节奏（音节率代理）：浊音段内的能量峰 ────────────────────────────
  const smooth = new Float64Array(db.length)
  for (let f = 0; f < db.length; f++) {
    const a = Math.max(0, f - 1)
    const b = Math.min(db.length - 1, f + 1)
    let s = 0
    for (let i = a; i <= b; i++) s += db[i]
    smooth[f] = s / (b - a + 1)
  }
  const inSpeech = new Uint8Array(db.length)
  for (const [a, b] of segments) for (let f = a; f <= b; f++) inSpeech[f] = 1
  let peaks = 0
  let lastPeakFrame = -Infinity
  for (let f = 1; f < db.length - 1; f++) {
    if (!inSpeech[f]) continue
    if (smooth[f] < thresholdDb + 3) continue
    if (smooth[f] >= smooth[f - 1] && smooth[f] >= smooth[f + 1] && (f - lastPeakFrame) * HOP_MS >= 60) {
      peaks++
      lastPeakFrame = f
    }
  }
  const syllableRate = speechSec > 0.3 ? peaks / speechSec : null

  // ── 印象分：三个可测量的维度加权，全部标注为启发式 ──────────────────
  const rateScore = syllableRate === null ? 0 : clamp((syllableRate - 4.2) / 1.6, -1, 1)
  const pitchScore = f0Std === null ? 0 : clamp((f0Std - 24) / 18, -1, 1)
  const energyScore = clamp((rangeDb - 20) / 16, -1, 1)
  const pauseScore = clamp((3 - pauses.length) / 2, -1, 1)
  const arousal = rateScore * 0.35 + pitchScore * 0.3 + energyScore * 0.2 + pauseScore * 0.15

  let candidates
  if (arousal > 0.3) candidates = ['急促', '兴奋', '激动', '催促']
  else if (arousal < -0.3 && pauses.length >= 2) candidates = ['平缓', '疲惫', '迟疑', '低落']
  else if (arousal < -0.3) candidates = ['平静', '克制', '铺垫']
  else candidates = ['平稳陈述']

  uncertainty.push('声学层不做转写：这里没有一个字的内容，只有「怎么说」')
  uncertainty.push('正面/负面情绪（valence）无法仅凭声学可靠判断，本工具不猜')
  uncertainty.push('以上印象标签是启发式规则，不是模型判断，置信度低')
  if (durationSec < 1.5) uncertainty.push('音频短于 1.5 秒，节奏与停顿指标不稳')

  const delivery = [
    `语速${tempoLabel(syllableRate)}`,
    f0Std === null ? '音高起伏未知' : f0Std > 35 ? '音高起伏大' : f0Std < 18 ? '音高变化平' : '音高起伏中等',
    `${endingPitch}`,
    `${endingEnergy}`,
    pauses.length === 0 ? '句间几乎没有停顿' : `有 ${pauses.length} 处停顿${longestPauseMs >= 600 ? '（含明显长停顿）' : ''}`,
  ].join('、')

  return {
    ok: true,
    durationSec: round(durationSec, 2),
    speech: {
      speechSec: round(speechSec, 2),
      speechRatio: round(speechMs / 1000 / Math.max(durationSec, 1e-6), 3),
      segmentCount: segments.length,
      segmentDurationsMs: segContours.map((s) => s.durationMs),
    },
    pause: {
      count: pauses.length,
      durationsMs: pauses.map((p) => Math.round(p)),
      longestMs: Math.round(longestPauseMs),
      meanMs: pauses.length ? Math.round(mean(pauses)) : 0,
      leadingSilenceMs: Math.round(segments[0][0] * HOP_MS),
    },
    tempo: {
      syllablesPerSec: round(syllableRate, 2),
      energyPeaks: peaks,
      label: tempoLabel(syllableRate),
    },
    pitch: {
      meanHz: round(f0Mean, 1),
      stdHz: round(f0Std, 1),
      p10Hz: round(percentile(f0Sorted, 0.1), 1),
      p90Hz: round(percentile(f0Sorted, 0.9), 1),
      voicedFrames: f0Values.length,
      contours: segContours,
      firstContour,
      ending: endingPitch,
    },
    energy: {
      thresholdDb: round(thresholdDb, 1),
      noiseFloorDb: round(noiseFloorDb, 1),
      rangeDb: round(rangeDb, 1),
      endingLabel: endingEnergy,
      endingSlopeDbPerSec: round(endingSlope, 1),
    },
    voiceQuality: {
      zcrMean: round(mean(Array.from(zcr)), 3),
    },
    delivery,
    impression: {
      arousal: arousal > 0.3 ? '偏高' : arousal < -0.3 ? '偏低' : '中等',
      arousalScore: round(arousal, 2),
      valence: '无法判断（不猜）',
      candidates,
      confidence: '低（启发式）',
    },
    uncertainty,
  }
}

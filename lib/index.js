/**
 * dsh-ears —— 给 DeepSeek Harness 加一双耳朵。
 *
 * 存在理由：DSH 的内容块只有 text / reasoning / image / file / tool-*，
 * 模型模态只有 text / image（`ContentBlockMap` 与 `ModelModalityMap`），
 * 音频字节进不了模型上下文；而官方的语音链路是 speechToText → Transcript，
 * 也就是**转写**——它丢掉了说话方式。
 *
 * 本插件走旁路：音频 → ① 本地声学层（零依赖、离线、不转写，给出语速/停顿/
 * 音高/句尾/力度的量化描述）→ ② 可选的听音模型层（真的听过，补上语义与情绪）
 * → 一份结构化文本，交给主模型。
 *
 * 纪律：
 *  · 不假装听见内容 —— 声学层 words 是空字符串，且把这一点写进 uncertainty；
 *  · 不猜 valence（正负情绪），明确标注无法判断；
 *  · 同一份音频同一关注点只算一次（缓存），不重复烧算力。
 */
import { readFileSync, statSync } from 'node:fs'
import { isAbsolute, resolve } from 'node:path'
import { decodeWav } from './wav.js'
import { analyze } from './hearing.js'
import { listenWithModel } from './model.js'

export const name = 'dsh-ears-prosody'
export const inject = ['tools']

const MAX_BYTES_DEFAULT = 12 * 1024 * 1024
const LOCAL_TIMEOUT_MS = 60_000

const OUTPUT_SCHEMA = {
  type: 'object',
  required: ['ok'],
  properties: {
    ok: { type: 'boolean' },
    source: { type: 'string' },
    audio: { type: 'object', additionalProperties: true },
    hearing: { type: 'object', additionalProperties: true },
    words: { type: 'string' },
    modelLayer: { type: 'object', additionalProperties: true },
    uncertainty: { type: 'array', items: { type: 'string' } },
  },
  additionalProperties: true,
}

function tag(bytes, o, len) {
  let s = ''
  for (let i = 0; i < len; i++) s += String.fromCharCode(bytes[o + i])
  return s
}

/** 从魔数认容器 —— 不看扩展名，扩展名会说谎。 */
export function sniffContainer(bytes) {
  if (bytes.length >= 12 && tag(bytes, 0, 4) === 'RIFF' && tag(bytes, 8, 4) === 'WAVE') return 'wav'
  if (bytes.length >= 4 && bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3) return 'webm'
  if (bytes.length >= 3 && tag(bytes, 0, 3) === 'ID3') return 'mp3'
  if (bytes.length >= 2 && bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0) return 'mp3'
  if (bytes.length >= 4 && tag(bytes, 0, 4) === 'OggS') return 'ogg'
  if (bytes.length >= 12 && tag(bytes, 4, 4) === 'ftyp') return 'm4a'
  return 'unknown'
}

function normalizeConfig(config) {
  const model = config?.model
  let normalizedModel = null
  if (model && typeof model === 'object' && model.endpoint && model.model) {
    const apiKeyEnv = typeof model.apiKeyEnv === 'string' ? model.apiKeyEnv : 'EARS_API_KEY'
    normalizedModel = {
      endpoint: String(model.endpoint),
      model: String(model.model),
      // 密钥不在这里解析：凭据服务的契约要求「每次操作重新解析、不跨操作缓存」，
      // 所以交给 resolveApiKey 在 execute 里读 —— 用户换了 key 不必重启插件。
      apiKey: typeof model.apiKey === 'string' ? model.apiKey : '',
      apiKeyEnv,
      timeoutMs: Number.isFinite(model.timeoutMs) ? model.timeoutMs : 120_000,
      jsonMode: model.jsonMode !== false,
      headers: model.headers && typeof model.headers === 'object' ? model.headers : {},
    }
  }
  return {
    toolName: typeof config?.toolName === 'string' && config.toolName ? config.toolName : 'listen_audio',
    mode: ['auto', 'local', 'model'].includes(config?.mode) ? config.mode : 'auto',
    address: resolveAddress(config?.address),
    maxBytes: Number.isFinite(config?.maxBytes) ? config.maxBytes : MAX_BYTES_DEFAULT,
    model: normalizedModel,
  }
}

/** 怎么称呼使用插件的人。配置优先，其次环境变量，最后兜底「老大」。 */
const DEFAULT_ADDRESS = '老大'
function resolveAddress(value) {
  if (typeof value === 'string' && value.trim()) return value.trim()
  const fromEnv = process.env.EARS_ADDRESS
  if (typeof fromEnv === 'string' && fromEnv.trim()) return fromEnv.trim()
  return DEFAULT_ADDRESS
}

/**
 * 解析听音模型的密钥，三档从具体到通用：
 *   1. model.apiKey —— 直接写在配置里（最不推荐：密钥会留在配置文件）
 *   2. DSH 凭据服务 —— 按「环境变量名」解析，它本身已经把进程环境变量、
 *      托管存储与 .env 三层叠好了，所以这是最省事也最安全的一档。
 *      契约要求每次操作重新解析、不跨操作缓存，这里遵守。
 *   3. 同名环境变量兜底 —— 凭据服务不可用（换个组合跑）时仍能工作。
 */
async function resolveApiKey(ctx, modelCfg) {
  if (modelCfg.apiKey) return modelCfg.apiKey
  const credentials = typeof ctx.get === 'function' ? ctx.get('credentials') : undefined
  if (credentials && typeof credentials.resolve === 'function') {
    try {
      const resolved = await credentials.resolve(modelCfg.apiKeyEnv)
      if (resolved?.value) return resolved.value
    } catch (error) {
      console.error(`[dsh-ears] 凭据解析失败，退回环境变量：${error}`)
    }
  }
  return process.env[modelCfg.apiKeyEnv] ?? ''
}

function sourceKeyOf(path) {
  if (/^https?:\/\//i.test(path)) return `remote:${path}`
  try {
    const abs = isAbsolute(path) ? path : resolve(path)
    const st = statSync(abs)
    return `local:${abs}:${st.size}:${st.mtimeMs}`
  } catch {
    return `unreadable:${path}`
  }
}

async function loadBytes(source, maxBytes) {
  if (/^https?:\/\//i.test(source)) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(new Error('下载音频超时')), 60_000)
    try {
      const res = await fetch(source, { signal: controller.signal })
      if (!res.ok) throw new Error(`下载音频失败：HTTP ${res.status}`)
      const buf = Buffer.from(await res.arrayBuffer())
      if (buf.length > maxBytes) throw new Error(`音频 ${buf.length} 字节，超过上限 ${maxBytes}`)
      return buf
    } finally {
      clearTimeout(timer)
    }
  }
  const abs = isAbsolute(source) ? source : resolve(source)
  const buf = readFileSync(abs)
  if (buf.length > maxBytes) throw new Error(`音频 ${buf.length} 字节，超过上限 ${maxBytes}`)
  return buf
}

function renderReport(value) {
  if (!value?.ok) {
    return `听音失败：${value?.reason ?? '未知原因'}${value?.detail ? ` —— ${value.detail}` : ''}`
  }
  const lines = []
  const audio = value.audio ?? {}
  lines.push(`音频：${audio.container ?? '?'} · ${audio.durationSec ?? '?'}s · ${audio.bytes ?? '?'}B`)
  const h = value.hearing
  if (h?.ok) {
    lines.push(`说话方式：${h.delivery}`)
    lines.push(
      `说话段 ${h.speech.segmentCount} 段 / 有效语音 ${h.speech.speechSec}s（占比 ${h.speech.speechRatio}）` +
        `；停顿 ${h.pause.count} 处（最长 ${h.pause.longestMs}ms）`,
    )
    lines.push(
      `音高：均值 ${h.pitch.meanHz}Hz，起伏 σ=${h.pitch.stdHz}Hz，首段走向 ${h.pitch.firstContour}，${h.pitch.ending}`,
    )
    lines.push(`力度：动态范围 ${h.energy.rangeDb}dB，${h.energy.endingLabel}`)
    lines.push(`听感印象（启发式·低置信）：唤醒度${h.impression.arousal}，候选 ${h.impression.candidates.join('、')}`)
  } else if (h) {
    lines.push(`本地声学层：${h.reason ?? ''} ${h.detail ?? ''}`)
  }
  const m = value.modelLayer
  if (m && m.reason !== 'not_requested') {
    if (m.ok) {
      if (m.words) lines.push(`听到的话：${m.words}`)
      else lines.push('听到的话：（听音模型未给出确定转写）')
      if (m.delivery) lines.push(`听音模型的说话方式：${m.delivery}`)
      if (m.emotion?.label) lines.push(`情绪：${m.emotion.label}（${m.emotion.intensity ?? '强度未知'}）—— ${m.emotion.evidence ?? ''}`)
      if (Array.isArray(m.emphasis) && m.emphasis.length) lines.push(`重读：${m.emphasis.join('、')}`)
      if (m.background) lines.push(`背景音：${m.background}`)
      if (m.notes) lines.push(`其它线索：${m.notes}`)
    } else {
      lines.push(`听音模型未按 JSON 作答，原文如下：\n${m.raw ?? ''}`)
    }
  } else {
    lines.push('未配置听音模型：本次只有声学层结论，不含任何语义内容。')
  }
  if (Array.isArray(value.uncertainty) && value.uncertainty.length) {
    lines.push(`不确定：${value.uncertainty.join('；')}`)
  }
  return lines.join('\n')
}

export function apply(ctx, config = {}) {
  const cfg = normalizeConfig(config)
  const cache = new Map()

  const tool = {
    name: cfg.toolName,
    description:
      '通过 dsh-ears 听取一段音频，回答「它是怎么说的」而不是「说了什么」。' +
      '本地声学层零依赖离线运行，给出语速、停顿结构、音高起伏与句尾走向、力度变化，' +
      '并明确标注启发式与不确定项；它**不做转写**，没有配置听音模型时 words 是空字符串。' +
      '当消息里出现语音、录音、会议片段或任何音频路径/URL 时使用本工具；' +
      '把返回的量化结论当作证据引用，不要替它编造内容。' +
      '支持 wav（内置解码）；其它容器需要配置听音模型层。' +
      '回复约定（用户明确要求，很重要）：先用一两句大白话把这段语音的听感说出来——' +
      '活泼、口语化、别念参数名、不要加「语气：」这类标签，也不要写成硬邦邦的分析报告；' +
      '然后再正常回答对方的内容。称呼用户为「' + cfg.address + '」（配置项 address 可改，环境变量 EARS_ADDRESS 亦可），' +
      '必要时可以直接用「你」。整体保持简短。',
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: '音频的绝对本地路径或 http(s) URL',
        },
        focus: {
          type: 'string',
          description: '可选：希望额外关注的点（例如「对方是不是在敷衍」「有没有背景音乐」）',
        },
        mode: {
          type: 'string',
          enum: ['auto', 'local', 'model'],
          description: 'auto=本地+模型（若已配置）；local=只要声学层；model=只用听音模型层',
        },
      },
      required: ['path'],
    },
    output: { schema: OUTPUT_SCHEMA, render: (_args, value) => [{ type: 'text', text: renderReport(value) }] },
    timeoutMs: LOCAL_TIMEOUT_MS + (cfg.model?.timeoutMs ?? 0) + 20_000,
    isConcurrencySafe: () => true,
    presentCall: (args) => ({
      card: 'generic',
      title: cfg.toolName,
      kind: 'read',
      rawInput: args,
      ...(typeof args?.path === 'string' && !/^https?:\/\//i.test(args.path)
        ? { locations: [{ path: args.path }] }
        : {}),
    }),

    async execute(args) {
      const source = args?.path
      if (typeof source !== 'string' || source.trim() === '') {
        throw new Error(`${cfg.toolName} 需要一个非空的字符串 path`)
      }
      const mode = args?.mode ?? cfg.mode
      const focus = typeof args?.focus === 'string' ? args.focus : ''
      const cacheKey = `${sourceKeyOf(source)}|${focus}|${mode}`

      const cached = cache.get(cacheKey)
      if (cached) return structuredClone(cached)

      const bytes = await loadBytes(source, cfg.maxBytes)
      const container = sniffContainer(bytes)
      const uncertainty = []

      // ① 本地声学层：只有 wav 能本地解（纯 JS，无依赖）
      let hearing = null
      let durationSec = null
      if (container === 'wav') {
        try {
          const decoded = decodeWav(bytes)
          durationSec = Number(decoded.durationSec.toFixed(2))
          hearing = analyze(decoded.samples, decoded.sampleRate)
          if (Array.isArray(hearing.uncertainty)) uncertainty.push(...hearing.uncertainty)
        } catch (error) {
          hearing = { ok: false, reason: 'decode_failed', detail: String(error?.message ?? error) }
          uncertainty.push(`WAV 解码失败：${error?.message ?? error}`)
        }
      } else {
        hearing = {
          ok: false,
          reason: 'container_not_locally_decodable',
          detail: `容器是 ${container}：本地声学层只会解 wav`,
        }
        uncertainty.push(
          `本地声学层无法解码 ${container}；要么先把音频转成 wav，要么配置听音模型层（model.endpoint / model.model）`,
        )
      }

      // ② 听音模型层：可选，需要端点、模型与密钥
      let modelLayer = null
      const wantsModel = mode === 'model' || (mode === 'auto' && cfg.model !== null)
      if (wantsModel) {
        if (!cfg.model) {
          if (mode === 'model') throw new Error('模式是 model，但没有配置 model.endpoint / model.model')
          uncertainty.push('未配置听音模型层')
        } else {
          const apiKey = await resolveApiKey(ctx, cfg.model)
          if (!apiKey) {
            uncertainty.push(
              `听音模型缺少密钥：在 DSH 凭据里配置 ${cfg.model.apiKeyEnv}（推荐）、或设同名环境变量、或写 model.apiKey`,
            )
          } else {
            if (!['wav', 'mp3'].includes(container)) {
              uncertainty.push(`听音模型层通常只接受 wav/mp3，本次容器是 ${container}，可能被服务端拒绝`)
            }
            modelLayer = await listenWithModel({ bytes, format: container, focus, ...cfg.model, apiKey })
          }
        }
      }

      const words =
        modelLayer?.ok && typeof modelLayer.words === 'string' && modelLayer.words.trim() !== ''
          ? modelLayer.words.trim()
          : null

      const value = {
        ok: true,
        source,
        audio: { container, bytes: bytes.length, durationSec },
        hearing,
        words: words ?? '',
        modelLayer: modelLayer ?? { ok: false, reason: 'not_requested' },
        uncertainty,
      }
      cache.set(cacheKey, value)
      return structuredClone(value)
    },
  }

  // 注册失败要"响亮地退化"，不能把整个插件树带崩：同名冲突、或宿主注册表
  // 形状变化时，本插件应当是"少了一个工具"，而不是"DSH 起不来"。
  try {
    ctx.tools.register(tool)
  } catch (error) {
    console.error(`[dsh-ears] ${cfg.toolName} 注册失败，插件已退化（不影响其它功能）：${error}`)
    return () => {}
  }
  console.log(
    `[dsh-ears] ${cfg.toolName} 已注册 · 本地声学层就绪` +
      (cfg.model ? ` · 听音模型层 ${cfg.model.model}` : ' · 未配置听音模型层（words 为空字符串）'),
  )
  return () => cache.clear()
}

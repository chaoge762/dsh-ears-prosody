/**
 * 听音模型层（可选）：把音频交给一个「会听」的模型，拿回结构化描述。
 *
 * 关键区别：这不是转写器。提示词明确要求它回答「怎么说的」——
 * 语速、停顿、重音、情绪、犹豫、背景音——转写只是其中一个字段，
 * 而且允许留空。这样语义与副语言信息都不会在 ASR 那一步丢掉。
 *
 * 只依赖全局 fetch（Node 18+），不引任何包。端点走 OpenAI 兼容的
 * chat/completions + input_audio 内容块；换成任何支持音频输入的服务商
 * 只需改 baseUrl / model。
 */

const INSTRUCTION = `你在替一个听不见声音的文本模型充当耳朵。你的任务不是逐字转写，而是回答「这个人是怎么说的」。

只报告你能从音频里真正得到的证据。听不清就写进 uncertainty，绝对不要编造内容或情绪。

输出一个 JSON 对象，字段如下：
{
  "words": "听到的话。听不清、或不确定，就留空字符串，不要猜",
  "delivery": "说话方式的自然语言描述：语速、停顿、音高起伏、句尾走向、力度变化",
  "emotion": { "label": "情绪词", "intensity": "低|中|高", "evidence": "支持这个判断的听觉证据" },
  "emphasis": ["被重读的词或短语"],
  "background": "背景音：有无音乐、噪声、多人、回声",
  "notes": "其它副语言线索：叹气、笑、口吃、犹豫、清喉咙、压低声音、突然加快等",
  "uncertainty": ["你不确定的地方"]
}`

function stripFence(text) {
  const t = text.trim()
  const m = t.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/)
  return m ? m[1] : t
}

/**
 * @param {object} o
 * @param {Uint8Array} o.bytes 音频字节
 * @param {string} o.format 容器格式（wav/mp3/...）
 * @param {string} [o.focus] 额外关注点
 * @param {string} o.endpoint 形如 https://host/v1
 * @param {string} o.model 模型 id
 * @param {string} o.apiKey
 * @param {number} [o.timeoutMs]
 * @param {boolean} [o.jsonMode]
 * @param {Record<string,string>} [o.headers]
 */
export async function listenWithModel(o) {
  const url = `${o.endpoint.replace(/\/+$/, '')}/chat/completions`
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('听音模型超时')), o.timeoutMs ?? 120_000)

  const body = {
    model: o.model,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: o.focus ? `${INSTRUCTION}\n\n额外关注：${o.focus}` : INSTRUCTION },
          { type: 'input_audio', input_audio: { data: Buffer.from(o.bytes).toString('base64'), format: o.format } },
        ],
      },
    ],
    max_tokens: 900,
  }
  if (o.jsonMode !== false) body.response_format = { type: 'json_object' }

  let res
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(o.apiKey ? { authorization: `Bearer ${o.apiKey}` } : {}),
        ...(o.headers ?? {}),
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
  } catch (error) {
    clearTimeout(timer)
    throw new Error(`听音模型请求失败：${error?.message ?? error}`)
  }
  clearTimeout(timer)

  const text = await res.text()
  if (!res.ok) {
    throw new Error(`听音模型返回 ${res.status}：${text.slice(0, 400)}`)
  }

  let content
  try {
    content = JSON.parse(text)?.choices?.[0]?.message?.content
  } catch {
    throw new Error(`听音模型返回的不是 JSON：${text.slice(0, 300)}`)
  }
  if (typeof content !== 'string') {
    throw new Error(`听音模型响应里没有 content：${text.slice(0, 300)}`)
  }

  try {
    const parsed = JSON.parse(stripFence(content))
    return { ok: true, ...parsed }
  } catch {
    // 模型没按 JSON 作答时，原样交回，让人/主模型自己看，而不是丢掉内容
    return { ok: false, reason: 'unparsable', raw: content.slice(0, 4000) }
  }
}

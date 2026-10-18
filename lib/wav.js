/**
 * 最小 WAV 解码器 —— 只认 PCM / IEEE float，产出单声道 Float32。
 *
 * 为什么手写：这个插件的一条纪律是「能用 node 内置就不加依赖」。
 * 原生解码库（或 ffmpeg）会把这个插件变成需要构建授权的东西，
 * 而我们要它「装上就能用」。
 *
 * 支持：8/16/24/32 位整数 PCM、32/64 位浮点、WAVE_FORMAT_EXTENSIBLE 外壳、
 * 任意声道数（下混为单声道）、块按偶数字节对齐（RIFF 规范）。
 * 不认的容器（webm/opus、mp3、m4a）会抛出可读错误，由上层提示用户
 * 转成 wav，或交给「听音模型层」处理。
 */

const int24 = (bytes, o) => {
  const v = bytes[o] | (bytes[o + 1] << 8) | (bytes[o + 2] << 16)
  return (v & 0x800000) ? v - 0x1000000 : v
}

/**
 * @param {Uint8Array} bytes WAV 文件字节
 * @returns {{sampleRate:number, channels:number, durationSec:number, samples:Float32Array}}
 */
export function decodeWav(bytes) {
  const tag = (o) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3])

  if (bytes.byteLength < 44) throw new Error('WAV 太短：不足 44 字节，不可能是合法 RIFF 文件')
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
    throw new Error('不是 WAV：缺少 RIFF/WAVE 头（webm/opus、mp3、m4a 需要先转成 wav）')
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let fmt = null
  let dataOffset = -1
  let dataLength = 0

  let o = 12
  while (o + 8 <= bytes.byteLength) {
    const id = tag(o)
    const size = view.getUint32(o + 4, true)
    const body = o + 8
    if (id === 'fmt ' && size >= 16) {
      let audioFormat = view.getUint16(body, true)
      const numChannels = view.getUint16(body + 2, true)
      const sampleRate = view.getUint32(body + 4, true)
      const bitsPerSample = view.getUint16(body + 14, true)
      // WAVE_FORMAT_EXTENSIBLE：真正的格式码在 SubFormat GUID 的头两字节
      if (audioFormat === 0xfffe && size >= 40) {
        audioFormat = view.getUint16(body + 24, true)
      }
      fmt = { audioFormat, numChannels, sampleRate, bitsPerSample }
    } else if (id === 'data') {
      dataOffset = body
      dataLength = Math.min(size, bytes.byteLength - body)
    }
    o = body + size + (size % 2) // RIFF 块按偶数字节对齐
  }

  if (!fmt) throw new Error('WAV 缺少 fmt 块：无法得知采样率与位深')
  if (dataOffset < 0) throw new Error('WAV 缺少 data 块：没有音频数据')
  const { audioFormat, numChannels, sampleRate, bitsPerSample } = fmt
  if (!(numChannels >= 1) || !(sampleRate >= 4000)) {
    throw new Error(`WAV 参数不合理：${numChannels} 声道 / ${sampleRate} Hz`)
  }
  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(`不支持的 WAV 编码 audioFormat=${audioFormat}（只认 1=PCM、3=float）`)
  }

  const bytesPerSample = bitsPerSample >> 3
  if (bytesPerSample <= 0) throw new Error(`不支持的位深 ${bitsPerSample}`)
  const frameBytes = bytesPerSample * numChannels
  const frames = Math.floor(dataLength / frameBytes)
  if (frames <= 0) throw new Error('WAV 里没有采样帧')

  const samples = new Float32Array(frames)
  for (let i = 0; i < frames; i++) {
    let acc = 0
    for (let c = 0; c < numChannels; c++) {
      const p = dataOffset + i * frameBytes + c * bytesPerSample
      let v
      if (audioFormat === 3) {
        v = bitsPerSample === 64 ? view.getFloat64(p, true) : view.getFloat32(p, true)
      } else if (bitsPerSample === 8) {
        v = (bytes[p] - 128) / 128 // 8 位 PCM 是无符号的
      } else if (bitsPerSample === 16) {
        v = view.getInt16(p, true) / 32768
      } else if (bitsPerSample === 24) {
        v = int24(bytes, p) / 8388608
      } else {
        v = view.getInt32(p, true) / 2147483648
      }
      acc += v
    }
    samples[i] = acc / numChannels // 下混单声道
  }

  return {
    sampleRate,
    channels: numChannels,
    durationSec: frames / sampleRate,
    samples,
  }
}

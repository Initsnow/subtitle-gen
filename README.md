# subtitle-gen

高性能、纯 Rust 编写的端到端字幕生成工具。基于 GGUF 原生运行时、Qwen3 ASR 语音识别与强制对齐（Forced Aligner）、本地形态素分析与声学物理停顿断句，直接将音频/视频转录并排版为专业级字幕文件（SRT、VTT、LRC、JSON、TXT）。

同时支持给无时间戳歌词/文本进行精确对齐（Align），以及通过 LLM 进行可选的字幕校对与多语言翻译。

---

## 核心特性

- **纯 Rust 核心与单二进制交付**：无 Python、PyTorch 或外部 FFmpeg 进程依赖；内置 `symphonia` 解码多媒体容器（MP4、MKV、WebM、MP3、FLAC、WAV 等）与 `rubato` SIMD 16 kHz 重采样。
- **GGUF 全链路与硬件自适应**：原生集成 GGUF 模型，首选 CUDA 12 GPU 加速，在无独立显卡或驱动缺失时平滑自动降级至 Vulkan / Metal / CPU AVX2，杜绝崩溃。
- **本地智能分行与声学停顿吸附**：彻底告别昂贵且缓慢的 LLM 断句。集成 `lindera` 形态素词性标注（助词/连词/用言连用形），结合词间声学物理静音停顿（$\ge 150\text{ms}$）进行动态规划排版分句。
- **歌词/文本全局对齐（Needleman-Wunsch）**：提供精确二次 DP 与针对长篇文本的带状走廊 DP（Banded Corridor DP），支持片假名/平假名折叠匹配与单调时间戳平滑插值。
- **安全与可控缓存**：默认不产生磁盘缓存（纯内存流式处理）；可通过 `--cache` 按需开启基于 SHA-256 的持久化增量断点恢复与 LRU 自动清理。
- **开箱即用与交互体验**：提供 `indicatif` 多阶段动态进度条与多媒体批量处理能力。

---

## 安装与构建

### 1. 下载预编译二进制（推荐）

从 [GitHub Releases](https://github.com/CrispStrobe/subtitle-gen/releases) 下载适用于对应操作系统的单可执行文件：
- Windows: `subtitle-gen-x86_64-pc-windows-msvc.zip`
- Linux: `subtitle-gen-x86_64-unknown-linux-gnu.tar.gz` 或 `subtitle-gen-x86_64-unknown-linux-musl.tar.gz`

### 2. 从源码编译

要求 Rust 1.80+ (MSRV):

```console
# 克隆仓库
git clone https://github.com/CrispStrobe/subtitle-gen.git
cd subtitle-gen

# 编译 release 单二进制
cargo build --release -p subtitle-cli

# 二进制位于 target/release/subtitle-gen (Windows 下为 subtitle-gen.exe)
./target/release/subtitle-gen --help
```

---

## 快速上手

### 1. 基础转录

```console
# 自动识别并生成与视频同名的 SRT 字幕
subtitle-gen video.mp4

# 指定输出文件路径
subtitle-gen audio.wav --out output.srt
```

### 2. 批量处理与多格式输出

```console
# 批量处理所有视频，输出到指定目录并同时生成 SRT、WebVTT 与 JSON 格式
subtitle-gen *.mp4 --out-dir ./subtitles --format srt,vtt,json
```

### 3. 设备与性能控制

```console
# 强制使用 CPU 进行计算（0 层卸载至 GPU）
subtitle-gen audio.wav --gpu-layers 0

# 指定 CPU 线程数与量化等级
subtitle-gen audio.wav --n-threads 8 --quant q4_k

# 指定源语言提示
subtitle-gen audio.wav --language zh
```

### 4. 增量缓存（按需启用）

默认关闭磁盘缓存。在处理长音频或易中断任务时，可开启增量缓存以支持断点续跑：

```console
# 开启增量缓存
subtitle-gen lecture.mp4 --cache

# 指定缓存存储目录
subtitle-gen lecture.mp4 --cache --cache-dir ./my_cache

# 强制重新生成（覆盖已有缓存）
subtitle-gen lecture.mp4 --cache --overwrite-cache
```

---

## 进阶功能与子命令

### 1. 歌词与文本时间戳对齐 (`align`)

将没有时间轴的纯文本歌词或 LRC 与音频强制对齐，生成带精确时间戳的歌词或字幕：

```console
# 对齐歌词生成 LRC 文件
subtitle-gen align song.mp3 lyrics.txt --out song.lrc

# 对齐并去除行末标点符号
subtitle-gen align song.mp3 lyrics.txt --out song.lrc --strip-punctuation
```

### 2. LLM 字幕校对 (`proofread`)

使用大语言模型对已生成的 SRT 文件进行错别字、上下文一致性校对（需配置 API Key）：

```console
# 基于 config.toml 中配置的 LLM 进行校对
subtitle-gen proofread input.srt --out input.proofread.srt

# 覆盖指定模型
subtitle-gen proofread input.srt --llm-model deepseek-chat
```

### 3. LLM 字幕翻译 (`translate`)

将已有 SRT 字幕翻译为目标语言：

```console
subtitle-gen translate input.srt --out input.zh.srt --translate zh
```

---

## 配置文件 (`config.toml`)

程序按以下优先级探测并加载配置文件：
1. **命令行参数**：`--config <path>` 显式指定
2. **环境变量**：`SUBTITLE_GEN_CONFIG=<path>`
3. **当前工作目录**：`./config.toml`（项目级局部配置）
4. **系统级配置目录**：`config.toml`（用户全局通用配置）：
   - Windows: `%APPDATA%\subtitle-gen\config.toml`
   - Linux: `~/.config/subtitle-gen/config.toml`（遵循 `$XDG_CONFIG_HOME`）
   - macOS: `~/Library/Application Support/subtitle-gen/config.toml`
5. 若均未找到，则自动回退至内置默认配置。

可参考 [`config.example.toml`](config.example.toml) 进行自定义配置：

```toml
# 缓存根目录
cache_dir = ".subtitle-gen-cache"

[model]
asr_model = "qwen3-asr-0.6b-q4_k"
aligner_model = "qwen3-forced-aligner-0.6b-q4_k"
quant = "q4_k"
# n_threads = 0          # 0 = 自动根据 CPU 逻辑核心数调度
# gpu_layers = 0         # 0 = 仅 CPU 计算；省略 = 自动最大化 GPU 卸载
# language = "zh"        # 语言提示: zh, ja, en 等

[vad]
enabled = true           # 开启语音活动检测
# max_chunk_duration = 120.0
# speech_padding = 0.25

[segment]
mode = "local"           # local (本地形态素与停顿切分) 或 none (不切分)
# max_duration = 7.0     # 单条字幕最长持续时间 (秒)
# max_chars_zh = 16      # 中文字符上限
# max_chars_ja = 13      # 日文字符上限
# max_chars_en = 42      # 英文字符上限

[cache]
enabled = false          # 默认关闭缓存
cleanup_enabled = true   # 开启 LRU 淘汰机制
max_media_entries = 12   # 保留最多 12 个媒体的缓存条目

[llm]
model = "deepseek-v4-pro"
base_url = "https://api.deepseek.com/v1"
api_key = "your-api-key"

[output]
default_formats = ["srt"]
strip_punctuation = false
```

---

## 项目架构与 Workspace 划分

```
subtitle-gen/
├── crates/
│   ├── subtitle-core/    # 纯 Rust 领域逻辑（零 C++ 依赖）
│   │   ├── src/audio.rs      # Symphonia 解码与 Rubato 16kHz 重采样
│   │   ├── src/formats.rs    # SRT / WebVTT / LRC / JSON / TXT 解析与生成
│   │   ├── src/text.rs       # Lindera 词性标注、声学 DP 分句与 Needleman-Wunsch 对齐
│   │   ├── src/cache.rs      # SHA-256 增量缓存与 LRU 维护
│   │   ├── src/config.rs     # TOML 配置与 CLI 参数合并
│   │   ├── src/llm.rs        # async-openai 异步结构化校对与翻译
│   │   └── src/pipeline.rs   # Tokio MPSC 异步流式调度流水线
│   │
│   ├── subtitle-infer/   # 核心推理引擎抽象
│   │   ├── src/device.rs     # CUDA / Vulkan / Metal / CPU 硬件探测与平滑降级
│   │   ├── src/vad.rs        # FireRed-VAD 语音起止点检测
│   │   ├── src/asr.rs        # Qwen3-ASR 语音转录
│   │   ├── src/aligner.rs    # Qwen3-ForcedAligner 词级别时间戳强制对齐
│   │   └── src/engine.rs     # 专用 Worker Thread 托管 !Send ggml 运行时
│   │
│   └── subtitle-cli/     # 命令行可执行程序
│       └── src/main.rs       # clap 参数解析与 indicatif 动态多阶段进度条
```

---

## 开发与测试

```console
# 运行全部单元测试与端到端集成测试
cargo test --workspace

# 运行代码规范检查（严格模式）
cargo clippy --workspace --all-targets -- -D warnings

# 代码格式化
cargo fmt --all
```

---

## 许可证

本项目遵循 [GPL-3.0](LICENSE) 许可证开源。

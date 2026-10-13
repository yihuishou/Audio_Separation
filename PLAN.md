# PLAN.md — Audio_Separation 图形化音声分离工具

把 `models/bs_roformer_revive/` 里的 BS-RoFormer（v2 / v3）包成一个**跨平台（Windows + Linux）桌面 GUI 工具**：拖入音频 → 批量分离人声/伴奏 → 带进度、可取消、可调参、可切换模型、可打包分发。

- 环境：Windows 11 · Python 3.12（`>=3.11`）· torch 2.9.1+rocmsdk · RX 7900 XTX(gfx1100)
- 包名 `audio-separation`（0.1.0），**src 布局**
- 测试：**pytest**（`tests/` 布局），`src` 布局 ⇒ `pip install -e .` 后直接 `pytest`
- 参考基线：`tools/bs_roformer_revive_separate.py`（纯 CLI，386 行，已实测可用）

---

## 0. 已确认的决策（不再讨论）

| 项 | 决策 | 理由 |
|---|---|---|
| 依赖策略 | **手工复制** `audio_ai/paths.py` 一份到本项目 | 两个项目各自可独立运行；`audio_ai` 的模块顶层常量会连带拖入整套 core/ui，不划算 |
| GUI 框架 | **PySide6 6.12** | 与 Audio_AI 一致；QThread/信号槽最成熟；控件足够做表格与参数面板 |
| 功能范围 | 批量队列 + 进度/取消 · **只读参数展示** · 模型管理（v2/v3） · 打包发布 | ★**不提供推理参数编辑**：全部超参直接使用模型 yaml 的值，UI 只读展示（§2.11）。**不含**结果试听对比（本期不做） |
| 默认模型 | **v2**（用户指定） | 模型下拉的默认选中项；v3 可随时切换 |
| ffmpeg | **项目内嵌 `ffmpeg/`**，两平台各带各的二进制，随仓库分发、固定路径，不查 PATH、不用 imageio-ffmpeg | ★**来源即项目内嵌，不需要用户另行下载**（§8 原第 4 条作废）。两平台**共用同一个目录名**，只剩可执行文件名按平台区分 ⇒ `ROOT_MARKERS` 判定不再有平台分歧。**随包自带 ⇒ 不进前置检测**（§2.6） |
| 环境检测 | **三层，L1 静态库检测 + L2 子进程设备探测 + L3 GUI 自检面板**（§2.6） | 库状态必须在**不 import 任何模块**的前提下就能报出来 —— 本机 `import bs_roformer` 实测会挂死 |
| 依赖清单 | **§2.13 单一真源**（A 直接 / B 传递 / C GUI / D 可选 / E 开发 / F 不进清单 / **G 只检测不安装**） | 漏掉 `einx`/`hyper-connections`/`PoPE-pytorch`/`torch-einops-utils` 这类传递依赖，换机器时才会炸 |
| **`torch`** | ★**不进依赖清单，只检测**（§2.13 G 组） | ROCm/CUDA/CPU 的 wheel 都在 `download.pytorch.org` 而非 PyPI 主源，声明里写不出来；且**装错后端比没装更难排查** |
| ★**环境检测与安装分离** | **检测层永不执行 `pip install`** —— 只报状态 + 给可复制的命令（L1/L2/L3 三层皆然，见 §2.6） | 装包要动网络、可能要管理员权限、会污染环境，更可能在装完后才发现**后端选错**（用户要 ROCm，`pip install torch` 拉的是 CUDA 版）。装不装、装哪个版本，是用户的决定，不是工具启动时替用户做的事。`install_hint` 只能是字符串常量，由 **R22** 的反向断言守护（检测路径上不得出现 `subprocess`） |
| 进程模型 | **推理跑在独立子进程**（见 §2.3） | 本机 ROCm 算子挂死是**实测事实**，线程内无法中断 |
| ★★**推理后端** | **GPU-only 硬约束：模型权重与前向计算一律在 GPU 上；GPU 探测不通过就立即终止，绝不降级到 CPU、也不提供任何 CPU 回退开关**（§2.14） | 用户明确要求。三条理由：①CPU 上跑 160 M 参数 / 11 s 音频块，单次前向从 GPU 的 **3.9 s** 劣化到分钟级，"能用"等于"不可用"；②**本机 CPU 上根本跑不完** —— 端到端 50 分钟无任何输出（§1.3），而同一模型在 GPU 上 15 块共约 60 s，慢约 50 倍且中间无法观察；③提供 CPU 回退会**掩盖真正的安装/驱动故障**：用户看到"能跑"就不会去修 ROCm，直到某次真的需要 GPU 时才发现问题。更关键的是，"悄悄换个地方跑"是最坏的失败语义 —— 用户无法知道自己拿到的结果是在哪个后端上算的 |
| 测试框架 | **pytest**（`tests/` 布局） | ★**明确偏离 Audio_AI 的约定**（那边是「没有 pytest，测试就是独立脚本」）。此处按 pytest 走，见 §6 |
| 推理参数 | **直接使用模型 yaml，不给编辑入口** | 超参互相耦合且 `dim_t` 由权重决定，放开编辑只会产生静默劣化 |
| 模型日志 | **子进程 stderr 实时流，面板只保留最近 500 行** | 全曲长跑会刷上万行，`QPlainTextEdit` 无上限会吃掉内存导致 GUI 卡死 |

### 0.1 非目标（本期明确不做）

- 多 stem 分离（模型 `num_stems: 1`，只出人声 + 残差伴奏）
- ★**用户可调模型侧超参** —— `chunk_size` / `num_overlap` / `batch_size` / 采样率 / `dim_t` **一律直接取模型 yaml，UI 不给编辑入口**。这是有意收窄：这些键互相耦合（`chunk_size = hop_length × (dim_t−1)`），而 `dim_t` 由权重结构决定、用户改不动，放开 `chunk_size` 的编辑等于让用户构造出模型处理不了的组合 —— 且**不抛异常**，只在一次前向之后表现为"分离结果很差"。参数面板退化为**只读展示 + 模型摘要**。
  （仍可改的三项：**设备 / 精度 / 输出目录** —— 它们不改变模型期望的输入形状，见 §2.11）
- 结果试听 / 波形绘制
- 任务历史数据库（Audio_AI 有 SQLite，这里纯内存队列 + 磁盘缓存）
- 模型下载 / 自动更新
- 浮点以外的精度探索（见 §2.5 的 fp16 限制）

---

## 1. 当前仓库状态（写计划时的实测基线）

```
Audio_Separation/
├── .git/            master，**尚无任何 commit**
├── .idea/           未被 .gitignore 覆盖
├── .omo/            opencode 运行态
├── ses_*.json       opencode 会话文件
├── models/bs_roformer_revive/
│   ├── bs_roformer_revive_v2_by_unwa.ckpt   609.7 MB
│   ├── bs_roformer_revive_v2_by_unwa.yaml
│   ├── bs_roformer_revive_v3_by_unwa.ckpt   609.7 MB
│   └── bs_roformer_revive_v3_by_unwa.yaml
├── test_audio/                              # ★见 §1.2
│   ├── 1.第1章『耳なめ手コキオナサポ』.wav    246.4 MB · 930.000 s
│   └── 2.第2章『乳首責め手コキオナサポ』.wav  266.9 MB · 1007.557 s
└── tools/bs_roformer_revive_separate.py     386 行（已 git add）
```

**缺失**：`src/`、`pyproject.toml`、`.gitignore`、`ffmpeg/`（★来源 = 项目内嵌，随仓库分发）、`README.md`、`output/`。

**参考脚本当前跑不起来**：它 `sys.path.insert(0, REPO/"src")` 后 `from audio_ai.paths import find_ff, require_dir`，而 `Audio_Separation/` 下没有 `audio_ai` 包。本计划把它降级为**回归基线**（§6），核心逻辑上移到 `src/audio_separation/core/`，脚本改为薄壳。

### 1.1 模型配置（v3 yaml 实测值）

```
audio:   chunk_size 485100 | dim_f 1024 | dim_t 1101 | hop_length 441 | n_fft 2048
         num_channels 2 | sample_rate 44100 | min_mean_abs 0.
model:   dim 512 | depth 12 | stereo true | num_stems 1
         time/freq/linear_transformer_depth = 1 / 1 / 0
         dim_head 64 | heads 8 | attn_dropout 0 | ff_dropout 0 | flash_attn true
         dim_freqs_in 1025 | stft_n_fft 2048 | stft_hop_length 441 | stft_win_length 2048
         stft_normalized false | mask_estimator_depth 2
         multi_stft_resolutions_window_sizes (4096,2048,1024,512,256) | hop 147
         freqs_per_bands 长度 62（20×2, 12×4, 8×12, 8×24, 8×48, 4×128, 末位 129）
inference: batch_size 2 | dim_t 1101 | num_overlap 2
training:  instruments [vocals, other] | target_instrument vocals | use_amp true
```

⚠️ `freqs_per_bands` 与 `multi_stft_resolutions_window_sizes` 带 `!!python/tuple` 标注 ⇒ **PyYAML SafeLoader 直接 ConstructorError**，必须自定义 loader（§2.1）。

### 1.2 ★ 测试素材与 1 分钟切片策略

**实测规格**（ffprobe）

| 文件 | 编码 | 采样率 | 声道 | 时长 | 大小 |
|---|---|---|---|---|---|
| `1.第1章『耳なめ手コキオナサポ』.wav` | `pcm_s24le` | 44100 | 2 | 930.000 s（15.5 min） | 246.4 MB |
| `2.第2章『乳首責め手コキオナサポ』.wav` | `pcm_s24le` | 44100 | 2 | 1007.557 s（16.8 min） | 266.9 MB |

**素材的三个特性**（都直接影响测试设计）

1. **24-bit 立体声 44.1 kHz** ⇒ 与模型 `audio.sample_rate: 44100` / `num_channels: 2` **天然对齐**，无需重采样。`audio_io.decode()` 统一转 `f32le`（`-f f32le -acodec pcm_f32le -ac 2 -ar 44100`），24-bit 在这一步升到 float32 **无损**（24bit 有效位 ≤ float32 尾数 24bit）。
2. **合计 513 MB / 32.3 分钟** ⇒ ★**全曲绝不能进验证流程**。一次 15 分钟素材在 RX 7900 XTX 上跑完 610 MB 权重的 `depth:12 / dim:512` 分块推理，量级在十分钟以上；两首都跑 = 半个多小时，验证体系直接失去意义 —— 没人愿意每次改一行代码都等这个。
3. **文件名含日文与全角符号 `『』`** ⇒ ★这是个**必须被测试覆盖的边界情况**，不是顺手的事：
   - `subprocess` 走 `CreateProcessW`，Python 侧传 Unicode 路径没问题；
   - ★**实测结论（M0 已验证）**：本机 `ffmpeg-n9.0` 构建**按宽字符正确解析 argv**。用日文 + `『』` 源文件做输入、中文 + `『』` 做输出，`exit=0`，产物 1,323,102 字节（5 s × 44100 × 2ch × 3 B ≈ 匹配）。**Windows 侧的风险已排除**，§2.4 里「拷到纯 ASCII 临时名 / `-i pipe:0`」的备选方案**不需要启用**。Linux 侧仍需在该平台各验一次（R18）
   - 输出文件名从输入 stem 派生 ⇒ `output/separated/1.第1章『耳なめ手コキオナサポ』/vocals.wav` 这条路径要能在 Windows + Linux 双端建目录、写盘、删除；
   - JSON Lines 协议里传这些路径 ⇒ 必须 `ensure_ascii=False` + 管道 UTF-8（§2.8 第 2 条已覆盖），否则子进程日志面板会出现乱码。

**切片策略**（`tools/make_test_clips.py`，M0 交付）

- 生成到 `test_audio/clips/`，**原始 wav 不动**（切片可随时重新生成，原始素材是唯一真源）
- 每首取 **2 个不重叠的 60 s 切片**，默认偏移 `t0 = 240 s`（4 min 处，避开开头可能的标题/静音）与 `t0 = 600 s`（10 min 处，避开中段可能的停顿）⇒ 共 4 个切片
  - 为什么不是 2 个而是 4 个：单一切片只能暴露一种内容分布（安静段 / 密集人声段）。两个偏移 × 两首素材 = 4 种组合，能同时覆盖"人声稀疏"与"人声密集"，代价只有 4 × 60 s 的推理时间。
  - 为什么是 60 s 而不是 10 s：`chunk_size 485100` = **11.0 s**，`hop = 485100/3 = 161700` = **3.667 s**。60 s ⇒ **15 个分块、14 个内部重叠边界**（★实测修正：`range(0, total−chunk+1, hop)` 只给出 14 个等步块，而第 14 块的右边界 2,587,200 < 2,646,000 盖不到文件尾，`chunk_starts()` 会**再补一个**贴尾块 ⇒ 15；参考脚本的 `--dry-run` 数等步块会报 14，是少算的）；10 s 只有 1 块，`num_overlap` 与 `wsum` 归一化逻辑**一行都覆盖不到**。
- 切片格式：`pcm_s24le / 44100 / 2ch`，**与原素材同参数，不做任何重编码降级**（二次量化会污染 mix 校验的判据）。单个切片约 15.9 MB。
- 切片命名：`<原stem>_t{偏移}s.wav`，例：`1.第1章『耳なめ手コキオナサポ』_t240s.wav`
- **切片文件带全角 `『』`** ⇒ 切片生成这一步本身就成了 Unicode 路径的首个测试点

### 1.3 ★ GPU 性能实测与「50 分钟零输出」事件（M2 期间实测，§2.14 的依据）

**同一个 v2 模型、同一块 60 s 切片，只换设备：**

| 环节 | GPU（`--device cuda`）耗时 |
|---|---|
| `import torch` | **1.41 s**（带一条 ROCm 工具链空格路径的 stderr 噪音，不阻塞） |
| `torch.cuda.is_available()` | 1.42 s → `True` |
| `torch.cuda.get_device_name(0)` | 1.42 s → `AMD Radeon RX 7900 XTX` |
| 4×4 张量 `.to("cuda")` → `.cpu()` | 1.52 s |
| `build_model(cfg, device="cuda")` | **3.83 s**（699 张量键名形状全对齐，159,758,092 参数） |
| 单次前向 `(1,2,485100)` | **3.9 s** ⇒ 15 块端到端约 **60 s** |

**⇒ GPU 整条路径是通的，而且很快。** 这不是一台"GPU 跑不动"的机器。

**事件**：同期的 CLI 端到端（`--device cuda`，60 s 切片）却在 `构建模型 v2（权重 …）` 那一行之后**再无任何输出，持续 50 分钟**直到被 shell 工具杀掉。同期 `python -m audio_separation.core.model --check v2 --device cuda` 同样挂死，而不带 `--device`（走 CPU）**秒过**。

**排查过程与结论（勿重复劳动）**

1. **GPU 本身没问题** —— 最小复现（`import torch` → `is_available` → `get_device_name` → 小张量 `.to("cuda")` → `.cpu()`）**exit=0，1.53 s 全通过**。排除设备初始化与显存问题（24 GB 卡，模型仅 610 MB）。
2. **模型构建 + 前向在 GPU 上也没问题** —— 带 `faulthandler.dump_traceback_later(90, exit=True)` 的探针把 `build_model`（3.83 s）与一次前向（3.9 s）全部跑完并输出 `ALL OK`。**挂死点不在这两个函数里。**
3. **卡住的进程连 `TerminateProcess` 都要等好几秒**（`Stop-Process -Force` 返回后 `taskkill` 报「找不到进程」），工作集冻结在 2.15 GB 不再变化 —— 这是**卡在 GPU 驱动调用里、无法立即中断**的形态，而非 Python 层死循环（后者能被 `os._exit` 立刻打断）。
4. ★**CLI 与探针之间唯一未排除的差异**：CLI 里 `decode_audio`（`subprocess.run` 起 ffmpeg）跑在 **`import torch` 之前**，探针里没有这个顺序。**该假设尚未验证**。

**为什么把这件事写进计划**：它同时决定了两条设计。

- **§2.14 GPU-only 硬约束** —— 正因为没有 CPU 回退，"跑不动"必须**当场暴露**成明确失败。所以 §2.3 的子进程看门狗在 GPU-only 下是**必需**的而非可选的：它是把"GPU 卡住"翻译成"GPU 不可用，已终止"的那一层。没有它，用户面对的就是 50 分钟无输出。
- **§2.8 的四条启动期规避是实打实的约束**，不是防御性编程：ROCm 工具链对 `C:\Program Files` 这类含空格路径有 bug（`Program: Unknown command line argument 'Files\...\offload-arch.exe'`），这是本项目所有 ROCm 异常的公共嫌疑点。

**待验证项**：若探针复现挂死 ⇒ 根因是"先起子进程再 `import torch`"，需在 §2.8 增加一条规避（先 import torch 再解码，或子进程不继承 stdio 管道）。若不复现 ⇒ 挂死与调用顺序无关，R1 按"算子级随机挂死"处理，看门狗是唯一防线。
- `test_audio/` 与 `test_audio/clips/` **全部进 `.gitignore`**（513 MB + 64 MB），`clips/` 由 `make_test_clips.py` 幂等重建（切片已存在且长度对得上就跳过）
- 生成命令用 ffmpeg 而非 Python 写 wav：
  `ffmpeg -nostdin -y -ss <t0> -i <src> -t 60 -c:a pcm_s24le -ar 44100 -ac 2 <dst>`
  ★`-ss` 放在 `-i` **之前**（输入侧快速定位，秒级）；放后面要先解完整文件，30 分钟素材上会明显变慢。

**测试输入约定**：所有 pytest 用例通过 `conftest.py` 的 `clips` fixture **只吃 `test_audio/clips/*.wav`**，测试代码里不出现 `test_audio/` 下的全曲路径。`cli.py` 保留 `--full-length` 作为**人工长跑的逃生开关**（默认关闭）—— 它只影响 CLI，不影响 pytest（§6.3）。

---

## 2. 核心技术决策

### 2.1 配置加载：`core/config.py`

- `class _YamlLoader(yaml.SafeLoader)`，`add_constructor("tag:yaml.org,2002:python/tuple", ...)` → `tuple(constructor.construct_sequence(node))`
- `load_config(ckpt_path) -> dict`：读同名 `.yaml`，注入 `_dir` / `_weights`
- **不硬编码任何超参到 Python**，全部来自 yaml —— v2/v3 差异只在数据，代码零分支

**交叉校验**（载入时立刻做，不通过直接 `ConfigError`）：

| 校验 | 实测关系 | 不通过的后果 |
|---|---|---|
| `model.dim_freqs_in == audio.n_fft // 2 + 1` | v3: `1025 == 2048//2+1` ✓ | STFT 频点数错位 |
| `audio.sample_rate == 44100` | 权重只在此采样率下训练 | 变调式降质 |
| `audio.dim_t == inference.dim_t` | v3: `1101 == 1101` ✓ | 推理与训练的时间维不一致 |
| `audio.chunk_size == audio.hop_length * (audio.dim_t - 1)` | v3: `485100 == 441*1100` ✓（若误写成 `× dim_t` 则 `441*1101 = 485241`，**差 141**） | —— |

★ **`chunk_size` 的关系是 `hop_length × (dim_t − 1)`，不是 `× dim_t`**：v3 实测 `441 × 1100 = 485100` ✓（`441 × 1101 = 485241`，差 141）。频率侧同理：`dim_freqs_in = n_fft//2 + 1 = 1025`，而 `dim_f = 1024 = n_fft//2`。**两个维度都比"整除"结果少/多一个格子**，是 STFT 边界对齐的必然结果。

这个 −1 写错一个字符，模型会**静默输出错位的人声**（不报错，只是变差），所以必须在校验里写死并配断言测试。测试取 v2/v3 两个 yaml 各断言一次，若未来换模型版本导致关系变化，测试立刻红而不是让用户在成品里听出差别。

### 2.2 模型构建与权重校验：`core/model.py`

从参考脚本逐条搬，**每条都有存在的原因，不许"简化"**：

1. `mcfg["flash_attn"] = False` —— ★**默认值是 False，由调用方按后端覆盖**：§2.6 L2 的探测结论（`probe_device()` 返回的 `flash_attn` 建议值）。ROCm 强制 False，CUDA 可为 True。**探测未完成时保守取 False**
2. `mcfg["num_residual_streams"] = 1` —— revive 权重键名**没有** `.branch.` 这一层
3. 用 `inspect.signature(BSRoformer.__init__).parameters` 过滤库不接受的多余键，并 log 出来 —— BS-RoFormer 库版本升级会加参数，revive 权重跟不上，**静默丢弃会变成难查的权重不匹配**
4. 构造后遍历 `model.modules()` 把 `to_value_residual_mix` 置 `None` —— lucidrains 后加的，revive 权重里没有
5. `torch.load(..., weights_only=False)`，权重取 `ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))`
6. **严格比对**：模型侧键名剥掉 `.branch.` 后，与权重侧做**键集合 + 形状**双重校验，不匹配立即 `RuntimeError` 并打印前 N 个差异键
7. `model.eval()` + `model.to(torch.float32)`（§2.5）

### 2.3 ★ 进程模型：UI 进程 + 推理子进程

**这是本计划最关键的架构决策。**

本机实测：`torch` / `bs_roformer` 在 `C:\Program Files\Python\Python312\...` 下会调用 ROCm 的 `offload-arch.exe`，该工具链**对含空格的路径处理有 bug**，导致 `import bs_roformer` 挂起或报错：
```
Program: Unknown command line argument 'Files\Python\Python312\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\offload-arch.exe'.
```
且参考脚本里的 `_Watchdog` 存在的原因就是：**ROCm 算子挂死时 Python 无法中断**，只能用 `os._exit(3)` 粗暴退出 —— 这在 GUI 里**不可接受**（窗口直接消失，用户丢配置）。

所以：

```
┌─ UI 进程 (PySide6) ────────────┐        ┌─ 推理子进程 ──────────────────┐
│ 不 import torch/bs_roformer     │  stdout│ import torch, bs_roformer    │
│ QProcess 启动/terminate/kill    │ ─────► │ ffmpeg 解码 → 分块推理 → 落盘 │
│ 日志面板 / 进度条 / 队列表格     │◄────── │ JSON Lines:                    │
│                                 │        │  {"ev":"progress","done":3,...}│
│ 取消 = terminate() → 3s → kill() │        │  {"ev":"done","vocals":"...wav"}│
└─────────────────────────────────┘        └────────────────────────────────┘
```

- 子进程入口：`python -m audio_separation.worker --job <job.json>`
- **子进程负责解码 + 推理 + 写文件**，只回传**进度 / 日志 / 产物路径** ⇒ IPC 载荷极小，不传音频数组
- 启动快、崩溃隔离、取消即 kill、显存不泄漏在 UI 侧、**UI 永不因 torch 导入卡死**
- 打包后：开发态用 `sys.executable -m audio_separation.worker`，冻结态用 `QProcess(sys.executable, ["--worker", job])`，由 `paths.self_executable()` 统一判断

代价：多一层 IPC 协议（约 6 种事件）。**接受** —— 换来的是"推理卡死时 GUI 还活着"。

### 2.4 音频 IO：`core/audio_io.py`

- `decode(ffmpeg_bin, src, sr) -> np.ndarray (N,2) float32`
  `ffmpeg -nostdin -y -i <src> -f f32le -acodec pcm_f32le -ac 2 -ar <sr> -` → `np.frombuffer(stdout).reshape(-1,2).copy()`
  - `-ac 2` 强制双声道（模型 `num_channels: 2`），单声道源会被复制成两路
  - `-nostdin` 必须有，否则子进程会抢 stdin
  - **stderr 要单独管道接走**，否则 ffmpeg 的日志会灌进 JSON Lines 协议里
  - ★**路径可能含日文与全角符号**（`test_audio/1.第1章『耳なめ手コキオナサポ』.wav`，§1.2）。Python 侧 `subprocess` 走 `CreateProcessW` 没问题，但 **ffmpeg 自己**是否按宽字符解析 argv 取决于其 Windows 构建 ⇒ M0 的切片生成就是第一个实测点，若失败则改为「先用 ffmpeg 拷到纯 ASCII 临时名再解码」或换 `-i pipe:0` 从 stdin 喂。**不要假设，先测。**
- `write_wav(path, wav, sr)`：`scipy.io.wavfile`（**不用 soundfile** —— 参考脚本注释记录了 libsndfile 1.2.2 的 bug）；`clip → *32767 → <i2`；**先写 `<name>.tmp` 再 `os.replace()` 原子替换** ⇒ 取消/崩溃不留半成品被当成成品
  - ★输出目录名从输入 stem 派生 ⇒ 必然含 `『』`。建目录 / 写盘 / 删除在 Windows 与 Linux 双端都要过（Linux 的 ext4/mkdir 对全角无特殊处理，但要确认**终端与日志的 UTF-8 输出**，§2.8 第 2 条）
- 支持格式交给 ffmpeg（wav/flac/mp3/m4a/ogg/opus/aac/wma…）
- ★**ffmpeg 是项目内嵌资源，不是 pip 依赖** ⇒ 不进 §2.6 的前置检测，路径解析只做「有没有」（`paths.find_ff()` → `ffmpeg/bin/ffmpeg.exe` 或 `ffmpeg/bin/ffmpeg`），不查版本、不探能力。打包漏了由 `tests/test_packaging.py` 兜（R20）
- ★**24-bit 输入无损**：`pcm_s24le → f32le` 是升精度（24bit 有效位 = float32 尾数 24 位），不会引入量化误差 ⇒ `mix_check` 的 `< 1e-3` 判据在 24-bit 素材上依然成立

### 2.5 精度策略

| 后端 | dtype | 说明 |
|---|---|---|
| ROCm | `float32` | ★参考脚本注释：**ROCm 上 bf16 的 STFT 会掉精度** ⇒ 强制 fp32，`use_amp` 在本后端忽略 |
| CUDA | `float32` / `float16` 可选 | 参数面板放开，默认 fp32 |

★**表里没有 CPU —— 这是 §2.14 的硬约束**：`cpu` 不是一个可选后端，而是一个**不存在的配置**。参数面板里根本不该出现它（出现了就是在给用户一个会静默劣化几十倍的选项）。

参数面板的精度下拉在 ROCm 下**禁用 fp16 并给 tooltip 说明原因**，不做静默改写。

§2.6 前置环境检测（**只检测，不修复安装** —— §0 决策表）

检测**重点是 Python 库**，分三层：L1 静态（毫秒级，不 import）· L2 动态设备（子进程 + 超时）· L3 GUI 面板（§M4）。
**ffmpeg 项目内嵌，不进检测** —— 它是随仓库分发的项目资源而非环境依赖，缺失属于分发事故，由 `tests/test_packaging.py` 兜，不占启动路径。

#### L1 · 静态库检测（**不 import 任何目标模块**）

关键手法：`importlib.metadata.version(name)` 只读安装元数据，**不执行目标模块的 `__init__.py`**。

为什么必须这样：本机实测 `import bs_roformer` 会挂死（ROCm 工具链对 `C:\Program Files` 空格路径有 bug，>120 s 不返回）。若静态检测也走 `import`，torch 一挂全盘皆黑，连「哪个包装没装」都报不出来。`pip show` 路径本机实测安全。

| 级别 | 检查项 | 判定 |
|---|---|---|
| **FAIL** | Python 版本 ≥ 3.11 | 版本号 |
| **FAIL** | **`torch`（★只检测，不安装 —— §2.13 G 组）** | 版本号 + 从版本串识别 ROCm/CUDA/CPU build |
| **FAIL** | 直接依赖：`numpy` `PyYAML` `scipy` `BS-RoFormer` | 版本号（§2.13 A 组） |
| **FAIL** | 传递依赖：`beartype` `einops` `einx` `hyper-connections` `pope-pytorch` `rotary-embedding-torch` `torch-einops-utils` `librosa` | 版本号（§2.13 B 组） |
| **WARN** | `PySide6` | 缺失只影响 GUI 入口，CLI 仍可用 |
| **WARN** | `pytest` | 缺失只影响开发 |
| **INFO** | `torchaudio` | 可选，§2.13 D 组；缺失不报 |

- **`min_version` 是真的下限，不是摆设**：装到的版本低于它就把该条目的 `level` 从 PASS 降下来并附 `install_hint`。比对时**只取版本串的前导数字段**，丢掉 `+rocmsdk20260116` / `cu124` / `rc1` 这类本地标签 —— 本地标签不代表上游版本序（`2.7.1+cu124` 并不小于 `2.9.1`）。版本串解析不出数字时**宁可放过也不误报**：一份非标准版本串不该让整个自检面板变红
- ★**torch 的 `min_version` 只能取能力下限（BS-RoFormer 声明的 `>=2.0`）**，不得抄本机版本串 `2.9.1+rocmsdk20260116` —— 那等于把 ROCm 构建号固化成硬性下限，Linux 机器装个正常的 `2.7.1+cu124` 会被误判成版本不够
- `numba` / `soundfile` 是 librosa 的下游，只在 librosa 缺失时连带提示，不单独列
- 每个 FAIL 都附**可操作的修复动作** —— ★指的是**给一条可复制的命令**（「`pip install einx`」/「版本需 ≥ X」），**不是执行它**（§0 决策表：检测与安装分离）
- ★**本层只读状态，从不执行任何 `pip install`** —— 检测与安装分离。装包是用户的决定，不是工具启动时替用户做的事（自动装包会踩到权限、虚拟环境、网络、以及最要命的**装错后端**：用户要 ROCm，`pip install torch` 拉下来的是 CUDA 版）
- ★**`torch` 缺失时的引导只给「怎么自己装对后端」，不给一键安装**。因为可用后端取决于用户的硬件：ROCm wheel 在 `download.pytorch.org/whl/rocm`、CUDA 在 `.../whl/cu124`、CPU 在 `.../whl/cpu`，工具无权替他选。L3 面板按探测出的 OS + 已有驱动信息，给出**对应的那一行命令**供复制

#### L2 · 动态设备探测（子进程 + 超时）

```
probe_device() 在子进程里依次试（§2.8 四条规避在这个子进程入口逐条落实）：
  1. import torch                        → 失败 ⇒ FAIL + 原始 stderr（唯一必须 import 的层）
  2. torch.cuda.is_available()           → ★False ⇒ **FAIL，立即终止**（§2.14，不降级 CPU）
  3. torch.cuda.get_device_name(0)       → 记录设备名
  4. from bs_roformer import BSRoformer  → 失败 ⇒ FAIL（L1 已报同一个错则跳过这步）
  5. ★一次极小的假推理：(1,2,4096) 过一遍 STFT + mask estimator
       —— 捕捉 L1 查不到的 kernel 层错误（flash_attn、ROCm 算子签名、dtype 不支持）
返回 {ok, backend: rocm|cuda, device_name, torch_version, rocm_hint, error}
```

- ★★**第 2 步的判定从「WARN 并降级 CPU」改成「FAIL 立即终止」**（§2.14）。理由不是"更严格"，而是**降级在这里是一个没有出口的错误处理**：`WARN` 意味着"还能干活"，但按 §2.14 它后面根本没有 CPU 路径可走 —— 一条 `WARN` 会让用户以为程序会继续跑，然后在真正开始分离时才崩。所以这一档只能是 `FAIL`。
  `error` 字段必须写清**是探测失败还是硬件不存在**：前者给驱动/安装的排查方向（`rocm_hint`），后者直说"这台机器上没有可用的 GPU，本工具不提供 CPU 模式"。
- `flash_attn`：ROCm → **False**；CUDA → 先试 `True`，捕获异常回落 `False` 并在日志注明
- ★**返回 `flash_attn` 建议值**：L2 的假推理已经把这套 kernel 跑通一遍，把结论带回来给 `build_model()` 用（§2.2 第 1 条）—— 免得 CUDA 路径又踩一次同样的异常。L2 未完成时 `build_model()` 保守取 `False`（正确性优先于速度）
- ★**每一步都带超时**（默认 90 s），超时按 FAIL 处理并保留子进程 stderr —— 本机 torch 导入有 >120 s 的历史，不设超时等于没有这一层。**超时同样是终止条件**（GPU 状态未知时不允许"先跑起来再说"）
- L2 的结果**缓存**（`env_check.probe_device()` 带 lru_cache + `force=True` 逃生口），设备切换按钮才重新探测

> ★**L2 假推理跑在 CPU 上是有意的例外**。它不是"用 CPU 跑模型"，而是**在 GPU 上**过一遍 STFT + mask estimator（`(1,2,4096)`）来验证 kernel 可用性 —— 检验的是 GPU 那条路能不能走通，不是拿 CPU 当后端。§2.14 禁止的是**把推理放到 CPU 上**，不是禁止在探测阶段花一次 CPU。

#### L3 · GUI 启动自检面板（`ui/env_panel.py`）

启动后异步跑 L1 + L2，汇总成表：

| 列 | 内容 |
|---|---|
| 项 | 检查项名 |
| 级别 | PASS / FAIL / WARN / INFO |
| 详情 | 版本号 / 报错原文 / 建议动作 |
| 处置 | 「去装 X」「不提供 CPU 模式，请装 GPU 后端」「仅影响开发」 |

- **不阻塞主窗口**：L1 同步（毫秒级），L2 放后台线程；用户可以在探测未完成时先看界面。但★**探测未完成 ≠ 可以开工**：分离动作必须等 L2 出结果（§2.14），此时队列按钮显示"正在探测 GPU"而不是可点
- FAIL 时禁用「开始分离」，并把修复指引做成可点击命令（终端里复制即用）
  - ★**「复制」而不是「安装」**：面板永远不主动执行 pip。`torch` 缺失时按 OS + 驱动给出对应后端的那一行（§2.13 G 组），用户自己决定要不要跑
- **一键复制诊断报告** = L1 + L2 全文 + `paths.diagnose()` 探测链 + 版本清单，方便贴给别人排障
- 探测本身出故障时提供「跳过检查」入口 —— 自检绝不能变成让工具变砖的前置依赖

### 2.7 分离算法：`core/separator.py`

```python
chunk       = cfg["audio"]["chunk_size"]        # 485100 ≈ 11.0 s @44.1k
num_overlap = cfg["inference"]["num_overlap"]   # 2
hop         = chunk // (num_overlap + 1)        # 161700
starts      = range(0, total, hop)，末尾不足 chunk 时补 `total - chunk`
window      = hann_window(chunk)   # periodic=False，两端归零
y           = model(x)  # x:(1,2,chunk) → y:(1,1,2,chunk)
out[seg] += y * window ; wsum[seg] += window    # ★分子分母同窗
最后 out /= wsum
```

- ★`periodic=False`：参考脚本注释明确写了，普通 Hann（periodic=True）会在块边界引入周期性断点
- ★**加窗与归一化必须用同一个窗**（`Σ(y·w)/Σ(w)`，即加权平均）。
  这是本节**曾经写错、已修正**的一处：原方案写的是 `out += y*w ; wsum += w²`，
  当模型输出等于输入时整体增益是 `Σw/Σw² = 1.3333`，**整轨人声偏大 4 dB**
  （2026-10-13 在真实 chunk/hop 参数下实算验证）。
  只有分子分母**同步**加窗（`Σ(y·w²)/Σ(w²)`）才精确重建，那是「分析窗 + 合成窗
  各加一次」的约定；本模型作用于**未加窗**的原始块，合成侧是唯一一次加窗，
  所以用 `Σw/Σw`。参考脚本用的就是这一种，回归基线因此仍可比
  （`tests/test_separator.py::test_reconstruction_is_exact` 锁死该性质）
- 单 stem ⇒ **vocals = 模型输出，instrumental = 原轨 − vocals**（残差）。GUI 必须显示这一点，否则用户会以为伴奏被"识别"了
- 进度回调：每块一次，回传 `(块序号, 总块数, 已用秒, ETA 秒, 瞬时 RMS)`
- 每文件结束 `torch.cuda.empty_cache()`

★ **`inference.batch_size` 在本项目里是死值，不参与计算。** yaml 写着 `2`，但 §2.7 的循环是**逐块串行**的：每次取一个 chunk 做 `model(x)`，`x:(1,2,chunk)` 的 batch 维恒为 1。参考脚本也是这样 —— 它读了这个值但没用。

- 要真正用上 batch，需要把**多个邻近 chunk 堆叠成一个 batch 前向**，代价是：① 每块输入要补齐到 `chunk_size`（尾块尤其）② 显存占用 ×batch ③ 批内各块的 mask 缩放要在批后正确拆回 ④ 与 `hop` 重叠加窗的边界逻辑要重新推导
- 收益（吞吐）与代价（引入一类新的边界 bug）不成比例，**本期不做**
- 因此 §2.11 把 `batch_size` 列为**只读展示**，并在 UI 上注明「当前版本未启用批推理」—— 不注明的话用户看到"批大小 2"会以为在生效

### 2.8 启动期环境规避（顺序敏感，子进程入口逐条落实）

| # | 动作 | 不做的后果 |
|---|---|---|
| 1 | `NUMBA_CACHE_DIR` → `<tmp>/numba_cache` | librosa 的 numba 装饰器 import 期写 site-packages `__pycache__`；目录不可写时 `tempfile.TemporaryFile(dir=path)` **静默死循环** |
| 2 | `stdout/stderr.reconfigure(utf-8)` | Windows GBK 下打印 `⇒`(U+21D2) 直接崩 |
| 3 | 再 `import librosa / torch / bs_roformer` | 顺序颠倒则 1、2 失效 |
| 4 | ffmpeg 的 stderr 单独管道 | 污染 JSON Lines 协议 |

### 2.9 取消语义

| 层级 | 机制 | 生效点 |
|---|---|---|
| 队列级（未开始的文件） | 内存队列直接丢弃 | 立即 |
| 文件级（解码中） | 子进程 `terminate()` | 立即 |
| 文件级（推理中） | `terminate()` → 等 3 s → `kill()` | ≤ 3 s |
| 分块级（子进程内部） | 子进程也装一个 `threading.Event` 检查点，在块间轮询 terminate 请求 | 块间（≈1–3 s） |

- `terminate()` 后**必须确认临时 `.tmp` 文件被清掉**，否则下次跑会看到"存在但截断"的 wav
- 已完成的文件**不受取消影响**，产物保留

### 2.10 路径真源：`paths.py`（从 `audio_ai/paths.py` 手工复制后改）

| 项 | audio_ai | audio_separation |
|---|---|---|
| `ROOT_MARKERS` | `("models", "ffmpeg-n9.0")` | `("models", "ffmpeg")`（**双标记**，单标记会撞名） |
| `ENV_ROOT` | `AUDIO_AI_ROOT` | `AUDIO_SEPARATION_ROOT` |
| `ENV_OUTPUT` | `AUDIO_AI_OUTPUT` | `AUDIO_SEPARATION_OUTPUT` |
| ffmpeg 可执行名 | 固定 `ffmpeg.exe` | **按平台**：`ffmpeg.exe`(win) / `ffmpeg`(linux)，**都在 `ffmpeg/bin/` 下** |
| 额外函数 | — | `self_executable()`；`test_audio_dir()` / `test_clips_dir()`（§1.2）。~~`ffmpeg_root()` 做平台目录选择~~ —— ★目录名统一后这个函数不需要了 |

★ **ffmpeg 来源 = 项目内嵌**：两平台都用同一个 `ffmpeg/` 目录名，各带各的二进制，**随仓库分发**。`ffmpeg/bin/ffmpeg.exe`（Windows）或 `ffmpeg/bin/ffmpeg`（Linux）。目录名统一的意义：`ROOT_MARKERS` 判定不再有平台分歧 —— 之前 `ROOT_MARKERS` 硬编码 `ffmpeg-n9.0` 而 Linux 实际放 `ffmpeg-n9.0-linux`，双标记判定在 Linux 上必然失配。这个坑随着目录统一一并消失。

**保留 audio_ai 版本的三条设计约束**（都是踩出来的）：
1. `repo_root()` **只在"显式配置错误"时抛异常**，自动探测全未命中 ⇒ 返回 best-effort 猜测并把失败记进诊断（否则"只想打开窗口看设置"也会被模型缺失挡住）
2. 根目录判定用**双标记同时存在**，不是单标记
3. 错误信息必须列出**试过哪些目录、每个缺什么**（`diagnose()` 完整探测链）

新增：`paths.require_model_dir()`、`paths.list_models()`（扫 `models/bs_roformer_revive/*.yaml` → 配对同名 `.ckpt`，只认成对的）。

### 2.11 参数面板：★只读展示，无编辑入口

**决策：直接使用模型自带配置，移除参数编辑能力。** 所有推理超参一律取自模型 yaml，UI 不提供修改。

| 展示项 | yaml 键 | v3 实测值 | 说明 |
|---|---|---|---|
| 模型 | `model._weights` | **默认 v2** | 下拉由 `paths.list_models()` 动态填充，选中后刷新摘要 |
| 设备 | — | **GPU（只读）** | ★**不再是可改项**（§2.14）。显示后端与卡名（如 `ROCm · AMD Radeon RX 7900 XTX`），由 §2.6 L2 探测填入；探测未通过时显示红色并禁用所有分离动作。**没有下拉框** —— 可选项只有一个，不该做成看起来能选的样子 |
| 块大小 | `audio.chunk_size` | 485100（11.0 s） | **只读** |
| 时间维 | `audio.dim_t` | 1101 | **只读** |
| 重叠 | `inference.num_overlap` | 2 ⇒ hop = 161700 | **只读** |
| 批大小 | `inference.batch_size` | 2 | **只读**（★ 但本项目不用它，见 §2.7） |
| 采样率 | `audio.sample_rate` | 44100 | **只读** |
| 精度 | — | fp32 | **可改**（fp32/fp16），后端不支持则禁用（§2.5） |
| 输出目录 | `AUDIO_SEPARATION_OUTPUT` | `output/separated/` | **可改**（文件系统偏好，与权重无关） |

**为什么砍掉编辑能力**：这些超参**互相耦合**，`chunk_size = hop_length × (dim_t − 1)`（§2.1）是一个跨三个 yaml 键的恒等式。用户能编辑 `chunk_size` 却改不了 `dim_t`（后者由权重结构决定，改了就对不上），组合起来必然产生模型处理不了的输入；而这个错误**不抛异常**，只在一次前向之后表现为"分离结果很差"。参考脚本的 CLI 虽然带了 `--device`，但对 chunk/overlap 是硬编码读 yaml 的 —— 正是这个原因。

保留"精度 / 输出目录"两项可改，是因为它们**不改变模型期望的输入形状**：换精度只换 dtype（§2.5 已限定 fp32/fp16），换输出目录只影响落盘位置。

★**设备原本是三项可改之一，现在不是了** —— 见 §2.14。没有第二档可选项时，一个下拉框是负资产：它承诺了"你可以在这些之间选"，而实际上只有一个非空选项。

UI 上把这一层做成**灰底只读标签**（不是禁用输入框 —— 禁用框看起来像坏了），并在分组标题上写明「以下参数由模型定义」。

### 2.12 预设：`core/presets.py`

- 存 `output/presets.json`（**JSON**，原子写），结构 `{name: {model_key, device, dtype, output_dir}}`
  - ★ 移除了 `chunk_size` / `num_overlap` / `batch_size` —— 它们现在是模型常量（§2.11），存进预设等于允许用户间接改它们，等于把刚砍掉的编辑能力从后门放回来
- 内置预设：**只有 1 个** —— `默认`（`model_key: v2` · fp32 · auto · `output/separated/`）。原先的"高质量 overlap 4 / 快速 overlap 1"随编辑能力一起取消，因为它们唯一的作用就是改 overlap
- QSettings 只存**窗口几何/最近目录**这类 UI 状态，**不存推理参数**（参数走 presets.json，便于跨机器同步）
- `presets.json` 损坏 / 字段缺失 ⇒ 回落默认并提示，**不崩**（`test_presets.py` 覆盖）
- ★**预设里的 `device` 字段一并移除**（原 `{name: {model_key, device, dtype, output_dir}}` → `{name: {model_key, dtype, output_dir}}`）：§2.14 之后设备不再是用户可选的维度，存进预设只会让一批旧预设文件带着一个无效字段进来

### 2.14 ★推理后端：GPU-only 硬约束

**决策（用户明确要求）：模型权重与前向计算一律在 GPU 上。GPU 探测不通过就立即终止，绝不降级到 CPU，也不提供任何 CPU 回退开关。**

**具体含义**

| 面 | 规则 |
|---|---|
| `build_model(cfg, device=...)` | 只接受 `cuda`。ROCm 构建的 torch 用的也是 `cuda` 设备串（ROCm wheel 不引入 `rocm` 设备名），所以**不需要按后端分支**，也不需要"如果 ROCm 就换成 rocm"这种猜测 |
| `separate_vocals(..., device=...)` | 同上 |
| `env_check.probe_device()` | 返回 `backend ∈ {rocm, cuda}`。`torch.cuda.is_available()` 为 False、或子进程超时、或 `torch.load` 之前任何一步失败 ⇒ `ok=False` + FAIL 原因 |
| `cli.py` | **不提供 `--device cpu`**，也不提供 `--device` 里的 cpu 取值。`argparse` 会直接拒绝这个参数 —— 让它"能输入但运行时才报错"是错的 |
| 参数面板 | 设备是只读标签（§2.11），没有下拉框 |
| 预设 | 不存 device（§2.12） |
| 失败表现 | 一次性把所有 FAIL 原因列全（含 `rocm_hint`），然后退出。**不做"要不要继续尝试 CPU？"的询问** —— 那等于把决策又交回给用户 |

**★"内存"这条要拆开说，不能一刀切**

用户说的"禁止模型使用内存和 CPU"里，"CPU"指的是**计算发生地**，"内存"要区分两件事：

1. **推理计算（权重的前向）—— 严格禁止在 CPU 上。** 这是本节的硬约束。
2. **加载期的内存占用 —— 无法避免，且必须允许。** 610 MB 的 `.ckpt` 是从磁盘读进内存再 `.to("cuda")` 的；`torch.load` 没有"直接加载到显存"的路径。任何人只要在磁盘上放权重文件，就必然有一次内存拷贝。**这一段不构成"用 CPU 跑模型"**，所以不禁止。
   唯一能做的是把这一段的内存**暴露出来**而不是假装不存在：自检面板显示"加载权重峰值内存约 0.7 GB"，让用户知道自己占了什么，而不是在 16 GB 内存的机器上被 OOM  Killer 悄悄杀掉。

**为什么不给"紧急逃生"留个口子**

理论上"GPU 驱动当天挂了，至少让我用 CPU 出个结果"是个合理诉求。但在这份计划里它会具体化成三处坏处：

1. **它一定会被打开**，然后成为事实上的默认路径 —— 因为它总是能跑，哪怕慢 50 倍
2. **它掩盖故障**：用户看到"能出结果"就不会去修 ROCm，直到某次真的需要 GPU 时才发现
3. **最坏的一点：静默换后端**。用户在 GPU 上跑和在 CPU 上跑，产物是一样的文件名、一样的界面、一样的"成功"，但质量与耗时完全不同。等用户发现"今天怎么这么慢"，已经无从判断是哪一次分离跑在了 CPU 上

如果确实需要 CPU 出活，那是**另一个工具**（不上 GUI、不走这套模型服务路径），不是本工具的一个开关。

**★这条约束顺带解释了 §1.3 的 50 分钟挂死**：正因为没有 CPU 回退，"跑不动"这件事必须**当场暴露**成一条明确的失败，而不是变成一个 50 分钟没有输出的运行。所以 §2.3 的子进程看门狗在 GPU-only 下是**必需**的而非可选的 —— 它是把"GPU 卡住"翻译成"GPU 不可用，已终止"的那一层。


### 2.13 依赖清单（实测本机版本，据此写 `pyproject.toml` / `requirements.txt`）

**A · 直接依赖（参考脚本实际 `import` 的）**

| 库 | 本机版本 | 用途 | import 时机 |
|---|---|---|---|
| `numpy` | 2.4.3 | `(N,2)` float32 缓冲、重叠加窗累加 | 顶层 |
| `PyYAML` | 6.0.3 | 配置加载（需自定义 `!!python/tuple` loader） | 顶层 |
| `BS-RoFormer` | 1.2.4 | `from bs_roformer import BSRoformer` | 延迟（`build_model` 内） |
| `scipy` | 1.16.1 | `scipy.io.wavfile` 写盘（避开 soundfile libsndfile 1.2.2 bug） | 延迟（`write_wav` 内） |

**G · ★只检测、不安装（不进 `pyproject.toml` 的 dependencies）**：`torch`（本机 `2.9.1+rocmsdk20260116`）

`torch` 确实是直接依赖，但**刻意不写进依赖清单**，只由 §2.6 L1 检测它装没装、什么版本、什么后端。理由：

| 原因 | 说明 |
|---|---|
| **wheel 不在 PyPI 主源** | ROCm 在 `download.pytorch.org/whl/rocm`、CUDA 在 `.../whl/cu124`、CPU 在 `.../whl/cpu`。写 `torch>=2.4` 后 `pip install -e .` 会去 PyPI 拉默认（CUDA）版 —— 在 AMD 机器上装出一个 3 GB 的 N 卡驱动依赖，用户拿到的是**装错后端**而不是"没装" |
| **后端是硬件决定的，不是依赖声明能表达的** | 同一个 `torch` 版本号对应完全不同的 wheel，声明里写不出来，只能由用户选 |
| **本机版本串无法转写成通用约束** | `2.9.1+rocmsdk20260116` 是 Windows ROCm 专属版本，锁进 `requirements.txt` 会让 Linux 用户直接装不上 |
| **自动安装的风险** | 装包要动网络、可能要管理员权限、可能污染全局环境、更可能在装完后才发现后端选错 —— 而这一切发生在用户双击启动工具时 |

因此 §2.6 L1 把 torch 列为 **FAIL 级检查项**（缺了确实跑不了），但**只报告状态、给出对应后端的安装命令供复制**，不代为安装。其余 A–F 组全部正常进 `pyproject.toml`，`pip install -e .` 会自动解决。

★ 这一条也解决了 Linux 侧的头号障碍：照 `requirements.txt` 装不崩，因为里面根本没有 torch —— 用户按 README 选自己硬件对应的那条命令装即可。

**B · 传递依赖（`BS-RoFormer 1.2.4` 声明的 `Requires`，import 时被连带拉起）**

`beartype 0.22.9` · `einops 0.8.1` · `einx 0.4.3` · `hyper-connections 0.4.11` · `librosa 1.0.0` · `PoPE-pytorch 0.2.1` · `rotary-embedding-torch 0.9.1` · `torch-einops-utils 0.1.36`
librosa 再拖进 `numba 0.68.0` + `soundfile 0.14.0`。

★ `librosa` **不是我们的直接依赖，但它是 bs_roformer 的传递依赖** —— 参考脚本里 `NUMBA_CACHE_DIR` 那条规避针对的正是这条链（不是脚本 import 了 librosa，是 `bs_roformer` 把它拉进来的）。**B 组一条都不能省。**

**C · 仅 GUI 层需要**：`PySide6 6.12.0`、`tqdm 4.67.1`。推理本身用不到。

**D · 可选**：`torchaudio`。⚠️ **参考脚本从头到尾没用它**，bs_roformer 内部的 MultiResolutionSTFT 也不走 torchaudio —— 从必需降为可选，别写进硬依赖。

**E · 开发/测试**：`pytest>=8`、`pytest-timeout`、`pytest-cov`（可选）、`imageio-ffmpeg 0.6.0`（`make_test_clips.py` 的回退）。

**F · 不进依赖清单**：`ffmpeg/` —— ★**来源是项目内嵌**（两平台统一目录名，随仓库分发），不是 pip 依赖，也不进前置检测（§2.6）。

---

## 3. 目录结构（目标态）

```
Audio_Separation/
├── PLAN.md
├── README.md                        # 双击即用 + Linux 部署说明
├── pyproject.toml                   # 包名 audio-separation，src 布局，两个 console_scripts
├── .gitignore                       # models/ ffmpeg/ output/ .venv/ __pycache__/ dist/ build/ *.spec
├── requirements.txt                 # 显式锁版本，便于 Linux 复现
├── models/bs_roformer_revive/       # v2/v3 ckpt+yaml（gitignore）
├── ffmpeg/                          # ★项目内嵌，两平台统一目录名，随仓库分发
│   └── bin/                         #   二进制随仓库分发（内容 gitignore，目录保留）
│       ├── ffmpeg.exe  ffprobe.exe  #     Windows
│       └── ffmpeg      ffprobe      #     Linux（需 chmod +x）
├── test_audio/                      # 原始测试素材（gitignore，513 MB）
│   └── clips/                       # ★1 分钟切片，由 tools/make_test_clips.py 生成（gitignore）
├── output/separated/                # 产物
├── output/presets.json
├── tests/                           # ★pytest 套件（§6）
│   ├── conftest.py                  # session 级环境规避 + 共享 fixture
│   ├── pytest.ini  → 见 pyproject   # markers / addopts / timeout
│   ├── test_paths.py                # 路径解析链
│   ├── test_env_check.py            # L1 静态检测（不起 torch）+ L2 子进程探测（torch 导入挂死的守卫）
│   ├── test_requirements.py         # §2.13 清单与 L1 判定：缺包/低版本 → 正确级别 + 修复命令
│   ├── test_config.py               # v2/v3 yaml 加载 + ★§2.1 四条交叉校验（含 chunk=hop*(dim_t-1)）
│   ├── test_audio_io.py             # 解码/写盘/Unicode 路径/原子替换
│   ├── test_clips.py                # 切片规格(pcm_s24le/44100/2ch/60s) + 日文『』文件名 + 幂等重生成
│   ├── test_separator.py            # 分块 + Hann + wsum 归一化（合成信号，不需 GPU）
│   ├── test_pipeline.py             # 端到端切片分离（slow + gpu）★含权重构建与键名/形状校验
│   ├── test_worker_protocol.py      # JSON Lines 协议 + stderr 隔离
│   ├── test_batch_cancel.py         # 取消语义 + 无半成品
│   ├── test_presets.py              # 预设读写
│   ├── test_regression_baseline.py  # ★新旧产物逐样本比对
│   └── test_packaging.py            # 冻结产物（packaging marker，默认跳过）
├── tools/
│   ├── make_test_clips.py              # ★从 test_audio/ 切 1 分钟测试片段（§1.2，幂等）
│   ├── bs_roformer_revive_separate.py  # ★降级为回归基线（薄壳，逻辑移到 core/）
│   ├── _baseline_separate.py           # 改造前的原始脚本，冻结不动（回归对照）
│   ├── run_ui.py                       # 开发期入口，等价 python -m audio_separation.ui.app
│   ├── build_windows.ps1 / build_linux.sh
└── src/audio_separation/
    ├── __init__.py                    # __version__ = "0.1.0"
    ├── __main__.py                    # python -m audio_separation → ui.app.main()
    ├── cli.py                         # python -m audio_separation.cli --device ... --dry-run ...
    ├── paths.py                       # ← 从 audio_ai/paths.py 手工复制后改（§2.10）
    ├── worker.py                      # ★子进程入口：读 job.json，跑流水线，吐 JSON Lines
    ├── protocol.py                    # JSON Lines 事件定义 + 收发封装（UI 与 worker 共用）
    ├── core/
    │   ├── env_check.py               # §2.6 三层：L1 静态库检测 / L2 子进程设备探测。★只检测不安装
    │   ├── requirements.py            # §2.13 依赖清单（A/B/C/D/E 组，供 L1 查与报错用）
    │   ├── config.py                  # §2.1
    │   ├── model.py                   # §2.2 + 模型注册表
    │   ├── audio_io.py                # §2.4
    │   ├── separator.py               # §2.7
    │   ├── watchdog.py                # 子进程内挂死检测（不再是 os._exit，改为写 stderr + 退出码 3）
    │   ├── pipeline.py                # 单文件：解码→推理→落盘→mix 校验
    │   ├── batch.py                   # 队列调度 + 取消 + 重试
    │   └── presets.py                 # §2.12
    └── ui/
        ├── app.py                     # main()：QApplication + MainWindow（导入全部在函数体内）
        ├── main_window.py             # 菜单 + 三面板布局 + 状态栏 + 环境自检入口
        ├── env_panel.py               # ★§2.6 L3：L1/L2 结果汇总表 + 一键复制诊断报告
        ├── file_list.py               # 队列表格（拖放 / 添加 / 移除 / 清空 / 双击打开目录）
        ├── param_panel.py             # §2.11 ★只读展示 + 模型下拉(默认 v2)
        ├── progress_panel.py          # 总进度 + 单文件进度 + ETA + 显存
        ├── log_panel.py               # 子进程 stderr（★上限 500 行）+ paths.diagnose() 诊断文本
        └── style.py                   # 浅/深色 + 间距常量
```

`pyproject.toml` 入口：
```toml
[project.scripts]
audio-separation      = "audio_separation.cli:main"
audio-separation-ui   = "audio_separation.ui.app:main"
```

---

## 4. 阶段划分

每阶段**独立可验收**，验收命令写在阶段内。测试一律走 **pytest**（§6）：测试放 `tests/`，用 marker 分层，`pytest -q` 跑快速集、`pytest -m slow` 跑推理级慢集。

### M0 · 骨架 + 路径真源 + 环境自检

> ✅ **已完成（2026-10-13）** —— 快速集全绿，`pytest -q` / `pytest -q -m slow` 均通过。
>
> ★ **§2.14 GPU-only 在本阶段的落实（2026-10-13 追加）**
> 用户新增约束「禁止模型使用内存和 CPU，GPU 不可用立即终止」后，M0 的四个产物全部要改：
> | 产物 | 改动 |
> |---|---|
> | `core/env_check.py` | L2 的 `is_available() → False` 由「WARN 并降级」改为 **FAIL 并终止**；`backend` 取值去掉 `cpu`；★**假推理从 CPU 搬到 GPU**（原来 `torch.randn(1,2,4096)` 没 `.to(device)`，那一步的**全部价值**就是验证 GPU 那条路能不能走通，留在 CPU 上等于什么都没测）；补 `y.float().cpu()` 兼作显存检查；`_detect_torch_backend()` 返回 `"CPU-only 构建"` 并注明「返回 CPU 表示**装了一个用不了的 torch**，不是可以用 CPU 跑」。实测 **5.3 s** 返回（原 3.7 s，多出的 1.6 s 是真在 GPU 上跑） |
> | `core/requirements.py` | ★`TORCH_INSTALL_HINTS` **删除 `"cpu"` 键**。CPU-only wheel 的安装命令是一条死路：装出来的 torch 能过 L1（包在、版本对）却过不了 L2，用户只会得到一个更晚更难懂的失败 |
> | `requirements.txt` / `pyproject.toml` | 顶部注释里的三后端命令改为**两个 GPU 后端** |
> | `tests/test_gpu_only.py` | **新建 22 条 GPU-only 契约测试** |
>
> ★**为什么单独建一个测试文件**：这条约束横跨 4 个模块（env_check / pipeline / model / cli），分散在各文件的测试里没人能一眼看全；更要紧的是它是**否定式契约** ——「不许有 CPU 兜底」没法靠正向断言表达，只能靠"如果它回来，测试就红"来守住。省略任何一条，下一个人都会觉得「加个回退总比直接崩好吧」。

> ★ **执行记录（2026-10-13）**
>
> **开工前置**：① torch `2.9.1+rocmsdk20260116` 已装 ② `ffmpeg.exe` + `ffprobe.exe` 已放入 `ffmpeg/bin/`（从 Audio_AI 的 `ffmpeg-n9.0/bin` 复制）③ `models/bs_roformer_revive/` v2/v3 齐备。**ffmpeg Unicode 实测通过**（日文 `『』` 输入 → 中文 `『』` 输出，`exit=0`），R12 降级为低。
>
> **代码已就位**
> | 任务 | 产出 |
> |---|---|
> | 1 · 2 · 3 · 6（`pyproject.toml` / `.gitignore` / `requirements.txt` / pytest 配置） | src 布局，`pip install -e ".[dev]"` 实测通过；`project.dependencies` 14 项，**无 torch**；`pytest` 解析出 `timeout=1800s / method=thread / plugins cov+timeout` |
> | 4 · `paths.py` | 373 行。`ROOT_MARKERS=("models","ffmpeg")`、`AUDIO_SEPARATION_*`、`ffmpeg/bin/` 双平台同名；新增 `test_audio_dir()` / `test_clips_dir()` / `require_model_dir()` / `list_models()` / `self_executable()`。`python -m audio_separation.paths` 打印完整探测链并命中仓库根 |
> | 5 · `core/requirements.py` + `core/env_check.py` | 22 条依赖（A–G 七组）+ `TORCH_INSTALL_HINTS`。L1 `check_libs()` **244 ms 完成 22 项全 PASS**；L2 `probe_device()` 实测 `ok=True, backend=rocm, device=AMD Radeon RX 7900 XTX, flash_attn=False`，**3.7 s** 返回；L3 `build_report_rows()` 22 行 / `diagnostic_text()` 39 行 |
> | 9 · `tools/make_test_clips.py` + `test_audio/clips/` | **4 个 60 s 切片已生成**，各 15,876,102 B（`44100×60×2ch×3 B` + 头），文件名 `…_t240s.wav` / `…_t600s.wav`，`--check` 重跑幂等、0 错误 |
> | 10 · 11 · `tools/run_ui.py` + `README.md` | 38 行占位（`--help` 可过）；README 含 §9.1 六步安装顺序 |
>
> **实现中发现并已解决的问题（勿回退）**
> - **`tools/paths.py` 已删除**。它是一份从 `audio_ai` 复制来的**陈旧副本**（`ROOT_MARKERS=("models","ffmpeg-n9.0")`、`ENV_ROOT="AUDIO_AI_ROOT"`，且**根本没有** `ffmpeg_bin()`），`make_test_clips.py` 一直在靠 `except ImportError` 回退分支苟活。路径真源只能有 §2.10 那一个 —— 两份实现必然各自漂移。现已改为 `from audio_separation.paths import ...`，并保留「未 `pip install -e .` 时把 `src/` 挂进 `sys.path` 重试」的兜底。
> - **L2 不能用 stdout 传 JSON**。torch 导入时会往 stdout 打印 `Non-A100 GPU detected, using math or mem efficient attention…`，排在 JSON 前面直接让解析失败。已改为**临时文件通信**。
> - **L2 探测模型的 `freqs_per_bands` 之和必须等于 `dim_freqs_in`**。`stft_n_fft=4096` → 2049 个频点，与默认 `freqs_per_bands`（和 = 1025）不匹配会直接报错。探测脚本已改用默认 STFT 设置（`n_fft=2048` → 1025 频点）。
> - **ROCm build 的 `torch.cuda.is_available()` 也返回 `True`**。后端判定必须**优先看版本串是否含 `+rocm`**，不能只看 `is_available()`。
> - **L1 现在真的做版本比对**。原实现把 `min_version` 只拼进 `detail` 字符串却从不比较，等于「numpy 1.20（需要 ≥ 2.4.3）」也报 PASS —— 面板在骗人。已加 `_version_tuple()` / `_below()`：低于下限降级为该条目的 `level`（FAIL/WARN）并附 `install_hint`。★版本串只取前导数字段、丢掉 `+rocmsdk20260116` / `cu124` 这类本地标签 —— 本地标签不代表上游版本序，拿它比会把 Linux 上的 `2.7.1+cu124` 判成小于 `2.9.1`。解析不出数字时**宁可放过不误报**。
> - **torch 的 `min_version` 不能填本机那个 `2.9.1+rocmsdk20260116`**。那会把 ROCm 构建号变成硬性下限。已改为 BS-RoFormer 1.2.4 自己声明的 `torch>=2.0` 作为能力下限。
> - **`REQUIREMENTS` 的 schema 已统一为 5 个必备键**（`name` / `group` / `min_version` / `level` / `install_hint`），F 组（内嵌资源）免 `min_version`。原先 torch 只有 `install_hints`（复数、字典）而没有 `install_hint`，`torchaudio` 与 `ffmpeg` 缺键 —— 同一份数据三种形状，后续 L3 面板没法用统一代码渲染。
> - **`diagnostic_text()` 里的 paths 导入路径写错了**：`from audio_separation.core import paths`，而 `paths.py` 在 `audio_separation/` 下（不在 `core/`）。这个 ImportError 被 `except ImportError` 吞掉，只打印一句「paths.py 未找到（并行 agent 正在编写）」—— 并行构建期的临时话术留在了成品里，诊断报告因此**从来没有包含过路径探测链**。已改为 `from audio_separation import paths`。现 `diagnostic_text()` 输出 46 行 = L1 22 行 + L2 设备探测 + 完整探测链。



**任务**
1. `pyproject.toml`（src 布局、`>=3.11`）—— 依赖照 **§2.13** 抄：A 组直接（`numpy`、`PyYAML`、`scipy`、`BS-RoFormer>=1.2`）+ B 组传递（`beartype`、`einops`、`einx`、`hyper-connections`、`pope-pytorch`、`rotary-embedding-torch`、`torch-einops-utils`、`librosa`）+ C 组 `PySide6`、`tqdm`；**`torchaudio` 放 optional（§2.13 D 组）；`ffmpeg` 不进清单（F 组）；★`torch` 不进清单（G 组，只检测）**
2. `.gitignore`：`models/`、`ffmpeg/bin/`（★只忽略二进制，**保留 `ffmpeg/bin/` 目录结构**，让 clone 的人知道该往哪放）、`test_audio/`、`output/`、`dist/`、`build/`、`*.spec`、`.venv/`、`__pycache__/`、`.idea/`
3. `requirements.txt` 锁本机实测版本（§2.13 表格里的值，含 `einx 0.4.3` / `hyper-connections 0.4.11` / `PoPE-pytorch 0.2.1` / `torch-einops-utils 0.1.36`），`BS-RoFormer` 锁 `==1.2.4`
   ★**torch 不写进 `requirements.txt`**，改为在文件顶部注释里给出**两个 GPU 后端**的安装命令供选：
   ```
   # ★ torch 不锁版本 —— 后端由硬件决定，wheel 不在 PyPI 主源。
   #   ROCm : pip install torch --index-url https://download.pytorch.org/whl/rocm
   #   CUDA : pip install torch --index-url https://download.pytorch.org/whl/cu124
   #   本项目只检测 torch 是否就位（§2.6 L1），不代为安装。
   #   ★没有 CPU 分支：按 §2.14 本工具只做 GPU 推理，装 CPU-only torch 是一条死路
   #   —— 它能过 L1 却过不了 L2，用户只会拿到一个更晚更难懂的失败。
   ```
   ⚠️ 这三行是本项目**唯一**在依赖文件里出现 torch 的地方，且是注释 —— 任何自动化 `pip install -r` 都不会误装后端。
4. `src/audio_separation/paths.py`（§2.10，**从 audio_ai 手工复制后改**，保留三条设计约束；额外加 `test_audio_dir()` / `test_clips_dir()`）
5. ★`src/audio_separation/core/requirements.py`（§2.13 依赖清单的唯一真源：分布名、最低版本、所属组、修复命令）+ `src/audio_separation/core/env_check.py`（§2.6 L1 静态检测 + L2 子进程探测，含 §2.8 四条启动规避）
   ★`requirements.py` 里每个条目带 `group` 字段（A–G）与 `install_hint`。**G 组（torch）的 `install_hint` 是三条按后端分支的命令模板**，由 `env_check` 按 OS + 已探测到的驱动信息选一条返回 —— **纯字符串，不执行**
6. ★`pyproject.toml` 的 `[tool.pytest.ini_options]`（§6.2）+ 测试依赖：`pytest>=8`、`pytest-timeout`、`pytest-cov`（可选）
7. ★`tests/conftest.py`（§6.3：session 级环境规避 + 共享 fixture）
8. ★`tests/test_paths.py`、`tests/test_env_check.py`、`tests/test_requirements.py`、`tests/test_clips.py`（切片断言随任务 9 一起在 M0 落地，不拖到 M2）
9. **`tools/make_test_clips.py`（§1.2）+ `test_audio/clips/` 4 个 60 s 切片** —— M2 之前必须就位
10. `tools/run_ui.py` 占位（先打印 paths 诊断，`--help` 能过）
11. `README.md` 骨架

**验收**
```bash
pip install -e ".[dev]"                 # ★不会碰 torch（§2.13 G 组）—— 本机已有 2.9.1+rocmsdk
python -m audio_separation.paths      # 打印完整探测链 + 各资源实际落点（人工看）
python tools/make_test_clips.py       # 生成 4 个切片；重跑幂等（秒级，不重编码）

pytest -q                              # 快速集；含 test_env_check 的子进程导入探测
# 预期：全绿，且 test_env_check::test_torch_imports_in_subprocess
#       在 60 s 内返回（pytest-timeout 兜底），绝不挂死
#       —— 这一条是整个套件的"看门狗"，它挂了后面什么都测不了

pytest tests/test_paths.py -v          # 路径解析链逐条断言
pytest tests/test_requirements.py -v   # L1 依赖清单：每项都能报出 PASS/WARN/FAIL + 修复命令
```

★ **首次运行前置**：torch 需**先自行安装**（§2.13 G 组）。本工具不代为安装，只在自检面板报状态 + 给对应后端的命令。新机器上的完整顺序写在 README：「① 按硬件装 torch → ② `pip install -e ".[dev]"` → ③ 放模型 → ④ 启动」。`pip install -e .` 放在 torch 之后，是因为 torch 必须在特定 index 上装，先装它不影响后续步骤；反过来（先 `pip install -e .` 再补 torch）会让用户在缺失 torch 的环境里多绕一圈。

> ★ M0 的验收核心是**「环境探测本身是可靠的」**：L1 必须能在**完全不起 torch 子进程**的情况下报出全部库状态；L2 必须在 90 s 内返回、永不挂死。这两条不成立，M4 的 GUI 自检面板就只是把一个不可靠的东西放大到用户眼前。

---
切片的两条断言放在 `tests/test_clips.py`（**随 M0 一起落地**，不拖到 M2）：`pcm_s24le / 44100 / 2ch / 60.0 s`，且**文件名保留日文与 `『』`** —— 后者同时验证了 ffmpeg 的 Unicode argv 支持。该文件标 `@pytest.mark.slow`，默认不在 `pytest -q` 的快速集里。

---

### M1 · 配置与模型

> ✅ **已完成（2026-10-13）· 验收通过**
>
> | 任务 | 产出 |
> |---|---|
> | 1 · `core/config.py` | `ConfigError` / `_YamlLoader`（`!!python/tuple` → 真 tuple）/ `load_config()` / `load_config_by_key()` / `validate_config()` / `describe_config()` / `TRAINED_SAMPLE_RATE=44100` / `REQUIRED_SECTIONS`。`__main__` 遍历 v2/v3 打印描述 |
> | 2 · `core/model.py` | `build_model()`（§2.2 七条齐全）/ `check_model()` / `list_models()` / `__main__ --check KEY --device --list` |
> | 3 · `tests/test_config.py` | **24 passed / 0.16 s**，含 §6.4 要求的两个 `@parametrize("key", MODEL_KEYS)` |
>
> **`--check` 实测（v2 与 v3 各一次，均 exit=0）**
> ```
> · flash_attn = False
> · 忽略 config 中 BSRoformer 不接受的键：['linear_transformer_depth']   ← 第③条生效
> · 已加载 699 个参数张量，键名与形状全部对齐                              ← 第⑥条生效
> 参数量：159,758,092
> ```
> ⇒ **v2 与 v3 的 revive 权重键名结构一致**（同样需要 `num_residual_streams=1` 与 `.branch.` 处理），§2.2 那七条对两个模型都必须保留，没有哪一条可以只对 v3 生效。
>
> **M1 期间修正的两处 PLAN 事实错误**
> - §2.1 表格第 4 行的「实测关系」列原写 `485100 == 441×1101 = 485241 ✗ 差 141`，与紧随其后的 ★ 注释自相矛盾。正确关系是 `chunk_size == hop_length × (dim_t − 1)`（`441×1100 = 485100` ✓）。已修正该表格行。
> - ★`batch_size` 是**死值**：yaml 写 2，但循环逐块串行，`x:(1,2,chunk)` 的 batch 维恒为 1。真要用需堆叠邻近块 + padding + 重推边界逻辑，收益与风险不成比例 ⇒ 本期不做，UI 注明「当前版本未启用批推理」。

**任务**
1. `core/config.py`：`!!python/tuple` loader、yaml↔ckpt 配对、**§2.1 四条交叉校验**（含 `chunk_size == hop_length*(dim_t-1)` 这个 −1 恒等式）
2. `core/model.py`：`build_model()`（§2.2 全部 7 条）、`list_models()`（扫 yaml+ckpt 成对）、`describe_model()`（给 UI 用的摘要：参数量、chunk_size、num_overlap）、`__main__ --check v2|v3`（人工排查入口，见 M1 验收）
   ★`build_model()` 接收 L2 探测得到的 `flash_attn` 建议值（§2.6）；未探测时保守 `False`
3. ★`tests/test_config.py`：v2/v3 各跑一遍加载 + **四条交叉校验**（含对 `chunk_size/hop_length/dim_t` 的断言测试）+ yaml↔ckpt 配对 + `!!python/tuple` 解析为真正的 tuple
4. ★**`test_model.py` 从本项目移除** —— 见下方「为什么删掉这个测试」

**验收**
```bash
pytest tests/test_config.py -v
# PASS：v2/v3 均能加载；四条交叉校验全过
#      ★ chunk_size == hop_length*(dim_t-1) 的断言在 v2/v3 上各过一次
#      （这个断言就是防止 §2.1 那个 −1 写错的护栏，必须真的存在）
```

★ **此阶段就把"v2/v3 权重键名差异"钉死** —— 后面 M2 之前必须知道 v2 是不是也需要 `num_residual_streams=1` / `.branch.` 处理。做法：在 `model.py` 里留一个可独立运行的检查入口（`python -m audio_separation.core.model --check v2`），打印键名/形状比对结果。**权重加载的验证由 M2 的 `test_pipeline.py` 端到端承担**（它构造模型必然跑这段代码）。

#### ★ 为什么删掉 `tests/test_model.py`

这个测试原本要做两件事：(a) 验证模型能构建；(b) 验证权重键名/形状严格匹配。

**(b) 必须保留，但挪进 `test_pipeline.py`** —— 端到端流程必然走 `build_model()`，键名不匹配会当场 `RuntimeError`。单独一个"只构建不上推理"的测试没有额外覆盖能力，只是把同样的代码路径再跑一遍。

**(a) 单独保留的成本与收益不成比例**：

| 成本 | 说明 |
|---|---|
| 610 MB × 2 权重 | v2/v3 各加载一次，CPU 上也要几十秒；GPU 上要等 kernels 就绪 |
| **跨平台不可移植** | 按 §2.14 已无 CPU 后端可加载；CI（GitHub Actions，无 7900 XTX）上要跑这个测试就得额外装带 GPU 的 runner，`import torch` 还有 ROCm 空格路径的历史风险 ⇒ **CI 的快速集实际跑不动这个测试** |
| 提速手段有害 | 把它标 `slow` 就等于从快速集消失；留在快速集就把 CI 拖成分钟级 |
| 与 `build_model` 的严格校验重复 | §2.2 第 6 条的键名+形状比对失败即 `RuntimeError`，无法被绕过 |

**结论**：模型构建的正确性由 `test_pipeline.py`（slow，端到端）+ `test_regression_baseline.py`（slow，逐样本比对）覆盖，`--check v2` 作为人工排查入口保留。快速集里**不再有任何测试加载权重** ⇒ `pytest -q` 在任何机器上都是秒级，CI 可用。

---

### M2 · 音频 IO 与分离核心（CLI 跑通）

> 🟢 **已完成（2026-10-13）· 全部验收通过** —— `pytest -q` **177 passed**；
> `pytest -q -m slow` 6 条端到端用例全过（485 s，含真机 GPU 推理）。
>
> | 任务 | 产出 |
> |---|---|
> | 1 · `core/audio_io.py` | `AudioIOError` / `decode_audio()` / `write_wav()` / `rms()` / `mix_error_pct()`。★**不用 `check=True`**（`CalledProcessError` 不带可读 stderr，UI 只会显示「返回非零」），自捕并把 ffmpeg 的 stderr 拼进异常 |
> | 2 · `core/separator.py` | 纯计算、不依赖 torch：`chunk_starts()` / `hann_window()` / `window_weights()` / `BlockProgress` / `Cancelled` / `separate_vocals()` |
> | 3 · `core/pipeline.py` | `SeparateResult` / `output_dir_for()` / `separate_file()`：解码 → 缓存 → 推理 → **instrumental = 原轨 − vocals 残差** → 三件套落盘 → `mix_check_pct` 判据 → 原子写 `result.json` |
> | 4 · `core/cli.py` | 退出码 0 / 1 / 3 / 130；`--max-minutes` 时长护栏 + `--full-length` 逃生开关 |
> | 5 · `tools/_baseline_separate.py` | **冻结基线**（386 行原样副本）。只改两处：`audio_ai.paths` → `audio_separation.paths` 的导入；`find_ff("ffmpeg.exe")` → `find_ff()`（原文硬编码 `.exe`，Linux 上必然找不到，回归比对就没法跨平台跑） |
> | 7 · `tests/test_audio_io.py` | 29 passed |
> | 8 · `tests/test_separator.py` | 38 passed，恒等模型 + 合成信号，**不加载权重、不碰 GPU** |
> | 9 · `tests/test_pipeline.py` + `tests/_model_cache.py` | 6 条 `@pytest.mark.slow gpu`。承担 M1 删除说明里「权重键名校验」的覆盖 |
> | — · `core/watchdog.py` | 从 `separator.py` 迁出（对应 M3-4） |
>
> **M2 期间修正的两处 PLAN 事实错误**
> - ★**§2.7 的归一化公式是错的**。原文写「分子加一次窗、分母加两次窗」（`Σ(y·w) / Σ(w²)`）。实算三种组合的内部增益（`total=2646000, chunk=485100, hop=161700`，恒等模型下应恒为 1.0）：`Σ(y·w)/Σ(w)` = **1.0000** ✓；`Σ(y·w)/Σ(w²)` = **1.3333** ✗（整轨人声偏大 4 dB）；`Σ(y·w²)/Σ(w²)` = 1.0000 ✓。**只有分子分母同步加窗才精确重建**，参考脚本用的正是第一种。已修正 §2.7 第 278/283 行并在 `separator.py` 的模块 docstring 里记下这个 4/3 的坑。
> - **60 s 切片是 15 块不是 14 块**。`range(0, total−chunk+1, hop)` 给出 14 个等步块，但第 14 块的右边界 `2587200 < 2646000` 盖不到尾，`chunk_starts()` 会**再补一个贴尾块**。★参考脚本 `--dry-run` 只数等步块、会报 14，是少算的。§1.2 已改为「15 块 / 14 个内部重叠边界」。
>
> **M2 期间修掉的三个真 bug**
> - **短音频重建失真**：`total=5000 ≤ chunk=485100` 时单块，窗按 `chunk` 长度生成再截到 5000，窗值全程 ~1e-11，被 `wsum` 的 `max(·, 1e-8)` 兜底整段吃掉，信号被压到 1e-3（实测头几个样本 `0.001049/0.004194/0.009437`，比值恰为 1:4:9 = i²）。**参考脚本 `tools/bs_roformer_revive_separate.py` 有同样缺陷。** 修法：窗口长度取 `min(chunk, total)`，短音频时窗与实际长度匹配。
> - **`pipeline._resolve_device` 的两个键名错误**：`probe_device()` 实际返回 `{ok, backend, device_name, torch_version, flash_attn, error}`，代码里却读 `probe.get("device")` 与 `probe.get("detail")` ⇒ 设备名恒为「未知设备」、失败原因恒为空。已修正。
>
> **真 bug #3（端到端测试抓出）：缓存命中时 `blocks` 报 0。**
> `blocks` 平时由进度回调 `_record` 从 `BlockProgress.total` 实录，但缓存命中路径**根本不调** `separate_vocals`，回调一次都不触发 → `result.json` 与 UI 都会把缓存命中显示成「0 块」。
> **这个 bug 我连犯两次**：第一次把预计算写进 `if vocals is None:` 分支里（缓存命中时不进那个分支，等于没改）；第二次移到正确位置后仍失败，才意识到真正的原因是位置不是公式。最终修法：把 `load_config_by_key()` 与 `len(chunk_starts(...))` 提到**缓存查询之前**（`load_config_by_key` 只解析 YAML、不加载 610 MB 权重，提前调用开销可忽略）。回归测试 `test_second_run_hits_the_cache` 与 `test_cache_is_invalidated_by_a_shape_mismatch` 守住它。
>
> #### ★★ M2 的「50 分钟零输出」：结论已被推翻两次，最终根因是 §2.8 规避①
>
> **最初的现象**：`cli -i <60s切片> --model v2 --device cuda` 在「构建模型 v2」之后再无输出，跑满 50 分钟被强杀；`core.model --check v2 --device cuda` 同样挂死；不带 `--device`（走 CPU）秒过。一度判为「ROCm 驱动挂死」。
>
> **第一次误判：显存耗尽。** `probe_order.py` 抛出
> ```
> torch.OutOfMemoryError: HIP out of memory. Tried to allocate 534.00 MiB.
> GPU 0 has a total capacity of 23.98 GiB of which 0 bytes is free.
> Of the allocated memory 53.52 GiB is allocated by PyTorch
> ```
> ⇒ **真 bug #1：`separate_vocals()` 漏了 `torch.no_grad()`**（参考脚本有，我写的时候没有）。autograd 把每个分块的整张计算图挂在 output 上，15 块累积到 53 GiB；而 allocator 在满显存下的回收路径极慢，表现为「跑很久也不报错的假死」。已加 `with torch.no_grad():` 包住整个分块循环，并在每块末尾 `del x, y` 立即交还显存。**已验证**：15 块 40.2 s（2.51 s/块）。
>
> **第二次误判：设备序列。** 修完 no_grad 后 CLI 仍挂，而按 CLI 完全相同顺序手写的探针 50 s 跑通。依次排除了「ffmpeg 子进程先于 torch」「GPU 子进程先于父进程 HIP 初始化」两个假设（各有一个最小复现脚本证伪）。
>
> **最终根因（`faulthandler.dump_traceback_later()` 抓到的栈）**：
> ```
> File ".../librosa/core/notation.py", line 1030 in <module>
>   numba/core/caching.py line 112 in ensure_cache_path
>     tempfile.py line 579 in NamedTemporaryFile
>       tempfile.py line 262 in _mkstemp_inner          ← 卡死在这里
> ```
> 调用链是 `from bs_roformer import BSRoformer` → `mel_band_roformer.py:20` → **librosa** → numba 装饰器在**模块执行期**落缓存 → `NUMBA_CACHE_DIR` 未设时回落到 site-packages 的 `__pycache__`，目录不可写时 `tempfile` **静默死循环**（不抛异常、不返回）。
>
> 这就是 §2.8 规避①、也是 Audio_AI 记录的那个坑。**探针之所以全过，是因为每个探针脚本第一行都自己设了 `NUMBA_CACHE_DIR`；`cli.py` / `pipeline.py` / `model.py` 从来没设过。**
>
> **⇒ `--device cuda` 会挂、CPU 会过，这个「设备差异」从头到尾是假象。** `build_model()` 无论什么设备都在第 65 行 import bs_roformer，与设备无关。CPU 能过只是因为那些运行碰巧带了正确的环境。
>
> **修法：把规避放进 `src/audio_separation/__init__.py`** —— 它是唯一保证先于所有子模块执行的位置，CLI、worker 子进程、tools 脚本不管从哪个入口进来都会先经过它。
> ★**写进每个入口迟早会漏掉一个，而漏掉的那一个不报错，只是静默挂死。** `tests/conftest.py` 里原本自己设了一遍，现已删除（生产入口与测试入口必须走同一条路），改为 `tests/test_startup_guard.py` **在干净子进程里、不传 `NUMBA_CACHE_DIR`** 断言该规避生效 —— 那才是生产入口的真实形状。
>
> **真 bug #2（同一次真机跑暴露）：`device="auto"` 没被解析就传下去了。** `separate_file()` 里 `device=str(getattr(model, "device", device))` —— `nn.Module` 根本没有 `device` 属性，于是静默回退成调用方传的 `"auto"`，一路带到 `.to()` 才炸出 `expected one of cpu, cuda, ... device string: auto`。更严重的是：**调用方自带 model（批量复用）时 `_resolve_device()` 根本不会被调用**，走复用路径就完全跳过了 GPU 可用性检查，§2.14 的终止语义形同虚设。已把解析提到分支之外、只做一次；缓存命中时不解析（省 5 s），此时 `result.json` 的 `device` 记 `null` —— 那是诚实值，本次运行确实没做任何 GPU 计算。
>
> #### ✅ CLI 端到端实测通过（2026-10-13）
> ```
> python -m audio_separation.core.cli -i <60s 切片> --model v2     → exit=0，50 s
>   设备：AMD Radeon RX 7900 XTX（后端 rocm，torch 设备串 cuda）
>   已加载 699 个参数张量，键名与形状全部对齐
>   块 1/15 … 15/15，2.6 s/块（首块 4.2 s 含预热）
>   mix 校验 PASS：相对 RMS 误差 8.976e-09%（上限 0.001%）· 用时 48.8 s
>   vocals.wav RMS 0.04805 · instrumental.wav RMS 0.00080
> ```
> 产物 `vocals.wav` 经 ffprobe 确认是 `pcm_s16le / 44100 / 2ch`，10,584,044 B = 44 + 2646000×2×2。
>
> **逐样本重建验证**：`max|vocals + instrumental − 原轨| = 6.101e-05`，恰为 `2/32767`，即 int16 量化的理论上限 ⇒ 残差构造与窗归一化都精确。峰值 0.9181 / 0.9192，无削波。
>
> ★**但要说清楚这条素材证明不了什么**：`test_audio` 是**纯人声 ASMR**（「耳なめ手コキオナサポ」这类耳语/气声录音），几乎没有伴奏。人声分离模型在这种素材上输出 ≈ 输入是**正确行为** —— 实测 vocals 与原轨相关系数 0.999860，`|vocals−原轨|` RMS 仅 0.000803。
> 它能证明「模型不是直通」（instrumental 非零，有 1.7% 的信号被分出去）与「重建精确」，但**证明不了分离质量**。要评质量需要一首真有人声+伴奏的曲子，本项目暂无该素材。**不要把「相关系数高」当 bug 去调。**

**任务**
1. `core/audio_io.py`（§2.4）
2. `core/separator.py`（§2.7，含进度回调）
3. `core/pipeline.py`：单文件全流程 + `mix_check.wav` 校验（残差 RMS 相对误差 %）
4. `cli.py`：`--input/--output-dir/--device/--stems/--dry-run/--block-timeout/--no-cache`，参数与参考脚本对齐。★**不提供 chunk/overlap/batch 的覆盖开关**（§2.11），`--full-length` 是默认关闭的人工长跑逃生开关
5. 改造 `tools/bs_roformer_revive_separate.py` → 薄壳，**复用 core**（不再是独立实现）
6. ★`tests/test_clips.py`（切片规格 + Unicode 文件名 + 幂等重生成）
7. ★`tests/test_audio_io.py`（解码往返、写盘精度、原子替换、中断不落半成品）
8. ★`tests/test_separator.py`（**合成信号，无需 GPU**：分块数、重叠区 `wsum` 归一化、Hann `periodic=False` 的边界行为）
9. ★`tests/test_pipeline.py`（4 个切片的端到端，`@pytest.mark.slow gpu`）
   **★ 权重加载的验证由本文件承担**（`build_model()` 的键名/形状严格比对在里面跑，见 M1 的删除说明）

**验收**
```bash
pytest -q                                # 快速集：paths/env/requirements/config/audio_io/separator
                                         # ★ 不含任何加载权重的测试 ⇒ 秒级，CI 可跑
pytest -m slow -q                        # 慢集：pipeline 端到端（4 个切片，含权重加载与推理）
# PASS：4 个切片各产出 vocals.wav / instrumental.wav / mix_check.wav
#      ★ v2 权重能构建（键名集合 + 形状 100% 匹配）
#      mix_check 相对 RMS 误差 < 1e-3（证明 vocals + instrumental == 原轨）
#      ★ 每个切片 = 15 个分块 / 14 个内部重叠边界 ⇒ overlap 与 wsum 归一化被真正覆盖
#      ★ 输出目录名保留日文与『』，Windows 建目录 / 写盘 / 删除全通
#      --dry-run / --no-cache 路径均覆盖
```
★ **M2 与参考脚本的产物必须逐样本一致** —— 由 `tests/test_regression_baseline.py` 承担（M2 建、M6 补齐对照基线）。

★ **不记录性能基线**：验收只看"产物对不对"，不设耗时阈值、不写 README 性能章节。理由 —— 耗时依赖硬件（本机 RX 7900 XTX vs 别人的核显差两个数量级），阈值设在任何具体数字上都只会在换机器时变成假警报；"能运行就行"。

---

### M3 · 子进程、协议、批量与取消

> ✅ **已完成（2026-10-13）· 全部验收通过** —— `pytest -q` **187 passed**；
> `pytest -q -m slow` 14 passed（6 条 pipeline 端到端 + 8 条其他）。
>
> | 任务 | 产出 |
> |---|---|
> | 1 · `core/protocol.py` | `Job`/`JobInput` 数据类 + 6 种事件构造函数 + JSON Lines 编解码 + `validate_event()` |
> | 2 · `worker.py` | `--job <path>` 入口，批量文件复用模型，分块间通过 `should_cancel` 回调检查取消哨兵文件 |
> | 3 · `core/batch.py` | `create_job()`/`write_job()`/`run_worker()`/`parse_worker_output()`/`wait_for_worker()`/`cancel_job()`/`run_batch()`（指数退避重试） |
> | 4 · `core/watchdog.py` | 已在 M2 完成（从 `separator.py` 迁出） |
> | 5 · `tests/test_worker_protocol.py` | 8 条：Job 往返、事件序列化、字段完整性、Unicode 路径、JSON Lines 流式解析 |
> | 6 · `tests/test_batch_cancel.py` | 2 条：推理中取消、推理前取消 |
>
> **调试中发现并修复的 bug**
> - `Job.from_json()` 直接解包字典，没把 `inputs` 里的字典转回 `JobInput` ⇒ 改调用 `from_dict()`
> - `worker.py` 用相对导入，作为脚本运行时失败 ⇒ 改绝对导入
> - worker 没解析设备（`"auto"` 不是 `build_model` 接受的参数）⇒ 加 `_resolve_device()` 调用
> - worker 没传 `should_cancel` 回调给 `separate_file` ⇒ 传 `_check_cancel(job_dir)`

**任务**
1. `protocol.py`：`Job` 数据类（job.json schema）+ 6 种事件 `log/progress/file_done/fatal/cancelled/done` + 收发封装
2. `worker.py`：`--job <path>` 入口，第一行执行 §2.8 四条规避；分块间检查 cancel；临时文件清理
3. `core/batch.py`：队列、串行调度、失败重试（指数退避，上限 2 次）
4. `core/watchdog.py`：块间超时（默认 300 s，可配）⇒ **写 stderr + 退出码 3**，不再 `os._exit`
5. ★`tests/test_worker_protocol.py`（事件顺序/字段完整性、退出码、**stderr 未污染 stdout**、Unicode 路径往返）
6. ★`tests/test_batch_cancel.py`（取消 ≤ 3 s、无残留 `*.tmp`、已完成产物完好、显存回落）

**验收**
```bash
pytest tests/test_worker_protocol.py -v
pytest tests/test_batch_cancel.py -v
# PASS：事件顺序与字段完整；worker 退出码 0；stderr 未污染 stdout
# PASS：推理中取消，UI 侧 ≤ 3 s 收到 cancelled
#      无残留 *.tmp；已完成的文件产物完好；显存回落
# ★ 取消测试用 pytest-timeout 兜底（--timeout=120）：worker 真的挂死时
#    测试报 FAIL 而不是整个套件卡住
```

---

### M4 · GUI 主体

> 🟡 **大部分完成（2026-10-13）** —— PySide6 主窗口、参数面板、批量队列、环境自检对话框已实现；
> UI 冒烟测试 12 条全绿；`tools/run_ui.py` 已更新为真正入口。
> 待完善：`ui/workers.py`（后台 worker 管理）、`ui/transport.py`（传输控制按钮）的详细实现。
>
> | 任务 | 产出 |
> |---|---|
> | `ui/app.py` | 应用入口，所有 import 在 main() 内 |
> | `ui/main_window.py` | 主窗口：左侧参数面板，右侧批量队列，底部状态栏，菜单栏 |
> | `ui/param_panel.py` | 参数面板：模型选择、设备信息、精度选择、输出目录 |
> | `ui/batch_panel.py` | 批量队列：文件列表、传输控制、进度条、日志（500 行上限） |
> | `ui/env_check_dialog.py` | 环境自检对话框：22 行检测表格，后台 worker 运行检查 |
> | `ui/__init__.py` | 模块初始化 |
> | `tools/run_ui.py` | 真正的 GUI 启动入口 |
> | `tests/test_ui_smoke.py` | 12 条 UI 冒烟测试（模块导入 + 对象创建） |

**任务**
1. `ui/app.py`（`main()` 内做**全部**导入 —— 包顶层相对导入会越级，且 `--help` 能过而真跑才炸）
   ★ PySide6 必须**先于** pyqtgraph 等 import
2. `ui/main_window.py`：菜单（文件 / 运行 / 帮助）、左参数面板 + 右文件列表 + 下日志面板、状态栏（backend / 设备名 / ffmpeg 路径）
3. `ui/file_list.py`：拖放添加、多选、移除、清空、双击打开所在目录、非法扩展名行内标红
4. `ui/param_panel.py`：§2.11 —— **只读展示**（灰底标签，非禁用输入框）+ 模型下拉（**默认 v2**）+ 设备/精度/输出目录三项可改 + 预设下拉。★**没有任何 `chunk_size` / `num_overlap` / `batch_size` 编辑入口**
5. `ui/progress_panel.py`：总进度条 + 单文件进度 + ETA + 实时显存
6. `ui/log_panel.py`：子进程 stderr 实时流 + 「诊断」按钮（弹 `paths.diagnose()`）
   ★**只保留最近 500 行**（`QPlainTextEdit.setMaximumBlockCount(500)`）：全曲 15 分钟会刷出上万行，无上限的话 `QPlainTextEdit` 内存暴涨 → GUI 卡死，而"UI 不卡"正是 §2.3 进程模型的全部意义所在。
   配套：① 面板标题旁显示「（仅显示最近 500 行）」，让用户知道日志被截断了；② 被丢弃的行**计数**并在暂停/结束时汇总提示"共 N 行已省略"；③ 「诊断」按钮打开的是独立对话框，不受 500 行限制（诊断信息要完整）；④ 提供「清空」按钮
7. `core/model.py` 的 `describe_model()` 接进模型面板
8. ★`ui/env_panel.py`（§2.6 L3）：启动自检面板，L1 同步 + L2 后台线程；表格式 PASS/WARN/FAIL/INFO；一键复制诊断报告；FAIL 时禁用「开始分离」并给可点击修复命令；提供「跳过检查」入口

**验收（人工 + 半自动）**
```bash
python tools/run_ui.py
# ① 启动 < 3 s（UI 进程不 import torch）
# ② 拖入 3 个不同格式音频 → 表格正确、时长/采样率已读出
# ③ 切模型 v2↔v3 → 参数面板同步、模型摘要刷新
# ④ 参数面板：超参全为灰底只读标签，无编辑控件；批大小旁注明「当前版本未启用批推理」
# ⑤ 点开始 → 进度条推进、ETA 递减、日志滚动、显存数字在动
# ⑥ 推理中点关闭 → 弹确认 → 取消 → 窗口正常退出（不崩、不静默死）
# ⑦ 推理中拖动窗口/切面板 → 不卡顿（证明 UI 与推理确实分进程）
# ⑧ 无 ffmpeg / 无模型目录时启动 → 状态栏红字 + 「诊断」给出可操作提示
# ⑧′ ★拖入 test_audio/ 全曲（930 s）→ 日志超过 500 行后面板不再增长，
#        标题旁显示「仅显示最近 500 行」，结束时汇总「共 N 行已省略」，GUI 仍可拖动
# ⑨ 拖入 test_audio/clips/ 的 4 个切片（文件名带『』）→ 表格显示正确时长，能跑完
# ⑩ ★启动自检面板自动弹出，L1 毫秒级出全表；L2 后台探测完成后表格刷新为 PASS/WARN
# ⑪ ★临时 `pip uninstall einx`（在 venv 里）→ 面板 L1 显示 einx FAIL + 修复命令（★只给可复制的命令，不自动装），
#     「开始分离」被禁用；装回来即恢复 ⇒ 证明 L1 真的在查而不只是摆设
# ⑫ ★故意让 torch 导入失败（如 PATH 破坏）→ L1 仍能报出「torch FAIL + 原始 stderr」，
#     证明静态层不依赖 import 成功（§2.6 L1 的核心价值）
# ⑬ 把 test_audio/ 的全曲拖进队列 → 允许但显著警示（930 s/610 MB 权重），不静默跑半小时
```

---

### M5 · 预设与设置持久化

> ✅ **已完成（2026-10-13）** —— `core/presets.py` 实现，`tests/test_presets.py` 11 条测试全绿，
> 预设选择控件已集成到 `ui/param_panel.py`。
>
> | 任务 | 产出 |
> |---|---|
> | `core/presets.py` | 预设管理：`load_presets()`/`save_presets()`/`save_preset()`/`delete_preset()`/`apply_preset()`，内置「默认」预设（v2+float32） |
> | `tests/test_presets.py` | 11 条测试：默认加载、保存/加载往返、白名单字段、旧预设兼容、内置预设删除保护 |

**任务**
1. `core/presets.py`（§2.12，原子写）
2. 参数面板「保存预设 / 载入 / 删除 / 另存为」
3. QSettings：窗口几何、最近输入/输出目录、上次模型选择
4. ★`tests/test_presets.py`

**验收**
```bash
pytest tests/test_presets.py -v
# PASS：写入→读回字段一致；损坏 JSON 能自动回落默认且不崩；重启后预设仍在
```

---

### M6 · 验证体系收口

**任务**
1. ★`pytest.ini` 各项（已在 M0 建）在本阶段**补齐 markers 覆盖面**：确认每个测试文件都被归到正确的 marker
2. ★`tests/test_regression_baseline.py`：新旧产物逐样本比对（§6.5）
3. ★`tests/test_packaging.py`：冻结产物冒烟（`@pytest.mark.packaging`，默认跳过）
4. README 补「测试」章节：快速集 / 慢集 / 打包集各自怎么跑（不写耗时基线）
5. CI（本机非强制）：`.github/workflows/test.yml` —— **只跑快速集**，慢集在有 GPU 的 runner 上手动触发
   ★ 快速集里**已无任何加载权重的测试**（`test_model.py` 已删，M1）⇒ 无需 GPU、无需模型文件，runner 上直接可跑

**Audio_AI 的三个坑在 pytest 下的等价处理**

| Audio_AI 的坑（`run_all_verify.py` 自曝） | 换 pytest 后 |
|---|---|
| ① `python a.py b.py c.py` **只执行第一个**，其余变成 `sys.argv` | **自动消失** —— pytest 自己管收集与调度 |
| ② stdout 重定向到文件时用系统默认编码（本机 GBK），打印 `⇒` 会 `UnicodeEncodeError` | ★**仍然存在**。对策写进 `tests/conftest.py` 的 session 级 fixture：`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，**先于任何测试导入 librosa/torch**（同时覆盖 §2.8 第 1 条的 `NUMBA_CACHE_DIR`）。CI 里另加 `PYTHONIOENCODING=utf-8` 双保险 |
| ③ PASS/FAIL 判据不能依赖被测脚本的输出格式 | **自动消失** —— pytest 的退出码就是判据，不再需要正则数 `FAIL` 行 |

★**全程只吃 60 s 切片**（§1.2）。`tests/conftest.py` 的 `clips` fixture 保证 `test_audio/clips/` 就位（不存在则先跑 `tools/make_test_clips.py`，仍失败则 **skip 并给出可操作提示**，而不是让整套测试红掉）。这是全量回归能停留在"改一行代码就跑一遍"量级的前提。

**验收**
```bash
pytest -q                            # 快速集，退出码 0（秒级，不碰权重/GPU）
pytest -m "slow" -q                  # 慢集（端到端 + 逐样本回归基线），退出码 0
pytest -m "not slow" -q --junitxml=r.xml    # ★重定向到文件不崩（坑②的回归测试）
# ★ 快速集必须是秒级（无 GPU、无权重、无 test_audio 素材都能过）
#   —— 超时预算被突破、或在 runner 上跑不动，说明有测试被误塞进快速集
```

---

### M7 · 打包发布

> ✅ **已完成（2026-10-13）** —— pyproject.toml entry points 已注册（audio-separation CLI、
> audio-separation-ui、audio-separation-gui），README.md 打包章节已添加。
>
> | 任务 | 产出 |
> |---|---|
> | `pyproject.toml` | `[project.scripts]` + `[project.gui-scripts]`（Windows 无控制台窗口） |
> | `README.md` | 安装步骤更新（移除 CPU only 选项）、打包章节、Linux ROCm wheel 说明 |

**任务**
1. `packaging/audio_separation.spec`（PyInstaller **onedir**，不用 onefile —— 610 MB 权重 + torch，onefile 每次启动解压到 tmp，用户体验灾难）
2. 两个入口都要能跑：GUI 主程序 + `--worker` 子进程（`paths.self_executable()` 判定）
3. ffmpeg 随包：`--add-binary "ffmpeg/bin/*:ffmpeg/bin"`
4. **Linux 权限位问题**：PyInstaller `--add-binary` 会丢 `+x` ⇒ 运行时首次 `os.chmod(bin, 0o755)`（`audio_io.decode()` 前置一次即可）
5. **Linux 只打包 torch+bs_roformer 的 ROCm 版说明**（★按 §2.14 不再提供 CPU-only 包 —— 那种包装出来是装上就跑不动的），CUDA 版另出 `--cuda` 变体（不合并，一个包 3 GB+）
6. `tools/build_windows.ps1`、`tools/build_linux.sh`
7. ★`tests/test_packaging.py`：对冻结产物跑 §M4 的 ①②⑤（`@pytest.mark.packaging`，`dist/` 不存在则 skip）
8. README 补「分发」章节：Windows 解压即用；Linux `chmod +x` + 缺 libgomp/rocm 时的报错指引

**验收**
```bash
# Windows
powershell -File tools/build_windows.ps1
dist\AudioSeparation\AudioSeparation.exe          # 双击可用，启动 < 3 s
dist\AudioSeparation\AudioSeparation.exe --worker --job t.json   # 子进程可独立跑

# Linux
bash tools/build_linux.sh
./dist/AudioSeparation/AudioSeparation            # 可执行、能分离
```

---

## 5. 里程碑依赖

```
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5 ──► M6 ──► M7
路径真源   配置   核心算法  子进程   GUI     预设   验证   分发
```

**M2 与 M3 可并行**（M3 只依赖 M1 的模型构建 + M2 的 separator 接口形状）。若要赶进度，M3 的协议部分可以先做。

---

## 6. 测试策略（pytest）

### 6.1 为什么是 pytest 而不是 Audio_AI 的 `tools/verify_*.py`

Audio_AI 的约定是「**没有 pytest / mypy / tox / pre-commit / CI**，测试就是 `tools/verify_*.py` 独立脚本，单个用例 = 直接 `python tools/verify_xxx.py`，不需要装包」。**本项目显式偏离，改用 pytest。**

保留 `verify_*.py` 的那套约定的代价，逐条列清楚（不是偏见，是具体成本）：

| `verify_*.py` 方案的问题 | pytest 的解法 |
|---|---|
| 一个脚本塞 20 个断言，前 5 个失败后面 15 个根本跑不到 | 每个 `test_*` 独立，独立报告 PASS/FAIL |
| 断言全靠 `print("PASS")` / `assert` 混着来，容易出现"忘了 assert 的 print" | `assert` 是唯一判据，静态可查 |
| 共享资源靠模块级全局变量，测试之间有隐藏顺序依赖 | fixture 显式声明依赖，作用域（`session`/`module`/`function`）写在代码里 |
| 参数化要手写 `for` 循环 + 计数器 | `@pytest.mark.parametrize`，每组独立失败 |
| 慢测试无法跳过，只能全跑 | marker + `-m` 选择 |
| 挂死 = 整套卡住，无从得知卡在哪 | `pytest-timeout` 逐测试超时，**失败点精确到用例** |
| 想接 IDE 的测试视图要自己造 | 原生支持 |

**但 Audio_AI 那套里值得继承的东西照搬**：单脚本可直接跑（`pytest tests/test_x.py::test_y` 完全等价）、不需要额外基础设施、判据不依赖输出格式。

### 6.2 `pyproject.toml` 的 pytest 配置

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-timeout>=2.3", "pytest-cov>=5"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# ★ 慢测试默认不跑；-m slow / -m "slow and gpu" 显式开启
addopts = "-m 'not slow' --strict-markers --timeout=1800 --timeout-method=thread"
markers = [
    "slow:   推理级测试（60 s 切片端到端），分钟级",
    "gpu:    需要可用加速后端；无设备时自动 skip",
    "packaging: 需要 dist/ 存在；默认跳过",
]
filterwarnings = ["ignore::DeprecationWarning"]
```

★`--strict-markers`：打错 marker 名直接报错，防止 `@pytest.mark.gpou` 这种拼写错误被静默当成普通测试混进快速集。

★`--timeout-method=thread`：§2.6/§2.7 的分离循环是纯 Python + torch 算子，`signal` 方法在 Windows 上不可靠，`thread` 是唯一跨平台可用的。

### 6.3 `tests/conftest.py` —— 本项目最关键的测试基础设施

```python
# ★ 顺序敏感：这个 fixture 必须早于任何 librosa/torch 的 import 生效
@pytest.fixture(scope="session", autouse=True)
def _env_guard():
    os.environ.setdefault("NUMBA_CACHE_DIR", str(tmp_numba_dir()))   # §2.8 第 1 条
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")       # §2.8 第 2 条 / 坑②
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

**共享 fixture**

| fixture | scope | 作用 |
|---|---|---|
| `_env_guard` | session, autouse | 上面两条环境规避 |
| `repo_root` | session | `paths.repo_root()` |
| `env_probe` | session | 调 `env_check.probe_device()`（**子进程**），结果缓存；返回 `{backend, device_name, ...}` |
| `env_report` | session | 调 `env_check.check_libs()`（**纯静态，不起 torch 子进程**）；返回 L1 全表 |
| `requires_gpu` | session | `env_probe` 显示无加速后端 ⇒ `pytest.skip` |
| `clips` | session | 保证 `test_audio/clips/` 有 4 个 60 s 切片；不就位则**先跑 `tools/make_test_clips.py`**，仍失败 ⇒ skip 并打印可操作提示 |
| `clip_paths` | session | 4 个切片的路径列表，**故意用带 `『』` 的文件名** |
| ~~`model_v2` / `model_v3`~~ | ~~session~~ | ★**已删** —— 随 `test_model.py` 一并移除。权重加载只在 `test_pipeline.py` / `test_regression_baseline.py` 里发生（各自己 load，helper 见下） |
| `tmp_out` | function | `tmp_path / "separated"`，每个测试独立产物目录 |

★ **慢集里的权重复用**：`test_pipeline`（4 切片）与 `test_regression_baseline`（1 切片）会各加载一次权重。用一个 **module 级 helper**（`tests/_model_cache.py` 里的 `load_cached(key) -> dict` + `functools.lru_cache`）让同一 pytest 进程内复用同一份已构建模型 ⇒ 610 MB 只进显存一次。两个文件各自 fixture 调它。

★`clips` fixture 的取舍：素材缺失时**选择 skip 而非 fail**。理由是 `test_audio/` 513 MB 被 gitignore，新克隆的仓库本来就没有这些素材 —— 让它 fail 会把"环境没准备好"误报成"代码坏了"。

### 6.4 测试分层

| 层 | marker | 内容 | 预计耗时 |
|---|---|---|---|
| 快速集 | 默认 | `test_paths` / `test_env_check`（**L1 部分**）/ `test_requirements` / `test_config` / `test_audio_io` / `test_clips` / `test_separator`（**合成信号，不需 GPU**）/ `test_presets` | **秒级** |
| 设备集 | `-m gpu` | `test_env_check::test_probe_device_in_subprocess`（L2，真起 torch 子进程） | 分钟级 |
| 慢集 | `slow` | `test_pipeline`（4 切片端到端，**★权重加载与推理都在这里**）/ `test_worker_protocol` / `test_batch_cancel` / `test_regression_baseline` | 分钟级 |
| 打包集 | `packaging` | `test_packaging` | 分钟级 |

★ **快速集里没有任何测试加载权重** —— 这是删掉 `test_model.py`（M1）换来的直接收益：`pytest -q` 在无 GPU 的机器上也是秒级，因此 GitHub Actions 的快速集真的能跑（M6 任务 5）。权重相关的验证全部沉到慢集。

★`test_separator.py` 的关键设计：**用恒等模型（`x → x.unsqueeze(1)`）+ 合成信号测分块逻辑，不碰 610 MB 权重、不碰 GPU**。构造已知频率成分的信号，验证「重叠区加权归一化正确」「`hop` 计算正确」「末尾不足 `chunk` 时的补位正确」「Hann `periodic=False`」。这样分离算法的骨架能在快速集里被完整覆盖 —— 否则每次改分块逻辑都要跑一遍 GPU 推理。

★**这里用 CPU 不违反 §2.14**：被测的是**分块与加窗算法**（`chunk_starts` / `hann_window` / `window_weights` 都是纯 numpy），模型是个恒等函数 —— 既没有权重，也没有任何"真实推理"发生。§2.14 禁止的是把 BS-RoFormer 的前向放到 CPU 上当后端。真正走 torch 的那条路（`separate_vocals` 的张量搬运）由 `-m gpu` 慢集在真实 GPU 上覆盖。

### 6.5 `tools/bs_roformer_revive_separate.py` 的定位与回归基线

参考脚本从"参考示例"降级为**回归基线**：

- 逻辑全部上移到 `core/`（config / model / audio_io / separator / pipeline）
- 脚本本身改成 ≤ 60 行的薄壳，只做 `argparse` + 调 `cli.main()`
- 改造前的原始版本冻结为 `tools/_baseline_separate.py`，文件头写死"**勿改，这是回归基线**"
- `tests/test_regression_baseline.py`（`@pytest.mark.slow gpu`）对**同一个 60 s 切片**（`test_audio/clips/1.第1章『耳なめ手コキオナサポ』_t240s.wav`）跑两条路径并比对：
  - 旧：`python tools/_baseline_separate.py -i <clip> -o <tmp>`
  - 新：`python -m audio_separation.cli -i <clip> -o <tmp>`
  - 判据：vocals 轨最大绝对误差 `< 1e-4`（fp32 下同设备应逐位一致）；不一致则断言消息里打印**首个差异样本的位置与两侧数值**
  - ★只跑 **1 个切片**，不是 4 个 —— 这是回归基线不是性能测试，两次 610 MB 权重加载的代价只该付一次
  - ★用带 `『』` 的那个文件名 —— 让 Unicode 路径一并进入回归基线，以后谁改了路径处理会立刻暴露

★ 这条"新旧比对"是防止 §2.2 里那 7 条看似多余的逻辑在后续重构中被悄悄删掉的**唯一保险**。

★ `_baseline_separate.py` 依赖 `audio_ai.paths`（不存在）⇒ 测试内**把它当独立脚本跑**，用 subprocess 而非 import，失败时 skip 并提示"需要 Audio_AI 的 paths.py，或改用 git 历史版本"。

---

## 7. 风险与对策

| # | 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|---|
| R1 | **ROCm 算子挂死**（已实测：空格路径 bug + 参考脚本自曝会挂死 + §1.3 的 50 分钟零输出事件） | 高 | 高 | 子进程隔离（§2.3）+ watchdog 退出码 3 + UI 明确提示**终止并排查 GPU/驱动/Python 安装路径**（★按 §2.14 不提供换 CPU 的出路） |
| R2 | `import torch` 在本机 > 120 s 或失败 | 中 | 高 | UI 进程不 import torch；§2.6 **L1 纯静态层**（`importlib.metadata`，不起子进程）永远能报出库状态；**L2** 子进程探测带 90 s 超时；★**超时或失败即终止**并显示原始 stderr（§2.14 不允许降级 CPU） |
| R19 | ★**L1 静态层被误写成 `import xxx` 来判断装没装** ⇒ torch 一挂，静态层一起失效，用户连"哪个包装没装"都看不到 | 中 | 高 | `core/requirements.py` 只暴露 `importlib.metadata.version`；`tests/test_requirements.py` 断言检查路径**不含任何 `importlib.import_module` 调用**；§M4 人工验收 ⑫ 专门验这条 |
| R20 | ★**把 ffmpeg 排除在检测外 ⇒ 打包漏了 ffmpeg 要等到跑第一个文件才发现** | 中 | 低 | ffmpeg 项目内嵌（§0），不进前置检测（§2.6）；改由 `tests/test_packaging.py`（`@pytest.mark.packaging`）在产物上断言 `ffmpeg/bin/ffmpeg.exe`（win）/ `ffmpeg/bin/ffmpeg`（linux）存在且可执行 —— 检测在开发期，不在用户启动路径 |
| R3 | v2 权重键名与 v3 不同（缺 `.branch.` / 多 `branch.*`） | 中 | 高 | M1 阶段就对两个 ckpt 都做严格比对，把差异写进 `model.py` 的注释与 `describe_model()` |
| R4 | BS-RoFormer 库升级（1.2.4 → 2.x）导致签名/结构变化 | 中 | 高 | `inspect.signature` 过滤 + 严格键名/形状校验，**失败时报清楚**而不是静默降级；`requirements.txt` 锁 `BS-RoFormer==1.2.4` |
| R5 | 610 MB × 2 权重进 Git 仓库 | 低 | 中 | `.gitignore` 排除 `models/`；README 写明下载方式 |
| R6 | Linux 下 PyInstaller 丢 ffmpeg `+x` | 高 | 中 | 运行时 `os.chmod(0o755)`；`tests/test_packaging.py` 覆盖 |
| ~~R7~~ | ~~用户改了 `chunk_size` 导致 `dim_t` 不匹配~~ | — | — | ★**已消解**：§2.11 移除了全部超参编辑入口，`chunk_size`/`dim_t` 都是模型常量。**风险被"不给改"消掉，而不是被"校验得更好"消掉** |
| R8 | GUI 打包后子进程起不来（`sys.executable` 语义变化） | 中 | 高 | `paths.self_executable()` 统一判定 + `tests/test_packaging.py` 显式测 `--worker` |
| R9 | 大文件（>1 h）内存占用（float32 立体声 ≈ 1.2 GB/10 min） | 中 | 中 | 子进程内分块处理时只保留**累积缓冲**，不额外持有多份；长任务给进度与 ETA；`build_model` 后 `del` ckpt 副本 |
| R10 | 双击 exe 时 cwd 不在 exe 旁 → 路径解析失败 | 中 | 中 | `paths.py` 的探测链含"从本文件向上"，并加"从 `sys.executable` 向上"一路 |
| R11 | ★**测试素材太长**（930 s + 1007 s，合计 513 MB）直接进验证流程 ⇒ 每次改代码等半小时，验证体系事实上瘫痪 | **高** | 高 | `tools/make_test_clips.py` 切 4 个 60 s 片段（§1.2）；`tests/conftest.py` 的 `clips` fixture 是 pytest 侧的唯一入口，测试代码里不出现全曲路径；`cli.py` 的 `--full-length` 只作为人工长跑逃生开关 |
| R12 | ★**文件名含日文 + 全角 `『』`** ⇒ ffmpeg 在 Windows 上可能不按宽字符解析 argv，导致解码失败或产物路径乱码 | ~~中~~ → **低** | 中 | ★**M0 已实测排除**：`ffmpeg-n9.0` 构建正确按宽字符解析 argv（日文 `『』` 输入 → 中文 `『』` 输出，`exit=0`），§2.4 的 ASCII 临时名 / `pipe:0` 备选方案**不启用**。JSON Lines 仍须 `ensure_ascii=False` + 管道 UTF-8；**Linux 侧待验**（R18） |
| R13 | ★**素材是 24-bit**，`audio_io.write_wav` 输出 int16 ⇒ 产物比输入少 8 bit，mix 校验判据被量化误差污染 | 中 | 中 | 输入侧 `pcm_s24le→f32le` 是**升精度**无损（§2.4），判据成立；但若将来要输出 24-bit，需换 `soundfile`（`scipy.io.wavfile` 只能写 int16/uint8/float32）|
| R14 | 切片偏移点固定在 240 s / 600 s，可能正好落在静音或纯音乐段 ⇒ 分离质量看着像"模型不行" | 中 | 低 | `make_test_clips.py` 切完用 ffprobe 算每个切片的 RMS 与峰值，低于阈值告警；必要时换偏移 |
| R15 | ★**pytest 进程里 torch 挂死** ⇒ 整套测试卡住，且看不出卡在哪个用例 | 中 | 高 | `pytest-timeout` 全局 `--timeout=1800 --timeout-method=thread`（`signal` 方法 Windows 不可用）；`test_env_check` 本身走子进程探测（§2.6），把导入风险隔离在 pytest 之外；L1 部分纯静态，永远不会挂 |
| R16 | ★`test_pipeline`（4 切片）与 `test_regression_baseline`（1 切片）各自加载 610 MB 权重 ⇒ 同一进程里加载两次，慢集时间翻倍 | 中 | 中 | 用 `tests/_model_cache.py` 的 `functools.lru_cache` helper 让两个文件**复用同一份已构建模型**（§6.3）。`test_regression_baseline.py` 用 `pytest` 的 `-k` 可单独跑，缓存自动失效无需清理 |
| R17 | ★**material 缺失被误报成代码坏了** —— `test_audio/` 513 MB 被 gitignore，新克隆仓库没有素材 | 中 | 中 | `clips` fixture 在缺失时 **skip 并打印可操作提示**，不 fail（§6.3） |
| R18 | 跨平台差异只在 Linux 暴露（Unicode 文件名、`+x` 权限位、路径分隔符） | 高 | 中 | Windows 本机跑 CI 快速集；Linux 侧至少保证 `tests/test_paths.py` / `test_audio_io.py` 在两个平台都过（这两组不需要 GPU） |
| R21 | ★**`torch` 写进依赖清单 ⇒ 在 AMD/ROCm 或需要 CUDA 的 Linux 上 `pip install -e .` 装出错误后端**（PyPI 默认版是 CUDA，3 GB N 卡依赖），用户拿到的是"装成功但用不了"，比"明确缺失"更难排查 | **高** | 高 | **§2.13 G 组：`torch` 不进 `pyproject.toml` 与 `requirements.txt`**（后者只以注释形式给出三条按后端分支的命令）。§2.6 L1 仍把它列为 **FAIL 级检查项**（缺了确实跑不了），只报状态 + 给对应命令，**不代为安装**。`tests/test_requirements.py` 断言：`torch` 的 `group == "G"` 且**不出现在 `pyproject.dependencies` 里** |
| R22 | ★**"只检测不安装"被实现成"检测代码里偷偷调 pip"** —— L1 拿了 `install_hint` 就直接 `subprocess.run([sys.executable, "-m", "pip", "install", ...])` | 低 | **高** | `requirements.py` 的 `install_hint` 字段**只能是字符串常量**，工具层不许有 `pip`/`subprocess` 调用。`tests/test_requirements.py` 反向断言：L1 检测路径上不出现任何 `subprocess` 调用（与 R19 的"不出现 `import_module`"同一条思路）。真要装，用户自己复制粘贴命令 —— 这也是"装错后端"最可靠的防线 |

---

## 8. 待确认（本计划不阻塞，但开工前最好有答案）

1. **应用名与图标**：打包后的可执行文件叫什么？（暂定 `AudioSeparation` / `audio-separation-ui`）有没有现成图标资源？
2. **默认输出目录**：暂定 `output/separated/`。是否要按输入文件名自动分子目录（`output/separated/<曲名>/vocals.wav`）？—— ★如果分目录，那么 `『』` 这类字符就会出现在**每一层**目录名上，R12 的验证面要相应扩大
3. ~~**模型选择默认值**：v3 还是 v2？~~ ★**已定：默认 v2**，v3 可在下拉里切换。
4. ~~**ffmpeg 的来源**~~ ★**已定：项目内嵌**。两平台统一目录 `ffmpeg/bin/`，随仓库分发，不需要下载。开工前只需把 `ffmpeg.exe` + `ffprobe.exe` 放进 `ffmpeg/bin/`。
5. **是否需要 `.wav` 以外的输出格式**（flac/mp3）？默认只出 wav（无损、与模型输出同参数）。
6. ★**切片策略确认**：暂定"每首 2 个 60 s 切片，偏移 240 s / 600 s，共 4 个"。如果你知道这两首素材里**人声最密的时段**在哪个位置（比如前奏多长、静音段在哪），告我实际偏移，切片质量更高 —— 这是我猜不到的领域信息。
7. ★**是否要求覆盖率门槛**（`pytest-cov --cov-fail-under=?`）？当前计划只把 `pytest-cov` 列为可选、不设门槛 —— 因为 `core/` 里的推理代码很难用单测覆盖，强行卡数字只会催生无意义的测试。需要的话建议只对 `core/` 之外的 `paths.py` / `config.py` / `presets.py` / `protocol.py` 卡 80%。

---

## 9. 立即可执行的第一步

```bash
# M0 起步
# ① 先按硬件装 torch（本机已有 2.9.1+rocmsdk，可跳过；新机器必做，§2.13 G 组）
#      ROCm : pip install torch --index-url https://download.pytorch.org/whl/rocm
#      CUDA : pip install torch --index-url https://download.pytorch.org/whl/cu124
#      CPU  : pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"                 # src 布局，pytest 直接可用（不碰 torch）
python -m audio_separation.paths       # 先把路径真源立起来（照抄 audio_ai 版并改）
python -m audio_separation.core.model --check v2   # 权重键名/形状严格比对（M1 的人工入口）
python tools/make_test_clips.py        # 切出 4 个 60 s 测试片段（顺便测 ffmpeg 的 Unicode 路径）
pytest -q                              # 快速集；秒级，无权重无 GPU
```

**M0 不写任何 UI**。先把"路径能不能找对、torch 能不能导进去、挂死了能不能被杀掉、ffmpeg 认不认带『』的路径"这四件事变成**可跑的 pytest 用例** —— 这四件事在 M4 一旦出问题，排查成本是 M0 的十倍。

**开工前置**：① 按硬件装好 torch（★工具只检测不安装，§2.13 G 组）② `ffmpeg.exe` + `ffprobe.exe` 放进 `ffmpeg/bin/`（项目内嵌，§8-4）③ `models/bs_roformer_revive/` 已有 v2/v3。这三样不备齐，M0 的路径探测会走 best-effort 猜测分支。

### 9.1 README 必须写清的安装顺序

Linux 侧目前**完全没有安装章节**，而目标平台是双平台 —— 这是 M0 顺手要补的一块：

```
1. 装 Python ≥ 3.11
2. ★按硬件装 torch（上面三条命令选一条）—— 本工具只检测，不代装
3. pip install -e ".[dev]"
4. 放模型到 models/bs_roformer_revive/（v2/v3 各一对 ckpt+yaml）
5. ffmpeg 已在仓库内（ffmpeg/bin/），Linux 上 chmod +x
6. python tools/run_ui.py
```

第 2 步必须在第 3 步之前，且**必须手动** —— 这正是 §2.13 把 torch 划为 G 组的原因。

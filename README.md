[README.md](https://github.com/user-attachments/files/34870380/README.md)
# auto-exposure

任意图片的自动曝光校正 —— 给豆包（Doubao Agent）用的 Skill。

输入 **`执行AEV`** 即可跑完：直方图诊断 → 生成式重打光 / 本地确定性校正 → 自动验收 → 前后对比图 + 报告。

- **过曝**（死白、高光爆掉）、**泛白**（逆光眩光把整幅画面蒙成灰白）、**欠曝**（发暗、暗部死黑）、**逆光高反差**、**发灰低反差** 都能处理
- 覆盖所有图片：JPG/PNG/WEBP/BMP/TIFF，横图/竖图/方图/长截图/超宽，小图到大图，单张或整批，照片/人像/静物/建筑/夜景/截图/证件/商品图
- 默认**不需要任何 API Key**，走豆包账号自带的生图额度
- 内容敏感图（文字、证件、白底商品）自动改走本地确定性校正，**从不拒绝执行**

## 为什么不是"调个亮度"

曝光问题分两种，能不能救取决于数据还在不在：

| 情况 | 数据状态 | 能做什么 |
| --- | --- | --- |
| 欠曝 | 暗部信息还在 | 本地曲线就能真实恢复 |
| 过曝 / 泛白 | 三通道到 255 的区域已丢失；眩光还把整幅黑场抬起来（p01 从 3% 抬到 60%） | 本地只能压成一片均匀灰，**必须靠生成式重绘补层次** |

所以本 Skill 的主路径是生成式，但把诊断和验收都做成了**可复算的确定性代码**：

```
直方图诊断（numpy 算出目标 EV/黑场）
        ↓
生成式重打光（模型按"平均亮度约 52%"这种数字目标改光）
        ↓
自动验收（亮度无关的结构相似度 ≥0.62、死白不增、比例不变）
```

验收里最关键的一条是**亮度无关的结构相似度**：把两张图做高通、零均值单位方差后算余弦相似度。这样"整体压暗 30%"不会被误判成"改了内容"，而真正的换脸、改字、删物体才会把分数打到 0.62 以下。

## 三条路径

| 路径 | 什么时候用 | 成本 |
| --- | --- | --- |
| **内置生图工具**（默认） | 普通图片修曝光 | 豆包账号每日免费次数 |
| **火山方舟 Seedream** | 批量、无人值守、要固定参数复跑 | 每模型免费额度：`5.0-lite` 50 张、`4.5`/`4.0` 各 200 张；超出 0.12–0.25 元/张 |
| **本地确定性校正** | 证件、票据、扫描件、小字海报、白底商品图、涉密图 | 0，纯本地 |

前两条都会输出生成式结果；第三条保证"像素一个不动"。

## 安装

技能本体是仓库里的 `auto-exposure/` 目录，**整个目录丢进技能根目录即可**（目录名保持 `auto-exposure`）。

```bash
# 1) 克隆
git clone https://github.com/<你的用户名>/auto-exposure.git
cd auto-exposure

# 2) 装到你的技能根目录（把路径换成你自己的）
python3 install.py --target "<技能根目录>"

# 或者手动复制
cp -r auto-exposure "<技能根目录>/"
```

`install.py --target` 不写也可以，只打印常见路径提示；它只做复制，不会改动技能目录里的其它内容。

**依赖**（只需一次）：

```bash
pip install -r requirements.txt
```

只需要 `numpy` + `Pillow`；`requests` 仅火山方舟批量路径需要。

### 装完先自检

```bash
python3 auto-exposure/scripts/selftest.py
```

会检查 Python 版本、依赖，并在一张合成图上跑完"诊断 → 校正 → 验收 → 读写"全链路。看到 `自检通过` 再开始用，能少走很多弯路。

## 使用

对豆包说：

```
执行AEV
```

附上一张图、多张图，或给一个目录路径。也可以不用口令，直接说"这张图太亮了，帮我修修曝光"。

**手动跑脚本**（适合批量 / 接自动化）：

```bash
cd auto-exposure

# 只看诊断，不花钱
python3 scripts/analyze.py photo.jpg --pretty

# 内置生图路径：先出诊断 + 提示词 + 送模型用的图
python3 scripts/prepare.py -i photo.jpg -o ./out
#   → out/<stem>_prompt.txt     提示词
#   → out/<stem>_input_for_model.png  送模型的图（极端曝光时已轻度归一化）
#   → 然后用你的生图工具（参考图=上面那张，提示词=prompt.txt，尺寸看 size_hint）

# 拿到生成结果后验收 + 出对比图和报告
python3 scripts/finish.py --before photo.jpg --after generated.png -o ./out \
  --source doubao --model seedream_5.0_flash

# 纯本地批量（不联网、不改像素结构）
python3 scripts/pipeline.py -i ./photos -o ./out --mode local

# 火山方舟批量（需先配 Key）
python3 scripts/ark_client.py --check
python3 scripts/pipeline.py -i ./photos -o ./out --mode ark
```

### 产出

```
<stem>_fixed.png          最终交付图
<stem>_compare.jpg        BEFORE | AFTER 对比图
report.md                 判定、前后指标、验收结论、限制说明
<stem>_report.json        机器可读的同款结果
```

### 验收口径

- **pass**：平均亮度落进目标 ±10%、死白不增、结构相似度 ≥0.62、宽高比变化 ≤3%、清晰度没掉（仅生成式查）
- **partial**：亮度或死白没完全达标，通常是物理上限（太阳本体、原始死白区）
- **fail**：结构漂移。`finish.py` 此时会**自动**补一版本地校正并交付，不会把改坏的图给你

## 可选：接火山方舟（批量用）

1. 用豆包/抖音账号登录 [火山方舟控制台](https://console.volcengine.com/ark)
2. 「API Key 管理」创建 Key
3. 写入配置：

```bash
python3 auto-exposure/scripts/ark_client.py --save-key "<ARK_API_KEY>"
```

免费额度**按张**计：`doubao-seedream-5-0-lite` 50 张、`4-5` / `4-0` 各 200 张，额度内 0 元。
注意这几个模型控制台标注"即将下线"，新账号可能开不出来，脚本会自动顺序回退到付费的 `5-0-flash`（0.12 元/张）。
细节见 `auto-exposure/references/ark-setup.md`。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| 会不会把我的图改样？ | 结构相似度 <0.62 会被判 fail 并自动换成本地版；要绝对保像素就明确用本地路径 |
| 文字/证件图能用吗？ | 能，但会自动走本地路径——生成式可能把字写错 |
| 竖图会不会被裁？ | 不会，按长边等比缩放到 2048（1600×2848 → 1151×2048） |
| 小图会不会糊？ | 送模型前会放大到长边 ≥1024 |
| HEIC / RAW 呢？ | 先转成 JPG 再跑 |
| 图会不会上传？ | 走生成式会上传到豆包生图服务；不愿出网就用 `--mode local` |
| 一张图多久？ | 本地 2–4 秒；生成式 10–60 秒 |
| 为什么有的图目标亮度是 37% 而不是 50%？ | 高光挤不出空间时，`calibrate_plan()` 会把目标降到这张图实际能达到的值，否则只能靠把窗户推成死白。报告里 `target_limited=true` 就是这个信号 |

## 目录结构

```
auto-exposure/
├── SKILL.md                    技能说明（触发条件与工作流）
├── references/
│   ├── ark-setup.md            方舟开通、Key、免费额度与模型 ID、报错速查
│   ├── diagnosis-rules.md      指标定义、判定阈值、目标亮度、物理限制
│   └── prompt-recipes.md       提示词配方、失败重试、不适用场景
└── scripts/
    ├── selftest.py             环境 + 全链路自检
    ├── analyze.py              只诊断
    ├── prepare.py              诊断 + 出提示词（生成式第 1 步）
    ├── finish.py               验收 + 对比图 + 报告（生成式第 2 步）
    ├── pipeline.py             端到端（方舟 / 本地）
    ├── correct_local.py        单张本地校正
    ├── ark_client.py           方舟调用 + 鉴权 + 计费元数据
    └── exposure_lib.py         指标、分类、色调映射、验收、提示词（核心库）
```

## 局限

- 原始就烧成纯白（三通道 255）的区域，信息已物理丢失，生成式只能"合理重建"，不是找回
- 生成式是重绘而非解码，纹理/细节可能有微调；法证级一致、含文字的场景请用本地路径
- 只处理曝光与明暗反差，不做白平衡、磨皮、美颜、去路人、抠图、超分
- 不做视频逐帧校正


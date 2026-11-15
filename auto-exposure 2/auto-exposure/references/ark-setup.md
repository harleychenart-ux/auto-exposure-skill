# 方舟接入：用自己的豆包账号拿到免费额度

本 Skill 的生成式校正是调用**用户自己的火山方舟账号**，不是共享额度。一次对话里只需要配一次。

## 1. 开通步骤

1. 用**豆包/抖音账号**登录火山引擎控制台：<https://console.volcengine.com/ark>
   （首次登录需完成实名认证，个人认证几分钟即可）。
2. 进入「火山方舟 → 模型 → 开通管理」，找到生图模型并**开通**（开通本身免费）。
3. 进入「API Key 管理」，创建一把长效 API Key，复制保存。
4. 把 Key 交给 Skill（任一方式）：

```bash
# 推荐：写入用户级配置（权限 600，不进版本库）
python3 scripts/ark_client.py --save-key "你的ARK_API_KEY"

# 或者用环境变量（临时会话）
export ARK_API_KEY="你的ARK_API_KEY"
```

配置文件：`~/.ark/exposure_correction.json`

```json
{
  "api_key": "xxxx",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "size": "2K",
  "models": ["doubao-seedream-5-0-lite-260128", "doubao-seedream-4-5-251128"]
}
```

自检：`python3 scripts/ark_client.py --check`
返回 `has_api_key: true` 即就绪；`endpoint_reachable: ok(http 401)` 表示网络通、Key 未带（正常）。

## 2. 计费口径（决定“免不免费”）

图片模型**按张计费**，免费额度也**按张给**（官方“免费额度”表，2026-11 版）：

| 模型 | 免费额度 | 超出后单价 | 备注 |
| --- | --- | --- | --- |
| `doubao-seedream-5-0-lite` | **50 张** | 0.22 元/张 | 新号最常说的“50 张免费”就是它 |
| `doubao-seedream-4-5` | **200 张** | 0.25 元/张 | 控制台标注即将下线 |
| `doubao-seedream-4-0` | **200 张** | 0.20 元/张 | 控制台标注即将下线 |
| `doubao-seedream-5-0-flash` | 无（表中未列） | 0.12 元/张 | 最便宜的付费档 |
| `doubao-seedream-5-0-pro` | 无 | 0.30 元/张起 | |
| `doubao-seedream-5-5` | 无 | 0.30 元/张起 | 编辑能力最强 |

其它规则：**输入图不计费**；**失败不计费**；免费额度按模型分别计算、额度内优先消耗。
注意别把这条和「安心体验模式的 50 万 token」搞混——那是**文本模型**的口径，视觉模型既不吃这套 token，
也不支持设推理限额。

结论：**先用 `doubao-seedream-5-0-lite` 的 50 张，用完再落到 4.5 / 4.0 的 200 张，最后才动就要花钱的 flash**。
每次校正只产 1 张输出图，所以 50 张免费额度 ≈ 50 张照片白嫖。批量前先报张数：
「这批 30 张，会吃掉 30/50 张免费额度，确认吗？」——额度不够时应主动提醒可以换 4.5。

**唯一要留心的坑**：这三个带免费额度的模型在控制台都标了「即将下线」。老账号还能用，
新注册账号可能已经开不出来；一旦返回 404，脚本会自动顺序回退，最终落到付费的 flash（0.12 元/张）。
所以跑之前先 `--check`，别默认免费。

Agent Plan 套餐用户：用套餐专属 Key `AGENT_API_KEY`，Base URL 必须换成
`https://ark.cn-beijing.volces.com/api/plan/v3`（带 `/plan`，混用会产生额外费用）。
本 Skill 检测到 `AGENT_API_KEY` 时会自动切换。

## 3. 模型与接口

- 接口：`POST {base_url}/images/generations`，请求头 `Authorization: Bearer <key>`
- 关键参数：
  - `model`：模型名（见下）
  - `prompt`：校正提示词（由 `exposure_lib.build_prompt()` 按诊断结果生成）
  - `image`：`["data:image/jpeg;base64,..."]`（也支持公网 URL 数组）
  - `size`：`"2K"`（也接受 `"1K"/"1.5K"/"auto"` 或 `"宽x高"`）
  - `response_format`：`"b64_json"`（免二次下载，链接 24h 失效）
  - `watermark`：`false`
- 候选模型（按顺序自动回退，前一个不可用就试下一个）：

```
doubao-seedream-5-0-lite-260128（50 张免费）
→ doubao-seedream-5-0-lite（别名）
→ doubao-seedream-4-5-251128（200 张免费）
→ doubao-seedream-4-0-250828（200 张免费）
→ doubao-seedream-5-0-flash-260915（0.12 元/张）
```

模型 ID 带日期后缀，`5.0` / `5-0` 两种写法在文档里混用，脚本都试；用 `--model` 可指定，
或改配置里的 `models` 数组。**以能调通的那个为准**，跑通后建议写死到配置里。

## 4. 报错速查

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| `401 AuthenticationError` | Key 错/被删/带空格 | 重新复制 Key，重跑 `--save-key` |
| `403 AccessDenied` / 未开通 | 该模型未开通 | 控制台「开通管理」开通 |
| `429 RateLimitExceeded` / 额度用尽 | 并发或免费额度耗尽 | 等 1 分钟；或换 flash；或充值 |
| `404 InvalidEndpointOrModel` | 模型名不对 | 脚本会自动换下一个；也可 `--model` 指定控制台里的完整 ID |
| 400 提到 `size` | 尺寸档位不支持 | 脚本自动降级 `2K → 1.5K → auto` |
| 审核不通过 | 图片被判敏感 | 改本地确定性校正（`--mode local`），或换图 |
| `unreachable` | 网络/代理不通 | 检查代理；或 `--mode local` 离线处理 |

## 5. 隐私说明

图片会以 base64 上传到火山引擎方舟做推理（属于用户自己的账号）。涉及证件、病历、客户隐私的图，
优先用 `--mode local`，本地不出网。

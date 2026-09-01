# 灵犀 · AI生图

[![Plugin Version](https://img.shields.io/badge/Version-1.3.0-4f8cc9?style=for-the-badge)](./CHANGELOG.md)
[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.16.0%2C%20%3C5-ff69b4?style=for-the-badge)](https://github.com/AstrBotDevs/AstrBot)
[![Platform](https://img.shields.io/badge/Primary-aiocqhttp-4caf50?style=for-the-badge)](#平台与限制)

灵犀 AI 生图插件，支持多服务商文生图 / 改图 / 自拍参考照 / 视频生成，支持命令调用、`LLM tool` 调用、预设提示词、批量出图、请求模式控制、多 `API Key` 轮询、失败兜底与超时配置。

> [!NOTE]
> 本插件基于原 AstrBot Gitee AI 图像生成插件改造，感谢原作者木有知、Zhalslar 的开源贡献。

> [!IMPORTANT]
> 这份文档对应灵犀 · AI生图 `1.2.0` 配置结构。
>
> - `v4` 与旧版 `v3 / v2` 配置不兼容，升级后请重新检查 WebUI 配置。
> - 插件主维护场景是 `QQ / aiocqhttp`，并针对个人微信 `weixin_oc` 增加了发送图片前优化。
> - 批量结果的“合并转发”当前只有 `aiocqhttp` 原生支持；其他平台会在开启回退时自动改为普通消息逐条发送。
> - 历史更新内容见 [CHANGELOG.md](./CHANGELOG.md)。

## 新版本重点

- `LLM tool` 统一为 `aiimg_generate(count=1)`；`count>1` 进入内部批量规划，不再暴露独立批量工具
- `aiimg_generate` 默认后台执行，接单后即可继续聊天；图片完成后自动发送并在原会话自然回应
- 后台单图与批量共享全局容量和 provider 并发限制，每个 provider 子请求仍固定 `n=1`
- 后台仅用于 LLM Tool；`/批量n`、`/自拍`、`/改图` 等手动命令保持同步语义
- 新增文生图预设：`/文生图 预设名 补充提示词`
- 新增统一批量命令：`/批量n aiimg ...`、`/批量n aiedit ...`、`/批量n 自拍 ...`
- 支持批量配合预设：`/批量n 文生图 预设名 补充提示词`、`/批量n 改图预设名 补充提示词`
- 文生图批量并发和改图 / 自拍批量并发拆开配置
- Provider 级新增 `generate_request_mode` / `edit_request_mode`
- 请求模式兼容旧配置：`auto` 不会再覆盖旧的 `enable_stream_*` 布尔配置
- 个人微信 `weixin_oc` 发送图片前可自动优化 4K 大图，并把适配器上传超时调高到配置值

## 功能概览

本插件支持：

- 文生图 `Text-to-Image`
- 图生图 / 改图 `Image-to-Image/Edit`
- 自拍参考照模式
- 图生视频 / 文生视频（取决于所选后端能力）
- 文生图预设、改图预设、视频预设
- 指令批量出图
- `LLM tool` 统一单图 / 批量调用

核心设计是把 **服务商实例 `providers`** 和 **功能链路 `features.*.chain`** 分开。你可以给同一类能力挂多个 provider，插件会按链路顺序兜底切换。

## 快速上手

### 1. 先配 `providers`

每个 provider 都需要一个唯一的 `id`。目前内置了多类模板，例如：

- Gemini 原生 `generateContent`
- Vertex AI Anonymous
- OpenAI 兼容 `Images API`
- OpenAI 兼容 `Chat` 出图解析
- OpenAI 兼容完整路径
- Flow2API
- Grok / Grok2API
- Gitee / Gitee 异步改图
- 即梦 / 豆包聚合
- 视频后端

一般选择建议：

- 标准 `POST /v1/images/generations` / `POST /v1/images/edits`：用 `OpenAI 兼容通用（Images）`
- 只在 `chat.completions` 回复里回图片：用 `OpenAI 兼容（Chat 出图解析）`
- 你的网关路径不是标准 `/v1/...`：用 `OpenAI 兼容-完整路径`
- 直连 Gemini 官方：用 `Gemini 原生`

### 2. 再配 `features.*.chain`

- `features.draw.chain`：文生图链路
- `features.edit.chain`：改图链路
- `features.selfie.chain`：自拍链路
- `features.video.chain`：视频链路

链路按顺序兜底。前面的 provider 失败后，插件会自动切到后面的 provider。

### 3. 按需开关功能

- `features.<mode>.enabled`：是否启用该能力
- `features.<mode>.llm_tool_enabled`：是否允许 `LLM` 通过工具调用该能力

## 命令速查

| 功能 | 命令 |
| --- | --- |
| 普通文生图 | `/aiimg [@provider_id] <提示词> [比例]` |
| 文生图预设 | `/文生图 [@provider_id] <预设名> [补充提示词]` |
| 改图 | `发送或引用图片 + /aiedit [@provider_id] <提示词>` |
| 改图预设 | `发送或引用图片 + /预设名 [@provider_id] [额外提示词]` |
| 自拍 | `/自拍 [@provider_id] <提示词>` |
| 自拍参考图管理 | `发送图片 + /自拍参考 设置`、`/自拍参考 查看`、`/自拍参考 删除` |
| 批量文生图 | `/批量n aiimg [@provider_id] <提示词>` |
| 批量改图 | `发送或引用图片 + /批量n aiedit [@provider_id] <提示词>` |
| 批量自拍 | `/批量n 自拍 [@provider_id] <提示词>` |
| 批量文生图预设 | `/批量n 文生图 [@provider_id] <预设名> [补充提示词]` |
| 批量改图预设 | `发送或引用图片 + /批量n <改图预设名> [额外提示词]` |
| 视频 | `发送或引用图片 + /视频 [@provider_id] <提示词或预设名>` |
| 文生图预设列表 | `/文生图预设列表` |
| 改图预设列表 | `/预设列表` |
| 视频预设列表 | `/视频预设列表` |
| 重发最近结果 | `/重发图片` |
| 查看改图帮助 | `/改图帮助` |

## 文生图预设

### 配置位置

在 `features.draw.presets` 里配置，格式是：

```text
预设名:英文提示词
```

示例：

```text
手办:Transform into collectible figurine style
胶片人像:Cinematic portrait, soft rim light, film grain, realistic skin texture
```

### 调用方式

```text
/文生图 手办 将这只猫做成高细节手办
/文生图 @gemini_chat 胶片人像 黑色高领毛衣，窗边逆光，半身构图
```

规则说明：

- `/文生图` 后面第一个 token 如果命中文生图预设名，就按“预设 + 补充提示词”处理
- 预设名后面的**全部文本**都会作为补充提示词，可包含空格，也可写成多行消息
- 如果第一个 token 没命中预设，就按普通 `/文生图` 文生图处理
- 指定 provider 时，`@provider_id` 要放在预设名前面

实际拼接方式是：

```text
预设提示词

补充要求：
你的补充提示词
```

## 改图预设

### 配置位置

在 `features.edit.presets` 里配置，格式同样是：

```text
预设名:英文提示词
```

示例：

```text
手办:Transform into figurine style
Q版化:Convert to chibi illustration style
```

### 调用方式

发送或引用图片后：

```text
/手办 加个透明亚克力底座
/手办 @grok2api 换成偏暖色棚拍
/Q版化
```

规则说明：

- 每个改图预设都会动态注册成一个独立命令，例如 `/手办`
- 预设命令后面的文本会作为额外提示词附加到预设后面
- 预设命令也支持 `@provider_id` 覆盖，例如 `/手办 @provider_xxx 补充词`
- `/预设列表` 可以查看当前所有改图预设

## 批量出图

### 基本语法

批量命令统一使用：

```text
/批量n ...
```

其中 `n` 是数量，例如：

```text
/批量4 aiimg 一个粉发少女，4 个不同镜头角度
/批量6 aiedit 把这张照片分别改成不同灯光和情绪
/批量8 自拍 同一套穿搭，不同姿势、表情和俯仰角
/批量5 文生图 手办 将这辆车做成桌面手办
/批量4 文生图 @gemini_chat 胶片人像 夜景路灯，表情和构图都不重复
/批量3 手办 加不同底座和背景陈列
```

### 支持的批量入口

- `/批量n aiimg ...`
- `/批量n 文生图 ...`
- `/批量n aiedit ...`
- `/批量n 自拍 ...`
- `/批量n 改图预设名 ...`

### 行为说明

- 单次数量上限由 `features.batch.max_count` 控制，默认 `8`
- 文生图批量并发由 `features.draw.batch_concurrency` 控制，默认 `2`
- 改图 / 自拍批量并发由 `features.edit.batch_concurrency` 控制，默认 `2`
- 批量并发数只限制已经明确创建的批量任务，不会把单图请求扩成多张
- 并发数设为 `1` 不会关闭批量，只会让批量子任务逐个串行执行
- 每个 provider 子请求固定 `n=1`；批量通过多个独立单图请求实现，避免服务商额外生成未被插件消费的图片
- 改图批量和自拍批量都要求当前消息里能读到输入图片；文生图批量不需要图片
- 批量结果会按顺序直接发送单张图片
- 除原插件自带表情反馈外，不额外发送标题、提示词、状态、失败摘要这类通知文本

### 批量结果展示

- 批量任务成功的图片会一张一张直接发出
- 不额外插入摘要、说明、失败提示等文本消息
- 如果某几张失败，只保留原插件自己的表情反馈，不额外发通知

## 自拍参考照

### 设置参考照

二选一即可：

1. 发送图片后执行：

```text
/自拍参考 设置
```

2. 直接在 WebUI 的 `features.selfie.reference_images` 上传

### 查看和删除

```text
/自拍参考 查看
/自拍参考 删除
```

### 生成自拍

```text
/自拍 日常自拍照，微笑，窗边自然光
/自拍 @provider_xxx 黑色外套，楼梯间，低头看镜头
```

说明：

- 如果 WebUI 里已经上传了参考照，优先使用 WebUI 配置
- 如果同时没有 WebUI 参考照，也没有通过命令保存参考照，`/自拍` 会直接报错
- 自拍链路为空时，可通过 `features.selfie.use_edit_chain_when_empty=true` 复用改图链路

## 视频生成

发送或引用图片后：

```text
/视频 镜头缓慢推进，人物轻微转头
/视频 @grok_video 黄昏街景，镜头跟拍
/视频 电影感 拉近镜头，轻微风吹头发
/视频预设列表
```

如果第一个 token 命中 `features.video.presets` 里的预设名，就会按“视频预设 + 额外提示词”处理。

## LLM 工具

插件只向 `LLM` 提供一个图片工具：`aiimg_generate`。

### `aiimg_generate`

统一工具参数为 `aiimg_generate(prompt, mode, backend, output, count=1)`。`count=1` 执行一次单图流程；只有用户明确要求多张或一组图片时才设置 `count>1`，并进入内部批量规划与执行。数量超过 `features.batch.max_count` 会直接拒绝，不会静默缩减。

在受支持环境中，`aiimg_generate` 默认使用后台任务：工具会在输入图片固化、容量预留和任务登记完成后立即返回，单图不会等待 provider，批量也不会等待 planner。当前 Agent 回合因此可以尽快结束，等待期间的新消息会作为正常新回合处理；图片按生成计划发送后，插件会在原会话提交一次自然的终态回应。

以下情况会自动回到原同步链路，不创建半成品后台任务：

- `features.background_llm_image.enabled=false`
- 平台不是 `aiocqhttp` 或 `weixin_oc`
- AstrBot 开启 streaming response
- 当前事件无法取得稳定 conversation 或平台不具备事件重建能力
- 后台任务账本初始化失败或当前不可用

手动 `/批量n ...`、`/自拍`、`/改图` 等命令不走该后台链路，原命令时序保持不变。后台任务可用 `/stop` 取消；成功执行 `/reset` 或 `/new` 后，旧会话任务会被取消并与新会话隔离。

模型只需要提供完整提示词并选择业务模式：

- `selfie_ref`：生成固定人物本人的自拍、生活照或穿搭照。用户附带或引用的图片可作为服装、姿势、构图或场景参考。
- `edit`：直接修改用户发送或引用的原图，例如换背景、删除元素、修图或改变原图风格。
- `text`：生成与固定人物无关的新图片，例如风景、人群、食物、物品或插画。

关键区别：带图片不等于改图。修改这张原图使用 `edit`；参考这张图重新拍一张固定人物本人的照片使用 `selfie_ref`。

`selfie_ref` 会按以下顺序提交图片：

1. 配置或命令保存的固定人物参考图，保持原顺序。
2. 引用消息中的用户参考图，保持原顺序。
3. 当前消息中的用户参考图，保持原顺序。

插件会根据实际数量在最终提示词中标明固定人物图和本次用户参考图的范围，不依赖“最后一张图”之类的模糊编号。

`backend`、`output` 和 `mode=auto` 继续在实现层兼容旧调用，但不再作为推荐的模型决策参数；默认服务商链和输出规格由插件配置决定。

## Provider 请求模式

部分支持双路径的 provider 可分别配置：

- `generate_request_mode`
- `edit_request_mode`

可选值：

- `auto`：由后端自己决定
- `stream`：强制优先走流式
- `non_stream`：强制直接走非流式

补充说明：

- 显式设置 `stream` / `non_stream` 时，它们优先级最高
- 如果你是从旧配置升级，且旧配置里还保留 `enable_stream_generate` / `enable_stream_edit`，当新的 `*_request_mode=auto` 时，插件会继续沿用旧布尔值，不会被 `auto` 覆盖
- 单路径后端即使显示了这个配置项，也可能会忽略该设置；校验阶段会给出提示

## 关键配置项

### LLM 后台生图

- `features.background_llm_image.enabled`：是否后台执行 `aiimg_generate`，默认 `true`
- `features.background_llm_image.max_running`：单图与批量子任务共享的全局 provider 并发上限，默认 `2`
- `features.background_llm_image.max_queued`：按图片张数计算的全局后台容量，默认 `16`；一组 4 张会原子占用 4 个容量

后台任务使用插件数据目录中的 SQLite 账本持久化状态和发送 receipt。发送边界发生异常时，插件会把状态记为无法确认，不会盲目重发可能已经送达的图片。

### 批量相关

- `features.batch.max_count`：单次批量最大张数
- `features.draw.batch_concurrency`：文生图批量任务并发上限，仅在 `count>1` 或批量命令时生效；设为 `1` 时串行执行
- `features.edit.batch_concurrency`：改图 / 自拍批量任务并发上限，仅在 `count>1` 或批量命令时生效；设为 `1` 时串行执行

### 并发与防抖

- `debounce_interval`：防抖时间，防止同一用户短时间重复提交同类任务
- `max_user_concurrency`：同一用户同时执行的图像任务上限
- `max_user_video_concurrency`：同一用户同时执行的视频任务上限

### 发送前处理

- `send.weixin_compress_images`：个人微信发送前压缩图片，默认开启，仅对 `weixin_oc` 生效。
- `send.weixin_image_max_side`：个人微信图片最长边，默认 `4096`。
- `send.weixin_image_max_size_kb`：个人微信图片目标大小，默认 `10240KB`。
- `send.weixin_api_timeout_seconds`：个人微信发送超时，默认 `60` 秒。

这组配置只影响 `weixin_oc`。QQ / OneBot 仍使用原有发送与兜底逻辑。若个人微信发送 4K 图片时出现 `upload_to_cdn TimeoutError`，优先调高 `send.weixin_api_timeout_seconds`，或降低 `send.weixin_image_max_size_kb`。

### 功能开关

- `features.draw.enabled`
- `features.edit.enabled`
- `features.selfie.enabled`
- `features.video.enabled`
- `features.<mode>.llm_tool_enabled`

## 平台与限制

### 官方维护 / 推荐环境

- `AstrBot >= 4.16.0, < 5`
- 主要维护平台：`QQ / aiocqhttp`
- 兼容平台：个人微信 `weixin_oc` 发送图片前优化

### 已知平台限制

- LLM 后台生图仅在 `aiocqhttp` 和 `weixin_oc` 启用；其他平台自动使用同步链路
- AstrBot 开启 streaming response 时，LLM 生图自动使用同步链路
- 批量结果默认就是普通消息逐张发送
- 视频发送依赖适配器是否支持 `Video.fromFileSystem` 或 `Video.fromURL`
- 某些需要 URL 回退输入的后端，依赖当前 AstrBot 环境具备文件服务能力
- `weixin_oc` 官方适配器仅支持个人私聊，不支持微信群聊；大图发送受微信 CDN 上传耗时影响。

### 关于“其他平台能不能跑”

单图文生图 / 改图这类能力，只要当前适配器支持普通文本和图片消息，很多时候都能工作；但本插件的主要测试和维护场景仍是 `QQ / aiocqhttp`。如果你跑在其他平台，建议先用小流量自测，尤其是：

- 批量结果展示
- 视频发送
- 合并转发回退行为
- 大图发送失败后的兜底链路

## Gitee AI API Key 获取

1. 访问 <https://ai.gitee.com/serverless-api?model=z-image-turbo>
2. 在对应模型页面开通服务并创建 `API Key`
3. 将 `API Key` 填到对应 provider 的 `api_keys`

## Gitee 支持的图像尺寸

> 仅对 Gitee 文生图能力生效。其他后端是否支持，取决于服务商本身。

| 比例 | 可用尺寸 |
| --- | --- |
| `1:1` | `256x256`、`512x512`、`1024x1024`、`2048x2048` |
| `4:3` | `1152x896`、`2048x1536` |
| `3:4` | `768x1024`、`1536x2048` |
| `3:2` | `2048x1360` |
| `2:3` | `1360x2048` |
| `16:9` | `1024x576`、`2048x1152` |
| `9:16` | `576x1024`、`1152x2048` |

## 常见问题

### `/文生图` 预设没有生效

先检查：

- `features.draw.presets` 里是否真的配置了该预设名
- 调用时预设名是否放在 `/文生图` 后面的第一个 token
- 如果用了 `@provider_id`，它必须写在预设名前面

### `/批量n aiedit ...` 没反应

改图批量要求当前消息里能读到输入图片。最稳妥的发法是：

- 先发图再跟命令
- 或直接回复图片消息执行命令

### 批量结果为什么没有额外说明文字

这是当前设计：批量结果默认只发图片本体，避免刷出不拟人的通知文本。

### 为什么 `request_mode=stream` 没起作用

并不是所有 provider 模板都支持“双路径请求模式”。单路径后端会忽略这个设置，插件会在校验时提示你。

### 个人微信生成图为什么发送失败或超时

个人微信 `weixin_oc` 发送图片会先上传到微信 CDN。即使图片低于 10MB，也可能因为网络或默认超时过短导致 `upload_to_cdn TimeoutError`。

建议：

- 保持 `send.weixin_compress_images=true`
- 将 `send.weixin_api_timeout_seconds` 设置为 `60-120`
- 如果仍超时，降低 `send.weixin_image_max_size_kb`
- 视频发送能力取决于当前 `weixin_oc` 适配器是否支持对应 `Video` 组件

## 原仓库展示内容（保留）

这一节保留原仓库 README 里的推广与展示内容，方便插件市场页和仓库首页继续正常展示。

### Gitee AI API Key 获取方法（原展示）

1. 访问 <https://ai.gitee.com/serverless-api?model=z-image-turbo>

2. ![PixPin_2025-12-05_16-56-27](https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841)

3. 免费额度

![PixPin_2025-12-05_16-56-49](https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb)

4. 可以涩涩，警惕违规被举报

5. 好用可以给个 💗

### 出图展示区（原展示）

![出图展示 1](https://github.com/user-attachments/assets/c2390320-6d55-4db4-b3ad-0dde7b447c87)

![出图展示 2](https://github.com/user-attachments/assets/3d8195e5-5d89-4a12-806e-8a81e348a96c)

![出图展示 3](https://github.com/user-attachments/assets/c270ae7f-25f6-4d96-bbed-0299c9e61877)

插件开发 QQ 群：`215532038`

![QQ群二维码](https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee)

# AIWork 生图 API 接入文档

## 结论

外部用户接入生图接口时，推荐统一使用 `response_format: "b64_json"`，不要默认使用 `response_format: "url"`。

返回 `b64_json` 后，客户端可拼接为图片地址：

```js
const imageSrc = `data:image/png;base64,${b64_json}`
```

## 认证方式

所有接口都需要在请求头中携带用户自己的 API Key：

```http
Authorization: Bearer sk-你的APIKEY
```

## 文生图

### 接口

```http
POST https://aiwork.fans/v1/images/generations
```

### 可用 curl

```bash
curl -X POST "https://aiwork.fans/v1/images/generations" \
  -H "Authorization: Bearer sk-你的APIKEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-image-2",
    "prompt": "画个苹果",
    "size": "1024x1024",
    "quality": "auto",
    "response_format": "b64_json",
    "n": 1,
    "output_format": "png",
    "background": "auto",
    "stream": false
  }'
```

### 请求头

| Header | 必填 | 说明 |
|---|---:|---|
| `Authorization` | 是 | `Bearer sk-你的APIKEY` |
| `Content-Type` | 是 | `application/json` |

### 请求体字段

| 字段 | 类型 | 必填 | 推荐值 | 说明 |
|---|---:|---:|---|---|
| `model` | string | 是 | `gpt-image-2` | 生图模型 |
| `prompt` | string | 是 | - | 生图提示词 |
| `size` | string | 是 | `1024x1024` | 图片尺寸 |
| `quality` | string | 否 | `auto` | 图片质量 |
| `response_format` | string | 是 | `b64_json` | 返回 base64 图片 |
| `n` | number | 否 | `1` | 生成张数 |
| `output_format` | string | 否 | `png` | 输出格式 |
| `background` | string | 否 | `auto` | 背景策略 |
| `stream` | boolean | 否 | `false` | 是否流式返回 |

### 成功返回示例

```json
{
  "data": [
    {
      "b64_json": "base64图片内容",
      "revised_prompt": "可能存在的改写提示词",
      "output_format": "png"
    }
  ]
}
```

## 参考图生图

### 接口

```http
POST https://aiwork.fans/v1/images/edits
```

### 单张参考图 curl

```bash
curl -X POST "https://aiwork.fans/v1/images/edits" \
  -H "Authorization: Bearer sk-你的APIKEY" \
  -F "model=gpt-image-2" \
  -F "prompt=额外加个苹果" \
  -F "size=1024x1024" \
  -F "n=1" \
  -F "quality=auto" \
  -F "response_format=b64_json" \
  -F "image[]=@/本地图片路径/reference.png;type=image/png"
```

### 多张参考图 curl

```bash
curl -X POST "https://aiwork.fans/v1/images/edits" \
  -H "Authorization: Bearer sk-你的APIKEY" \
  -F "model=gpt-image-2" \
  -F "prompt=把两张参考图融合成一张产品海报" \
  -F "size=1024x1024" \
  -F "n=1" \
  -F "quality=auto" \
  -F "response_format=b64_json" \
  -F "image[]=@/本地图片路径/1.png;type=image/png" \
  -F "image[]=@/本地图片路径/2.png;type=image/png"
```

### 请求头

| Header | 必填 | 说明 |
|---|---:|---|
| `Authorization` | 是 | `Bearer sk-你的APIKEY` |
| `Content-Type` | 否 | 使用 `curl -F` 时由 curl 自动生成 `multipart/form-data` |

### 表单字段

| 字段 | 类型 | 必填 | 推荐值 | 说明 |
|---|---:|---:|---|---|
| `model` | string | 是 | `gpt-image-2` | 生图模型 |
| `prompt` | string | 是 | - | 编辑或生成要求 |
| `size` | string | 是 | `1024x1024` | 图片尺寸 |
| `n` | number | 否 | `1` | 生成张数 |
| `quality` | string | 否 | `auto` | 图片质量 |
| `response_format` | string | 是 | `b64_json` | 返回 base64 图片 |
| `image[]` | file | 是 | - | 参考图文件，可传多张 |

### 成功返回示例

```json
{
  "data": [
    {
      "b64_json": "base64图片内容",
      "revised_prompt": "可能存在的改写提示词",
      "output_format": "png"
    }
  ]
}
```

## 常见错误

### `No available compatible accounts`

表示当前没有可调度的兼容账号，常见原因包括：

- 对应分组没有可用的 OpenAI 生图账号。
- 账号被上游拒绝后临时标记为不可用。
- 账号未绑定到当前 API Key 所属分组。
- 账号模型映射不支持当前请求的 `model`。

### `Upstream access forbidden`

表示请求已经转发到上游，但上游拒绝访问。遇到该错误后，服务端可能会临时标记该账号不可用，随后同类请求可能变成 `No available compatible accounts`。

## 接入注意事项

- 文生图不要默认使用 `response_format: "url"`。
- 推荐使用 `response_format: "b64_json"`。
- 参考图生图必须使用 `multipart/form-data`。
- 参考图字段名使用 `image[]`，不要改成 `images`。
- `curl -F` 上传文件时，不需要手动写 `Content-Type: multipart/form-data`，让 curl 自动生成 boundary。

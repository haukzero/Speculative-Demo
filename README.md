# Speculative Demo

一个简单的投机推理实现, 对于输入的 prompt, 仅支持 bs = 1

> NOTE: 在本示例项目中, 使用 facebook/opt 模型, 可切换为使用其他模型, 只需更换 `main.py` 中的 `MODEL_ZOO` 中模型位置即可

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Project Structure
```
.
├── kvcache_model.py   # 方便管理 kv-cache 的封装模型
├── main.py            # 主程序
├── sampling.py        # 自回归推理与投机推理实现
└── utils.py           # 常用函数
```
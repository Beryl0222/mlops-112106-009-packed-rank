# 位打包录音相似度

samples/scene.json 是本地合成的录音切片和按块读取交换。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 位打包距离与精确 top-k 查询

- `bitpack.py`：按设备字宽/字节序/位序解码分片包，按查询的有效位与权重构建打包投影，带权异或分子以机器字查表累计，不展开完整候选矩阵。
- `engine.py`：`Archive` 管理档案、版本沿革与工作区预算；`QuerySession` 绑定一次查询的版本与共享读块配额。未交付的可比较位只撑大距离区间上界；名次按 （距离， 编号） 的确定先后关系确认，剩余有效位无法改变排名时即签出，已签出不撤回。见证未定的确认项 `associable=false`，自动关联不得据以展开原片。`plan()`/`status()` 解释哪些块值得继续读、哪些距离已定不必再读；`to_dict()`/`from_dict()` 把未决查询交给值班端在同一档案上续取。
- `run_exchange.py`：按资料里的时间线驱动查询（open/continue/replace），输出每步的交付与解释。

```sh
python3 run_exchange.py samples/scene.json
python3 run_exchange.py samples/archival-gap.json
```

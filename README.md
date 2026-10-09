# CLIP Retrieval Demo

这个项目用 Jupyter Notebook 演示图文检索与评测：用 CLIP 编码 1000 张图片并保存特征，输入英文文本查看 Top-5 图片，也可用固定 ground truth 计算检索指标。

模型使用 OpenAI 的 `openai/clip-vit-base-patch32`。图片来自 Imagenette 160px：首次运行会下载约 94 MB 的压缩包，并从 10 个类别各固定抽取 100 张。整个过程只做推理，不训练模型。

## 在 WSL 中运行

在 VS Code 通过 Remote-SSH 连接 `wsl5070ti`，然后打开 WSL 原始项目目录：

```text
/home/wuqiang/projects/CLIP-Retrieval-Demo
```

在 VS Code 的 WSL 终端执行：

```bash
cd /home/wuqiang/projects/CLIP-Retrieval-Demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1" "torchvision==0.22.1"
python -m pip install -r requirements.txt
python -m ipykernel install --user --name clip-retrieval-demo --display-name "Python (CLIP Retrieval Demo)"
jupyter lab
```

打开 `clip_retrieval_demo.ipynb`，选择 `Python (CLIP Retrieval Demo)` 内核，然后按顺序运行单元格。Notebook 会优先使用 CUDA；没有可用 GPU 时会回退到 CPU。首次运行还会下载模型和图片，之后重复运行会复用本地文件及图片特征缓存。

模型默认从 Hugging Face 下载。如果 WSL 无法访问 Hugging Face 主站，可在 WSL 终端优先使用镜像，并从该终端启动 Jupyter：

```bash
export HF_ENDPOINT=https://hf-mirror.com
jupyter lab
```

如果已经在 WSL 下载好模型目录，可同时设置 `CLIP_MODEL_PATH=/path/to/clip-vit-base-patch32`。模型和数据均保存在被 Git 忽略的 `data/` 与 `cache/` 目录中。

## Notebook 内容

1. 下载并固定抽取 1000 张图片，每个类别 100 张，并展示十个类别各一张示例图片。
2. 加载 CLIP 图像和文本编码器，批量生成并归一化图片特征。
3. 将图片特征与清单写入缓存；图片清单或模型标识变化时重新编码。
4. 修改最后的 `query` 英文文本，查看 Top-5 图片和 cosine similarity。
5. 使用 100 条人工编写的逐图描述评测完整图库，并展示最多 5 条 Top-5 未命中案例。

## 运行检索评测

在 WSL 项目目录和已激活的 `.venv` 中执行：

```bash
python evaluate.py --ground-truth evaluation/ground_truth.json --output-dir evaluation/results
```

评测命令可独立准备 Imagenette 图片、加载 CLIP 模型，并复用 Notebook 的图片特征缓存。输出目录包含 `summary.json` 和逐 query 的 `queries.csv`；结果目录被 Git 忽略，ground truth 随项目版本管理。

Ground truth 每条记录包含唯一的 `query_id`、英文描述 `query` 和图库内项目相对路径 `image_path`。当前评测集按固定随机种子从十个类别中各选 10 张图片，再依据可见主体和场景细节编写描述。它是可复核的演示集，不是标准基准。描述可能同样适用于其他图库图片，而单目标标注只认可指定图片，因此这些指标衡量的是指定图片检索，可能低估语义上合理的结果。

Recall@K 表示正确图片排进前 K 名的 query 占比。MRR 是正确图片排名倒数的平均值；单目标二元相关性的 nDCG 为 `1 / log2(rank + 1)`，`nDCG@10` 将第 10 名之后的得分计为 0。目标图片在 Top-K 内即记为一次命中。

归一化后，两个向量的点积等于它们的 cosine similarity。检索直接对相似度排序，不经过 softmax。

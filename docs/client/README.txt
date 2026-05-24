南兴知识库 RAG — 绿色便携版
================================

【重要】本软件为完整绿色包，无需安装 Python，无需运行 uv / pip 等命令。
       解压后双击即可使用，全程在浏览器中完成配置与问答。

【运行】
  1. 将整个 NanxingRAG 文件夹解压到任意目录（路径尽量不含中文空格）
  2. 双击 NanxingRAG.exe
  3. 浏览器会自动打开安装向导或问答页

  若 exe 无法启动，可尝试同目录下的 NanxingRAG.bat（效果相同）。

【首次使用】
  1. 在 /setup 填写 LLM API 地址与 Key（需能访问您的 LLM 网关，无需 Hugging Face）
  2. 环境检测应显示内置向量 / Rerank / PDF 解析模型已就绪
  3. 上传 PDF 灌库，完成后进入问答页

【目录说明】
  NanxingRAG.exe   启动程序（推荐）
  runtime/         内置 Python 与依赖（勿删）
  web/             浏览器界面
  config/          配置（向导生成 config/.env）
  data/models/     内置 AI 模型（离线，勿删）
  data/rag_storage/  知识库数据
  logs/            运行日志

【离线说明】
  本安装包已内置 BGE-M3、BGE-Reranker、MinerU PDF 解析权重。
  客户环境无需、也不应访问 huggingface.co。

【技术支持】
  请联系交付方获取升级包或问题排查支持。

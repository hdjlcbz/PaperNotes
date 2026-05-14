# PaperNotes

论文阅读笔记，在线浏览：[hdjlcbz.github.io/PaperNotes](https://hdjlcbz.github.io/PaperNotes)

## 目录结构

```
YYYY-MM-DD/           # 按日期组织
└── paper-name/       # 论文笔记
    ├── *.md          # 图文笔记
    ├── *.html        # 独立 HTML 页面（MathJax + Mermaid）
    └── figures/      # 插图
```

## 使用

```bash
# 生成首页
./generate_index.sh > index.html
```

每篇笔记通过 paper2notes skill 生成 markdown，再通过 md2html skill 转为 HTML。

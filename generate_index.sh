#!/bin/bash
# Generate index.html by scanning date directories for HTML note files.
# Run from repo root: ./generate_index.sh > index.html

set -euo pipefail

cd "$(dirname "$0")"

cat <<'HEAD'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaperNotes — 论文阅读笔记</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,400..700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Inter", "Noto Sans SC", -apple-system, sans-serif;
    background: #f5f5f7;
    color: #1a1a2e;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px 80px;
    line-height: 1.6;
  }
  h1 {
    font-size: 2.2em;
    text-align: center;
    margin-bottom: 4px;
    color: #0d1117;
  }
  .subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 48px;
    font-size: 1em;
  }
  .paper-card {
    background: #fff;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s;
  }
  .paper-card:hover {
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
  }
  .paper-card a {
    font-size: 1.1em;
    font-weight: 600;
    color: #0969da;
    text-decoration: none;
  }
  .paper-card a:hover {
    text-decoration: underline;
  }
  .meta-line {
    font-size: 0.85em;
    color: #666;
    margin-top: 6px;
  }
  .meta-line .arxiv {
    color: #b62324;
    font-weight: 500;
  }
  .meta-line .venue {
    color: #1a7f37;
    font-weight: 600;
  }
  .tag {
    display: inline-block;
    background: #e8f0fe;
    color: #0969da;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78em;
    margin: 2px;
  }
  .date-heading {
    font-size: 1.3em;
    font-weight: 700;
    color: #0d1117;
    margin: 36px 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e1e4e8;
  }
  .footer {
    text-align: center;
    color: #999;
    margin-top: 60px;
    font-size: 0.85em;
  }
  .footer a { color: #0969da; }
</style>
</head>
<body>
<h1>PaperNotes</h1>
<p class="subtitle">论文阅读笔记 · 图文详解 · 公式与图表</p>
HEAD

REPO_OWNER="hdjlcbz"
REPO_NAME="PaperNotes"

# Scan date dirs in descending order
for date_dir in $(ls -d [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] 2>/dev/null | sort -r); do
    # Find all html files recursively
    html_files=$(find "$date_dir" -name "*.html" ! -name "index.html" 2>/dev/null | sort)
    if [ -z "$html_files" ]; then
        continue
    fi

    echo ""
    echo "<div class=\"date-heading\">$date_dir</div>"

    while IFS= read -r f; do
        title=$(grep -o '<title>[^<]*</title>' "$f" 2>/dev/null | head -1 | sed 's|<title>||;s|</title>||')
        if [ -z "$title" ]; then
            title=$(basename "$f" .html)
        fi
        title=$(echo "$title" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

        echo "<div class=\"paper-card\">"
        echo "  <a href=\"https://raw.githubusercontent.com/$REPO_OWNER/$REPO_NAME/main/$f\">$title</a>"
        echo "</div>"
    done <<< "$html_files"
done

cat <<'FOOT'
<p class="footer">
  Powered by <a href="https://github.com/hdjlcbz/PaperNotes">GitHub Pages</a> · Generated with Claude Code
</p>
</body>
</html>
FOOT

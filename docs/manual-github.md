# 将代码更新到你已经 clone 的 GitHub 仓库

你已有 `/Users/avian/Downloads/IMTS`，不用再次 clone。

## 1. 放入代码

将交付的 `IMTS_baselines_v1.zip` 下载到 Downloads。压缩包以 `IMTS/` 为根目录，包含代码与文档，不包含 `.git`、`.venv` 或数据集。

在 Mac Terminal 执行：

```bash
unzip -o /Users/avian/Downloads/IMTS_baselines_v1.zip -d /Users/avian/Downloads
cd /Users/avian/Downloads/IMTS
git status
git remote -v
```

此步骤往已有文件夹写入代码；原来的 Git 连接和本机 Python 环境保留。它会更新 README/.gitignore 等同名交付文件，适用于本次已确认没有现有代码的仓库。如果你已经新增了自己的代码，请先检查同名文件再覆盖。

`git remote -v` 应显示 `AvianHan/IMTS`，HTTPS 或 SSH URL 都可以。

## 2. 选择要记录的文件

```bash
git add README.md .gitignore pyproject.toml requirements*.txt configs src scripts slurm tests docs references
git diff --cached --stat
```

`git add` 把这些文件放入 staging area；此时尚未上传。这里明确选择本次代码文件，避免把本地其他文件一起加入。

## 3. 在本地保存一个版本

```bash
git commit -m "Add six Pyrregular baseline runners"
```

`git commit` 在你电脑上的 Git history 中新增一个版本；此时仍未上传。

## 4. 上传到 GitHub

```bash
git push
```

已有 clone 通常已经设置 upstream。如果提示当前 branch 没有 upstream：

```bash
git branch --show-current
```

若显示 `main`，再执行 `git push -u origin main`。不要使用 force push。

如果提示 `Author identity unknown`，需为当前 repo 设置你自己的 Git name/email；如果提示认证失败，则需要完成 GitHub 认证。不要把 password、token 或 private key 发到聊天里。把不含秘密的错误信息发来即可继续排查。

## 5. 验证

```bash
git status
git log -1 --oneline
```

应看到 working tree clean，并且 branch 与 origin 同步。浏览器刷新 GitHub repo 后应出现 `src`、`configs`、`scripts`、`slurm`、`tests` 等目录。

代码上传完成不代表开始 training。接下来按 `docs/orion.md` 准备服务器环境并提交 smoke job。

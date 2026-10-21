# 项目历史与恢复

统一入口：https://github.com/gezaifeng/color-card-detection

整合日期：2026-10-21（北京时间）。最早仓库的 `main` 直接向前扩展，无强制推送；现有仓库地址与关注记录继续保留。

## 时间与来源

四个独立仓库没有共同祖先。主线保留最早仓库 21 次原始提交，再按原作者日期重放其他仓库的 22 次提交，最后添加整合说明。包括 README 初始化、重复上传和空差异提交在内的每个快照都保留；切换阶段时，下一仓库最早的 README 初始化快照可能暂时只有 README，这是原始状态。

所有 43 个原始提交逐一对应到主线，文件树哈希完全相同。每个文件的路径、内容和 Git 文件模式随文件树保留。代码和 README 历史均未去重。新主线中的重放提交含 `Source-Repository`、`Source-Commit`、`Source-Author-Date`、`Reconstructed-On` 字段，原提交消息在正文保留。原作者日期保留，提交者日期如实记录整合时间，不伪装成当年已存在的一条历史。

日期仅来自 Git 元数据。尤其 `card_detection_9.18` 实际在 2026-01-19 提交，无法仅凭仓库名确定它是否在上一年 9 月开发。源仓库创建时间与每个提交映射保存在 `migration-manifest.json`。

## 阶段标签

| 标签 | 来源 | 原始版本终点 |
| --- | --- | --- |
| `stage-01-original-24patch` | `color-card-detection` | `4050dac805b9576ef79fa7b45fd02afcc55dbcc8` |
| `stage-02-9.18` | `card_detection_9.18` | `c4ef5a6a66d9b11eea21be38c92c8ed4b818a75e` |
| `stage-03-1.29` | `card_detection_1.29` | `2d28ba5bb9c3085a0b13c58724eaff2646fbd0f3` |
| `stage-04-2.1` | `card_detection_2.1` | `971297770eab2c1d6a7db6f9c73127dafbf1a563` |
| `stage-05-2026-04-01` | `card_detection_2.1` | `d7cad1a598998e0be7c5d7bf87a61822990ea370` |

## 原始历史归档

四条归档分支分别是：

- `archive/color-card-detection/main`：21 次原始提交，原终点 `4050dac805b9576ef79fa7b45fd02afcc55dbcc8`。
- `archive/card_detection_9.18/main`：3 次原始提交，原终点 `c4ef5a6a66d9b11eea21be38c92c8ed4b818a75e`。
- `archive/card_detection_1.29/main`：4 次原始提交，原终点 `2d28ba5bb9c3085a0b13c58724eaff2646fbd0f3`。
- `archive/card_detection_2.1/main`：15 次原始提交，原终点 `d7cad1a598998e0be7c5d7bf87a61822990ea370`。

## 查看、继续开发与恢复

继续开发使用 `main`。只读查看历史版本可切换标签；需从历史版本继续开发时创建分支，避免在标签的 detached HEAD 上遗失新提交：

```bash
git log --reverse --format="%h %aI %s" main
git show stage-01-original-24patch:"auto detection.py"
git switch -c experiment/from-24patch stage-01-original-24patch
git switch main
```

四份本地 `original.git` 镜像和 `original.bundle` 完整备份位于 `D:/Desktop/card-detection-git-backup-2026-10-21/<原仓库名>/`，随附 refs 清单、bundle SHA-256、GitHub 仓库元数据、issues/PR/releases 清单。备份不在公开仓库中。若需恢复任一旧仓库：

```bash
git clone "D:/Desktop/card-detection-git-backup-2026-10-21/card_detection_2.1/original.bundle" restored-card-detection
```

删除旧仓库的前提是：镜像和 bundle 校验通过，主线快照逐一核对通过，远端归档分支与标签通过独立克隆核对，且源仓库的远端 refs 自备份后未发生变化。删除结果以本地 `remote-verification.json` / `deletion-status.json` 为准；本文件不预先宣称删除成功。

本次仅整合上述四个色卡检测仓库，不合并光谱重建模型、数据集或完整应用仓库。附带的用户照片及 CR2 原始实验文件未上传。

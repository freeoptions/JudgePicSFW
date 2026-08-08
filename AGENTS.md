# JudgePicSFW 项目专属规则

通用规则见：`E:\@imFile-Download\AI-Useful-Prompt\通用开发工作规则.md`。

- 本项目是 Windows WPF C# 桌面应用，主项目在 `JudgePicSFW`，测试项目在 `JudgePicSFW.Tests`，另有 `PersonalAiTools` Python 辅助工具。
- 核心功能涉及图片分析、图像指纹和个人 AI 分类；修改分析结果、缓存、模型调用或图片解码时注意大文件性能和异常文件容错。
- 项目名与交付映射：`JudgePicSFW -> JudgePicSFW.exe`。
- 获得构建授权后只按 Release 流程生成并复制最终 EXE 到 `D:\@Software\JudgePicSFW\JudgePicSFW.exe`，不要复制 `bin`、`obj`、PDB 或临时模型缓存。
- 代码或 UI 修改完成后先检查现有测试项目；未经“11”“构建”或“打包”授权不构建最终 EXE。

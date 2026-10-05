# JudgePicSFW 项目专属规则

本项目继承 Codex 已加载的全局 `AGENTS.md`（本机：`C:\Users\freez\.codex\AGENTS.md`）；以下保留项目专属规则。

- 本项目是 Windows WPF C# 桌面应用，主项目在 `JudgePicSFW`，测试项目在 `JudgePicSFW.Tests`，另有 `PersonalAiTools` Python 辅助工具。
- 核心功能涉及图片分析、图像指纹和个人 AI 分类；修改分析结果、缓存、模型调用或图片解码时注意大文件性能和异常文件容错。
- 项目名与交付映射：`JudgePicSFW -> JudgePicSFW.exe`。
- 获得构建授权后只按 Release 流程生成并复制最终 EXE 到 `D:\@Software\JudgePicSFW\JudgePicSFW.exe`，不要复制 `bin`、`obj`、PDB 或临时模型缓存。
- 代码或 UI 修改完成后先检查现有测试项目；未经“11”“构建”或“打包”授权不构建最终 EXE。
- `JudgePicSFW.Tests` 是自定义断言控制台程序：以 Release 编译后直接运行其 `bin\Release\net8.0-windows\JudgePicSFW.Tests.exe`。`dotnet test` 不会执行这些断言；通过 `dotnet *.dll` 启动会使基于进程路径的日志目录落到 dotnet 安装目录。

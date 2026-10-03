# PowerShell 命令参考已拆分

原文档混合了项目构建流程与 PowerShell 语言说明，容易造成误解。现已拆分为两套文档：

## PowerShell 本身的逐条命令详解

从这里开始：

- [PowerShell 逐条命令详解总目录](powershell/README.md)
- [01：基础语法、帮助与命令发现](powershell/01-基础语法与帮助.md)
- [02：对象、管道、筛选、排序与格式化](powershell/02-对象与管道.md)
- [03：文件、目录、路径与文本](powershell/03-文件目录与文本.md)
- [04：进程、外部程序、环境变量与作业](powershell/04-进程环境变量与作业.md)
- [05：变量、集合、条件、循环、函数与脚本](powershell/05-脚本语言.md)
- [06：错误处理、调试、网络与系统管理](powershell/06-错误调试网络与系统.md)

## IndexTTS 项目操作流程

项目专用的构建、模型导出、CPU/GPU 合成和打包命令保留在：

- [IndexTTS Windows PowerShell 使用手册](WINDOWS_POWERSHELL.md)

PowerShell 模块和第三方软件可以注册数千条额外命令，因此“全部命令”应以当前机器查询结果为准：

```powershell
Get-Command
Get-Command -CommandType Cmdlet
Get-Help <命令名> -Full
Get-Help <命令名> -Examples
```

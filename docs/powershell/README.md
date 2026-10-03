# PowerShell 逐条命令详解

这组文档讲解 **PowerShell 语言与常用命令本身**，不以 IndexTTS、Rust 或任何特定项目的构建流程为主线。

PowerShell 自带及模块提供的 cmdlet 有数千个，不可能把每个第三方模块命令永久列全。因此这里覆盖：

- PowerShell 7 与 Windows PowerShell 5.1 的核心语法；
- 日常最常用的内置 cmdlet；
- 文件、对象、进程、环境变量、脚本、错误处理和系统管理；
- 查询任意未知命令的方法；
- 每条命令的用途、语法、参数、返回值、示例和注意事项。

## 分册

1. [基础语法、帮助与命令发现](01-基础语法与帮助.md)
2. [对象、管道、筛选、排序与格式化](02-对象与管道.md)
3. [文件、目录、路径与文本](03-文件目录与文本.md)
4. [进程、程序、环境变量与作业](04-进程环境变量与作业.md)
5. [变量、集合、条件、循环、函数与脚本](05-脚本语言.md)
6. [错误处理、调试、网络与系统管理](06-错误调试网络与系统.md)

## 如何查询“全部命令”

列出当前机器实际可用的所有命令：

```powershell
Get-Command
```

只列 cmdlet：

```powershell
Get-Command -CommandType Cmdlet
```

按模块列出：

```powershell
Get-Command -Module Microsoft.PowerShell.Management
```

搜索名称：

```powershell
Get-Command -Name '*Process*'
```

查看一条命令的完整帮助：

```powershell
Get-Help Get-Process -Full
```

查看示例：

```powershell
Get-Help Get-Process -Examples
```

查看在线文档：

```powershell
Get-Help Get-Process -Online
```

首次使用帮助系统时，可以管理员身份更新本地帮助：

```powershell
Update-Help
```

## 版本差异

查看版本：

```powershell
$PSVersionTable
```

- Windows PowerShell 5.1：系统内置，程序通常是 `powershell.exe`。
- PowerShell 7：跨平台新版，程序通常是 `pwsh.exe`。
- 文档优先采用两者都可用的写法；仅 PowerShell 7 支持的行为会特别注明。

## 阅读约定

```powershell
Command-Noun -Parameter Value
```

- `Command-Noun`：cmdlet 名称，通常遵循“动词-名词”。
- `-Parameter`：命名参数。
- `Value`：参数值。
- `[可选]`：可省略。
- `<必需值>`：必须由使用者提供，不要原样输入尖括号。

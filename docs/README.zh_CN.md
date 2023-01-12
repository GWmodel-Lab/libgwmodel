# 文档系统

## 依赖项

生成文档需要 [Doxygen](https://www.doxygen.nl/) 和 [Sphinx](https://www.sphinx-doc.org/) 。

如果没有 Doxygen ，请根据其官方网站的直到进行安装。

如果没有 Sphinx ，请使用 `pip` 命令和 [依赖项文件](./requirements.txt) 安装依赖。

```bash
pip install -r requirements.txt
```

## 生成

使用 `make` 命令生成文档并加上要生成的目标格式，例如 `html` 。

```bash
# Linux or macOS
DOXYGEN_LANGUAGE=zh_CN make html
```

```powershell
# Windows
$env:DOXYGEN_LANGUAGE="zh_CN"
make.bat html
```

然后文档会出现在 `/docs/_build` 目录中。

## 翻译

文档分为两个部分，一部分是在 `/docs` 文件夹下面专门编写的文档，用于介绍库的用法，称为“**使用文档**”；
另一部分是在源代码的中使用 DocString 格式编写的注释，用于介绍类、函数的详细信息，称为“**代码文档**”。
两部分文档需要分别进行翻译。

### 翻译代码文档

在代码注释中，使用 `\~chinese` 切换到中文模式（默认为英文模式）。
之后编写对应的中文文档内容即可。
例如

```cpp
/**
 * \~english
 * @brief Abstract algorithm class. This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms
 * 
 * \~chinese
 * @brief 抽象算法基类。
 * 该类无法被构造。该类型定义了一些在空间算法中常用的接口。
 * 
 */
class CGwmAlgorithm
```

如果英文部分前面有 `\~english`，那么这部分内容将不会在中文文档中出现。
如果没有，则也会在中文文档中出现。
如果暂时不能完全翻译，建议不要添加 `\~english` 以保证读者可以看到原文。

### 翻译使用文档

首先需要安装 `sphinx-intl` 

```bash
pip install sphinx-intl
```

生成可翻译消息（存储在 `/docs/_built/gettext/*.pot` 文件中）

```bash
DOXYGEN_LANGUAGE=zh_CN make gettext
```

生成翻译字符串（存储在 `/docs/locale/zh_CN/LC_MESSAGES/*.po` 文件中）

```bash
sphinx-intl update -p _build/gettext -l zh_CN
```

然后翻译 `.po` 文件中需要翻译的消息。
注意不要更改 `msgid` 后的内容，将翻译的文本放在 `msgstr` 后中。
例如

```po
msgid ""
"This library consist of C++ implementations of some geographically "
"weighted models. Currently, implemented models are:"
msgstr "该库包含一系列地理加权模型的 C++ 实现。目前已经实现的模型有"
```

翻译完成后，使用下面命令生成翻译过后的文档

```bash
DOXYGEN_LANGUAGE=zh_CN make html
```

注意，在 `locale/zh_CN/LC_MESSAGES/api` 目录中的 `.po` 文件也需要翻译。

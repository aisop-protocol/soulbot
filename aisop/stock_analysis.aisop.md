# AISOP：股票分析引擎

此 AISOP 定义了一个用于多维度股票分析的专业级执行流程。

> **协议版本**: AISOP 3.1
> **ID**: `soulbot.stock_analysis`

## 1. 视觉逻辑 (Mermaid)

```mermaid
graph TD
    Start[用户输入: 股票/代码] --> Parse[识别代码与市场]
    Parse --> Data[获取实时报价与基本面]
    Data --> Tech[分析技术趋势与指标]
    Tech --> Sentiment[获取近期新闻与情绪分析]
    Sentiment --> Synthesis[生成 SWOT 分析]
    Synthesis --> Conclusion[最终评估与免责声明]
    Conclusion --> End
```

## 2. 实现 (JSON)

```json
[
  {
    "role": "system",
    "content": {
        "protocol": "AISOP V1.0.0",
        "id": "soulbot.stock_analysis",
        "verified_on": ["Cursor", "Gemini CLI"],
        "tools": ["google_search", "terminal", "file_system", "web_browser"]
    }
  },
  {
    "role": "user",
    "content": {
        "instruction": "Execute aisop.stock_analysis",
        "aisop": {
            "main": "graph TD\n    Start[User Input: Stock/Ticker] --> Parse[Identify Ticker & Market]\n    Parse --> Data[Fetch Real-time Quotes & Fundamentals]\n    Data --> Tech[Analyze Technical Trends & Indicators]\n    Tech --> Sentiment[Fetch Recent News & Analyze Sentiment]\n    Sentiment --> Synthesis[Generate SWOT: Strengths, Weaknesses, Ops, Threats]\n    Synthesis --> Conclusion[Formulate Final Assessment & Disclaimer]\n    Conclusion --> End"
        },
        "functions": {
            "Data": { "step1": "Fetch P/E, Market Cap, 52W High/Low, and current price." },
            "Tech": { "step1": "Look for MA50/200 crossover, RSI levels, and volume trends." },
            "Sentiment": { "step1": "Scan news from Bloomberg, Reuters, or Yahoo Finance for recent catalysts." },
            "Conclusion": { "step1": "Provide a summary assessment. CRITICAL: Include 'Not Financial Advice' disclaimer." }
        }
    }
  }
]
```

## 3. 使用方法

要激活此 Soul，请将你的 `.env` 指向 `../aisop/stock_analysis.aisop.json`。

---
*生成自 `stock_analysis.aisop.json`*

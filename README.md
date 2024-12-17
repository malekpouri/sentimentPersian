# Persian Sentiment Analysis Using Deep Learning
# تحلیل احساسات متون فارسی با استفاده از یادگیری عمیق

[English](#english) | [فارسی](#persian)

<div id="english">

## Overview
This repository contains the implementation of a deep learning-based sentiment analysis system for Persian texts, achieving state-of-the-art accuracy of 86.01%. The system employs a novel hybrid architecture that combines word-level and character-level processing, specifically designed to handle the unique challenges of Persian text analysis.

### Key Features
- State-of-the-art accuracy (86.01%)
- Specialized Persian text preprocessing
- Three different architectures (LSTM, BiLSTM, Hybrid)
- Support for both formal and informal Persian texts
- Comprehensive evaluation metrics

## Requirements
```
tensorflow>=2.0.0
numpy>=1.19.2
pandas>=1.1.3
scikit-learn>=0.23.2
hazm>=0.7.0  # For Persian text processing
```

## Installation
```bash
# Clone the repository
git clone https://github.com/malekpouri/sentimentPersian.git

# Install dependencies
pip install -r requirements.txt
```

## Quick Start
```python
from sentiment_persian import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer(model_type='hybrid')

# Analyze sentiment
text = "غذا عالی بود و سرویس دهی هم سریع انجام شد"
sentiment = analyzer.predict(text)
print(sentiment)  # Output: positive
```

## Project Structure
```
├── data/
│   ├── raw/            # Raw data files
│   └── processed/      # Processed data files
├── models/
│   ├── lstm.py         # LSTM model implementation
│   ├── bilstm.py      # BiLSTM model implementation
│   └── hybrid.py      # Hybrid model implementation
├── preprocessing/
│   ├── normalizer.py  # Text normalization
│   └── tokenizer.py   # Text tokenization
├── utils/
│   └── data_loader.py # Data loading utilities
└── examples/          # Usage examples
```

## Model Architecture
The hybrid model combines word-level and character-level processing:

1. **Word-level Path:**
   - Embedding layer (100 dimensions)
   - LSTM layer (100 units)
   - Global Max Pooling

2. **Character-level Path:**
   - Character embedding (50 dimensions)
   - LSTM layer (50 units)
   - Global Max Pooling

3. **Combined Processing:**
   - Concatenation layer
   - Dropout (0.2)
   - Dense layer with sigmoid activation

## Performance Comparison
| Model   | Accuracy (%) | Parameters | Time/Epoch |
|---------|-------------|------------|------------|
| LSTM    | 85.81       | 1,080,501  | 35s        |
| BiLSTM  | 85.64       | 1,161,001  | 45s        |
| Hybrid  | 86.01       | 1,104,601  | 42s        |

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{malekpouri2024sentiment,
  title={Improving Persian Text Sentiment Analysis Using Deep Learning Architectures},
  author={Malekpouri, Mohammad and Kianifar, Mohammad Ali},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Contact
- Mohammad Malekpouri - malekpoor@aut.ac.ir
- Project Link: https://github.com/malekpouri/sentimentPersian/

</div>

<div id="persian" dir="rtl">

## نمای کلی
این مخزن شامل پیاده‌سازی سیستم تحلیل احساسات مبتنی بر یادگیری عمیق برای متون فارسی است که به دقت 86.01% دست یافته است. این سیستم از یک معماری ترکیبی جدید استفاده می‌کند که پردازش سطح کلمه و کاراکتر را ترکیب می‌کند و به طور خاص برای مواجهه با چالش‌های منحصر به فرد تحلیل متون فارسی طراحی شده است.

### ویژگی‌های کلیدی
- دقت پیشرفته (86.01%)
- پیش‌پردازش تخصصی متون فارسی
- سه معماری مختلف (LSTM، BiLSTM، Hybrid)
- پشتیبانی از متون فارسی رسمی و غیررسمی
- معیارهای ارزیابی جامع

## پیش‌نیازها
```
tensorflow>=2.0.0
numpy>=1.19.2
pandas>=1.1.3
scikit-learn>=0.23.2
hazm>=0.7.0  # برای پردازش متون فارسی
```

## نصب
```bash
# کلون کردن مخزن
git clone https://github.com/malekpouri/sentimentPersian.git

# نصب وابستگی‌ها
pip install -r requirements.txt
```

## شروع سریع
```python
from sentiment_persian import SentimentAnalyzer

# مقداردهی اولیه تحلیلگر
analyzer = SentimentAnalyzer(model_type='hybrid')

# تحلیل احساسات
text = "غذا عالی بود و سرویس دهی هم سریع انجام شد"
sentiment = analyzer.predict(text)
print(sentiment)  # خروجی: مثبت
```

## ساختار پروژه
```
├── data/
│   ├── raw/            # فایل‌های داده خام
│   └── processed/      # فایل‌های داده پردازش شده
├── models/
│   ├── lstm.py         # پیاده‌سازی مدل LSTM
│   ├── bilstm.py      # پیاده‌سازی مدل BiLSTM
│   └── hybrid.py      # پیاده‌سازی مدل Hybrid
├── preprocessing/
│   ├── normalizer.py  # نرمال‌سازی متن
│   └── tokenizer.py   # توکن‌سازی متن
├── utils/
│   └── data_loader.py # ابزارهای بارگذاری داده
└── examples/          # مثال‌های استفاده
```

## معماری مدل
مدل ترکیبی، پردازش سطح کلمه و کاراکتر را ترکیب می‌کند:

1. **مسیر سطح کلمه:**
   - لایه Embedding (100 بعدی)
   - لایه LSTM (100 واحد)
   - Global Max Pooling

2. **مسیر سطح کاراکتر:**
   - Embedding کاراکتر (50 بعدی)
   - لایه LSTM (50 واحد)
   - Global Max Pooling

3. **پردازش ترکیبی:**
   - لایه اتصال
   - Dropout (0.2)
   - لایه Dense با تابع فعال‌سازی sigmoid

## مقایسه عملکرد
| مدل     | دقت (%) | تعداد پارامترها | زمان/Epoch |
|---------|---------|----------------|------------|
| LSTM    | 85.81   | 1,080,501      | 35s        |
| BiLSTM  | 85.64   | 1,161,001      | 45s        |
| Hybrid  | 86.01   | 1,104,601      | 42s        |

## استناد
اگر از این کد در پژوهش خود استفاده می‌کنید، لطفاً به مقاله ما استناد کنید:
```bibtex
@article{malekpouri2024sentiment,
  title={Improving Persian Text Sentiment Analysis Using Deep Learning Architectures},
  author={Malekpouri, Mohammad and Kianifar, Mohammad Ali},
  year={2024}
}
```

## مجوز
این پروژه تحت مجوز MIT منتشر شده است - برای جزئیات به فایل [LICENSE](LICENSE) مراجعه کنید.

## مشارکت
از مشارکت‌ها استقبال می‌کنیم! لطفاً برای جزئیات به [CONTRIBUTING.md](CONTRIBUTING.md) مراجعه کنید.

## تماس
- محمد ملک پوری - malekpoor@aut.ac.ir
- لینک پروژه: https://github.com/malekpouri/sentimentPersian/

</div>

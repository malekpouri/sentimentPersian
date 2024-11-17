# src/preprocessing/normalizer.py

import re
import numpy as np
import pandas as pd

# Persian Stop Words
PERSIAN_STOP_WORDS = {
    'و', 'در', 'به', 'از', 'که', 'می', 'این', 'است', 'را', 'با', 'های', 'برای',
    'آن', 'یک', 'شود', 'شده', 'خود', 'ها', 'کرد', 'شد', 'ای', 'تا', 'کند', 'بر',
    'بود', 'گفت', 'نیز', 'وی', 'هم', 'کنند', 'دارد', 'ما', 'کرده', 'یا', 'اما',
    'باید', 'دو', 'اند', 'هر', 'خواهد', 'او', 'مورد', 'آنها', 'باشد', 'دیگر',
    'مردم', 'نمی', 'بین', 'پیش', 'پس', 'اگر', 'همه', 'صورت', 'یکی', 'هستند',
    'بی', 'من', 'دهد', 'هزار', 'نیست', 'استفاده', 'داد', 'داشته', 'راه', 'داشت',
    'چه', 'همچنین', 'کردند', 'داده', 'بوده', 'دارند', 'همین', 'میلیون', 'سوی',
    'شوند', 'بیشتر', 'بسیار', 'روی', 'گرفته', 'هایی', 'تواند', 'اول', 'نام',
    'هیچ', 'چند', 'جدید', 'بیش', 'شدن', 'کردن', 'کنیم', 'نشان', 'حتی', 'اینکه',
    'ولی', 'توسط', 'چنین', 'برخی', 'نه', 'دیروز', 'دوم', 'درباره', 'بعد', 'مختلف',
    'گیرد', 'شما', 'گفته', 'آنان', 'بار', 'طور', 'گرفت', 'دهند', 'گذاری', 'بسیاری',
    'طی', 'بودند', 'میلیارد', 'بدون', 'تمام', 'کل', 'تر', 'براساس', 'شدند', 'ترین',
    'امروز', 'باشند', 'ندارد', 'چون', 'قابل', 'گوید', 'دیگری', 'همان', 'خواهند',
    'قبل', 'آمده', 'اکنون', 'تحت', 'طریق', 'گیری', 'جای', 'هنوز', 'چرا', 'البته',
    'کنید', 'سازی', 'سوم', 'کنم', 'بلکه', 'زیر', 'توانند', 'ضمن', 'فقط', 'بودن',
    'حق', 'آید', 'وقتی', 'اش', 'یابد', 'نخستین', 'مقابل', 'خدمات', 'امسال', 'تاکنون',
    'مانند', 'تازه', 'آورد', 'فکر', 'آنچه', 'نخست', 'نشده', 'شاید', 'چهار', 'جریان',
    'پنج', 'ساخته', 'زیرا', 'نزدیک', 'برداری', 'کسی', 'ریزی', 'رفت', 'گردد', 'مثل',
    'آمد', 'ام', 'بهترین', 'دانست', 'کمتر', 'دادن', 'تمامی', 'جلوگیری', 'بیشتری',
    'ایم', 'ناشی', 'چیزی', 'آنکه', 'بالا', 'بنابراین', 'ایشان', 'بعضی', 'دادند',
    'داشتند', 'برخوردار', 'نخواهد', 'هنگام', 'نباید', 'غیر', 'نبود', 'دیده', 'وگو',
    'داریم', 'چگونه', 'بندی', 'خواست', 'فوق', 'ده', 'نوعی', 'هستیم', 'دیگران',
    'همچنان', 'سراسر', 'ندارند', 'گروهی', 'سعی', 'روزهای', 'آنجا', 'یکدیگر', 'کردم',
    'بیست', 'بروز', 'سپس', 'رفته', 'آورده', 'نماید', 'باشیم', 'گویند', 'زیاد', 'خویش',
    'همواره', 'گذاشته', 'شش', 'تر', 'ترین'
}

# Informal Words Dictionary
INFORMAL_DICT = {
    'خیلی': 'بسیار',
    'چیه': 'چیست',
    'چطور': 'چگونه',
    'چقدر': 'چه‌قدر',
    'واسه': 'برای',
    'دیگه': 'دیگر',
    'میشه': 'می‌شود',
    'نمیشه': 'نمی‌شود',
    'چی': 'چه',
    'خوبه': 'خوب است',
    'بده': 'بد است',
    'اینجوری': 'این‌گونه',
    'اونجوری': 'آن‌گونه',
    'چجوری': 'چگونه',
    'کجاس': 'کجاست',
    'چیکار': 'چه‌کار',
    'اینا': 'این‌ها',
    'اونا': 'آن‌ها',
    'خونه': 'خانه',
    'میدونم': 'می‌دانم',
    'نمیدونم': 'نمی‌دانم',
    'میخوام': 'می‌خواهم',
    'نمیخوام': 'نمی‌خواهم',
    'بودن': 'بوده‌اند',
    'نبودن': 'نبوده‌اند',
    'میکنم': 'می‌کنم',
    'نمیکنم': 'نمی‌کنم',
    'داداش': 'برادر',
    'آبجی': 'خواهر',
    'خاله': 'خاله',
    'عمه': 'عمه'
}


class PersianTextNormalizer:
    def __init__(self, remove_stopwords=True, convert_informal=True):
        self.remove_stopwords = remove_stopwords
        self.convert_informal = convert_informal

    def normalize(self, text):
        """Normalize Persian text"""
        if pd.isna(text):
            return ""

        # Convert to string if not already
        text = str(text)

        # Convert to lowercase (for any English text)
        text = text.lower()

        # Remove Arabic Unicode characters
        text = re.sub(r'[إأآ]', 'ا', text)
        text = re.sub(r'ي', 'ی', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'[‌]{2,}', '‌', text)  # Remove extra ZWNJ

        # Convert informal words to formal
        if self.convert_informal:
            for informal, formal in INFORMAL_DICT.items():
                text = text.replace(informal, formal)

        # Remove non-Persian characters
        text = re.sub(r'[^\u0600-\u06FF\s0-9\.,،!؟]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Remove stop words if enabled
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in PERSIAN_STOP_WORDS]
            text = ' '.join(words)

        return text

    def normalize_numbers(self, text):
        """Convert English numbers to Persian numbers"""
        persian_numbers = {
            '0': '۰',
            '1': '۱',
            '2': '۲',
            '3': '۳',
            '4': '۴',
            '5': '۵',
            '6': '۶',
            '7': '۷',
            '8': '۸',
            '9': '۹'
        }
        for eng, per in persian_numbers.items():
            text = text.replace(eng, per)
        return text

    def remove_duplicate_chars(self, text):
        """Remove duplicate characters that are used for emphasis"""
        pattern = r'(.)\1{2,}'
        return re.sub(pattern, r'\1', text)


# Example usage
if __name__ == "__main__":
    normalizer = PersianTextNormalizer()

    # Test text
    test_text = "خیلییییی خوبه! من نمیدونم چرا اینجوری شده؟؟؟ 123"
    normalized = normalizer.normalize(test_text)
    print("Original:", test_text)
    print("Normalized:", normalized)
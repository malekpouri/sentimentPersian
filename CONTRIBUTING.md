# Contributing Guidelines
# راهنمای مشارکت

[English](#english) | [فارسی](#persian)

<div id="english">

## How to Contribute
We love your contributions! We want to make contributing to Persian Sentiment Analysis as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Improving documentation
- Adding new language models or preprocessing techniques

## Development Process
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `master`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing style
6. Issue that pull request!

## Pull Request Process
1. Update the README.md with details of changes if needed
2. Update the requirements.txt if you add any dependencies
3. The PR will be merged once you have the sign-off of one other developer

## Code Style
- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add comments for complex operations
- Include docstrings for functions and classes

## Adding New Features

### For Machine Learning Models:
1. Place new model implementations in `models/`
2. Include performance metrics and comparisons
3. Add training examples in `examples/`
4. Update documentation with model architecture details

### For Preprocessing Tools:
1. Add new preprocessing functions in `preprocessing/`
2. Include examples of input and output
3. Document any Persian-specific considerations
4. Add tests for the new functionality

## Bug Reports
We use GitHub issues to track public bugs. Report a bug by opening a new issue.

Write bug reports with detail, background, and sample code:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening)

## License
By contributing, you agree that your contributions will be licensed under the MIT License.

</div>

<div id="persian" dir="rtl">

## نحوه مشارکت
ما از مشارکت شما استقبال می‌کنیم! می‌خواهیم مشارکت در پروژه تحلیل احساسات فارسی را تا حد ممکن ساده و شفاف کنیم، خواه:

- گزارش یک باگ باشد
- بحث درباره وضعیت فعلی کد
- ارائه یک اصلاح
- پیشنهاد ویژگی‌های جدید
- بهبود مستندات
- افزودن مدل‌های زبانی جدید یا تکنیک‌های پیش‌پردازش

## فرآیند توسعه
ما از GitHub برای میزبانی کد، پیگیری مشکلات و درخواست‌های ویژگی، و همچنین پذیرش pull request استفاده می‌کنیم.

1. یک fork از مخزن ایجاد کنید و شاخه خود را از `master` بسازید
2. اگر کدی اضافه کرده‌اید که باید تست شود، تست‌ها را اضافه کنید
3. اگر APIها را تغییر داده‌اید، مستندات را به‌روز کنید
4. اطمینان حاصل کنید که مجموعه تست‌ها پاس می‌شوند
5. مطمئن شوید کد شما از سبک موجود پیروی می‌کند
6. Pull request را ارسال کنید!

## فرآیند Pull Request
1. در صورت نیاز، README.md را با جزئیات تغییرات به‌روز کنید
2. اگر وابستگی جدیدی اضافه می‌کنید، requirements.txt را به‌روز کنید
3. PR پس از تأیید یک توسعه‌دهنده دیگر ادغام خواهد شد

## سبک کد
- از دستورالعمل‌های PEP 8 برای کد Python پیروی کنید
- از نام‌های معنادار برای متغیرها و توابع استفاده کنید
- برای عملیات پیچیده توضیحات اضافه کنید
- برای توابع و کلاس‌ها docstring اضافه کنید

## افزودن ویژگی‌های جدید

### برای مدل‌های یادگیری ماشین:
1. پیاده‌سازی‌های مدل جدید را در `models/` قرار دهید
2. معیارهای عملکرد و مقایسه‌ها را اضافه کنید
3. مثال‌های آموزشی را در `examples/` اضافه کنید
4. مستندات را با جزئیات معماری مدل به‌روز کنید

### برای ابزارهای پیش‌پردازش:
1. توابع پیش‌پردازش جدید را در `preprocessing/` اضافه کنید
2. مثال‌هایی از ورودی و خروجی را اضافه کنید
3. ملاحظات خاص زبان فارسی را مستند کنید
4. برای قابلیت جدید تست اضافه کنید

## گزارش باگ
ما از GitHub issues برای پیگیری باگ‌های عمومی استفاده می‌کنیم. با باز کردن یک issue جدید، باگ را گزارش کنید.

گزارش‌های باگ را با جزئیات، پیش‌زمینه و کد نمونه بنویسید:

- خلاصه و/یا پیش‌زمینه
- مراحل بازتولید
  - دقیق باشید!
  - در صورت امکان کد نمونه ارائه دهید
- انتظار داشتید چه اتفاقی بیفتد
- چه اتفاقی واقعاً می‌افتد
- یادداشت‌ها (احتمالاً شامل اینکه چرا فکر می‌کنید این اتفاق می‌افتد)

## مجوز
با مشارکت، موافقت می‌کنید که مشارکت‌های شما تحت مجوز MIT منتشر شود.

</div>

"""
اسکریپت تست برای اطمینان از نصب صحیح محیط شبیه‌سازی
"""

from drilling_env.drilling_env import DrillingEnv
import numpy as np

def test_environment():
    print("تست محیط شبیه‌سازی حفاری:")
    print("-" * 40)
    
    # ایجاد محیط
    env = DrillingEnv()
    print("✓ محیط با موفقیت ایجاد شد")
    
    # تست reset
    obs = env.reset()
    print("✓ تابع reset با موفقیت اجرا شد")
    print(f"مشاهدات اولیه: {obs}")
    
    # تست step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("\n✓ تابع step با موفقیت اجرا شد")
    print(f"عمل انجام شده: {action}")
    print(f"پاداش دریافتی: {reward:.2f}")
    
    # تست render
    print("\nنمایش وضعیت محیط:")
    env.render()
    
    print("\n✓ تمام تست‌ها با موفقیت انجام شدند")

if __name__ == "__main__":
    test_environment() 
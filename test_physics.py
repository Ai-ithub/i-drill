"""
اسکریپت تست برای کلاس DrillingPhysics با قابلیت‌های جدید
"""

from drilling_env.drilling_physics import DrillingPhysics, FormationType, AbnormalCondition
import numpy as np

def test_drilling_physics():
    # ایجاد نمونه از کلاس
    physics = DrillingPhysics()
    print("✅ کلاس DrillingPhysics با موفقیت ایجاد شد\n")

    # تست 1: محاسبه وزن مؤثر روی مته در زوایای مختلف
    print("🔍 تست وزن مؤثر روی مته:")
    test_wob = 20000  # نیوتن
    test_angles = [0, 30, 45, 60, 90]  # درجه
    
    collar_weight = physics.drill_collar.calculate_weight()
    print(f"- وزن گیره حفاری: {collar_weight/1000:.1f} کیلونیوتن")
    
    for angle in test_angles:
        effective_wob = physics.calculate_effective_wob(test_wob, angle)
        print(f"- زاویه {angle} درجه:")
        print(f"  • وزن اعمالی از سطح: {test_wob/1000:.1f} کیلونیوتن")
        print(f"  • وزن مؤثر روی مته: {effective_wob/1000:.1f} کیلونیوتن")
    print()

    # تست 2: نرخ نفوذ در سازندهای مختلف و زوایای مختلف
    print("🔍 تست نرخ نفوذ در شرایط مختلف:")
    test_rpm = 100    # دور بر دقیقه
    
    for formation in [FormationType.SOFT_SAND, FormationType.HARD_SHALE]:
        physics.current_formation = formation
        print(f"\nسازند: {formation.value}")
        
        for angle in [0, 45]:
            rop = physics.calculate_rop(test_wob, test_rpm, angle)
            print(f"- زاویه {angle} درجه:")
            print(f"  • نرخ نفوذ: {rop:.2f} متر بر ساعت")
    print()

    # تست 3: محاسبه گشتاور با اثر ژیروسکوپی
    print("🔍 تست محاسبه گشتاور:")
    test_bit_wear = 0.3  # 30% فرسایش
    test_rpms = [50, 100, 150]
    
    for rpm in test_rpms:
        torque = physics.calculate_torque(test_wob, test_bit_wear, rpm)
        print(f"- در {rpm} دور بر دقیقه:")
        print(f"  • گشتاور کل: {torque:.2f} نیوتن متر")
    print()

    # تست 4: شبیه‌سازی کامل
    print("🔍 تست شبیه‌سازی کامل:")
    current_state = {
        'depth': 1000.0,
        'bit_wear': 0.3,
        'rop': 10.0,
        'torque': 2000.0,
        'pressure': 15000000.0,
        'vibration_axial': 0.4,
        'vibration_lateral': 0.3,
        'vibration_torsional': 0.2
    }
    
    action = {
        'wob': test_wob,
        'rpm': test_rpm,
        'flow_rate': 0.05
    }
    
    # تست در سازندهای و زوایای مختلف
    test_conditions = [
        (FormationType.SOFT_SAND, 0),
        (FormationType.SOFT_SAND, 45),
        (FormationType.HARD_SHALE, 0),
        (FormationType.HARD_SHALE, 45)
    ]
    
    for formation, angle in test_conditions:
        physics.current_formation = formation
        print(f"\nسازند: {formation.value}, زاویه: {angle} درجه")
        
        new_state, _ = physics.simulate_step(current_state, action, 60, angle)
        
        print("وضعیت جدید:")
        print(f"- عمق: {new_state['depth']:.1f} متر")
        print(f"- دما: {new_state['temperature']:.1f} درجه سانتیگراد")
        print(f"- نرخ نفوذ: {new_state['rop']:.2f} متر بر ساعت")
        print(f"- وزن مؤثر روی مته: {new_state['effective_wob']/1000:.1f} کیلونیوتن")
        print(f"- گشتاور: {new_state['torque']:.1f} نیوتن متر")
        print(f"- فرسایش مته: {new_state['bit_wear']*100:.1f}%")
        print(f"- شرایط: {new_state['condition']}")

if __name__ == "__main__":
    test_drilling_physics() 
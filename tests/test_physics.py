"""pytest tests for drilling physics"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drilling_env.drilling_physics import DrillingPhysics, FormationType, AbnormalCondition
import numpy as np
import pytest


class TestDrillingPhysics:
    """Test class for drilling physics functionality"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.physics = DrillingPhysics()
    
    def test_physics_creation(self):
        """Test that physics object can be created successfully"""
        assert self.physics is not None
        assert hasattr(self.physics, 'drill_collar')
        assert hasattr(self.physics, 'current_formation')
    
    def test_effective_wob_calculation(self):
        """Test effective weight on bit calculation at different angles"""
        test_wob = 20000  # Newtons
        test_angles = [0, 30, 45, 60, 90]  # degrees
        
        collar_weight = self.physics.drill_collar.calculate_weight()
        assert collar_weight > 0, "Drill collar weight should be positive"
        
        for angle in test_angles:
            effective_wob = self.physics.calculate_effective_wob(test_wob, angle)
            
            # Effective WOB should be positive
            assert effective_wob > 0, f"Effective WOB should be positive at {angle} degrees"
            
            # At 0 degrees, effective WOB should be maximum
            if angle == 0:
                max_effective_wob = effective_wob
            else:
                assert effective_wob <= max_effective_wob, f"Effective WOB should decrease with angle"
    
    def test_rop_calculation(self):
        """Test rate of penetration calculation in different formations"""
        test_wob = 20000  # Newtons
        test_rpm = 100    # RPM
        test_angles = [0, 45]  # degrees
        
        formations = [FormationType.SOFT_SAND, FormationType.HARD_SHALE]
        
        for formation in formations:
            self.physics.current_formation = formation
            
            for angle in test_angles:
                rop = self.physics.calculate_rop(test_wob, test_rpm, angle)
                
                # ROP should be positive
                assert rop > 0, f"ROP should be positive for {formation.value} at {angle} degrees"
                
                # ROP should be finite
                assert np.isfinite(rop), f"ROP should be finite for {formation.value} at {angle} degrees"
        
        # Soft sand should generally have higher ROP than hard shale
        self.physics.current_formation = FormationType.SOFT_SAND
        rop_soft = self.physics.calculate_rop(test_wob, test_rpm, 0)
        
        self.physics.current_formation = FormationType.HARD_SHALE
        rop_hard = self.physics.calculate_rop(test_wob, test_rpm, 0)
        
        assert rop_soft > rop_hard, "Soft sand should have higher ROP than hard shale"
    
    def test_torque_calculation(self):
        """Test torque calculation with gyroscopic effects"""
        test_wob = 20000  # Newtons
        test_bit_wear = 0.3  # 30% wear
        test_rpms = [50, 100, 150]
        
        for rpm in test_rpms:
            torque = self.physics.calculate_torque(test_wob, test_bit_wear, rpm)
            
            # Torque should be positive
            assert torque > 0, f"Torque should be positive at {rpm} RPM"
            
            # Torque should be finite
            assert np.isfinite(torque), f"Torque should be finite at {rpm} RPM"
        
        # Higher RPM should generally result in higher torque due to gyroscopic effects
        torque_low = self.physics.calculate_torque(test_wob, test_bit_wear, 50)
        torque_high = self.physics.calculate_torque(test_wob, test_bit_wear, 150)
        
        assert torque_high > torque_low, "Higher RPM should result in higher torque"
    
    def test_bit_wear_calculation(self):
        """Test bit wear calculation"""
        initial_wear = 0.2
        rop = 10.0  # m/h
        dt = 60  # seconds
        
        new_wear = self.physics.calculate_bit_wear(initial_wear, rop, dt)
        
        # Wear should increase
        assert new_wear > initial_wear, "Bit wear should increase over time"
        
        # Wear should not exceed 1.0 (100%)
        assert new_wear <= 1.0, "Bit wear should not exceed 100%"
        
        # Wear should be finite
        assert np.isfinite(new_wear), "Bit wear should be finite"
    
    def test_pressure_drop_calculation(self):
        """Test pressure drop calculation"""
        flow_rate = 0.05  # m³/s
        depth = 1000.0  # meters
        
        pressure_drop = self.physics.calculate_pressure_drop(flow_rate, depth)
        
        # Pressure drop should be positive
        assert pressure_drop > 0, "Pressure drop should be positive"
        
        # Pressure drop should be finite
        assert np.isfinite(pressure_drop), "Pressure drop should be finite"
        
        # Deeper wells should have higher pressure drop
        pressure_shallow = self.physics.calculate_pressure_drop(flow_rate, 500.0)
        pressure_deep = self.physics.calculate_pressure_drop(flow_rate, 1500.0)
        
        assert pressure_deep > pressure_shallow, "Deeper wells should have higher pressure drop"
    
    def test_vibration_calculation(self):
        """Test vibration calculation"""
        wob = 20000  # Newtons
        rpm = 100   # RPM
        bit_wear = 0.3
        
        vibrations = self.physics.calculate_vibrations(wob, rpm, bit_wear)
        
        # Should return a dictionary with three vibration types
        assert isinstance(vibrations, dict), "Vibrations should be returned as dictionary"
        assert 'axial' in vibrations, "Should include axial vibration"
        assert 'lateral' in vibrations, "Should include lateral vibration"
        assert 'torsional' in vibrations, "Should include torsional vibration"
        
        # All vibration values should be non-negative and finite
        for vib_type, value in vibrations.items():
            assert value >= 0, f"{vib_type} vibration should be non-negative"
            assert np.isfinite(value), f"{vib_type} vibration should be finite"
    
    def test_abnormal_condition_detection(self):
        """Test abnormal condition detection"""
        # Test normal conditions
        normal_state = {
            'torque': 1500.0,
            'vibration_axial': 0.3,
            'vibration_lateral': 0.2,
            'vibration_torsional': 0.1,
            'rop': 10.0
        }
        
        condition = self.physics.check_abnormal_conditions(normal_state)
        assert condition == AbnormalCondition.NORMAL, "Should detect normal conditions"
        
        # Test high vibration conditions
        high_vibration_state = normal_state.copy()
        high_vibration_state['vibration_axial'] = 0.9
        
        condition = self.physics.check_abnormal_conditions(high_vibration_state)
        assert condition != AbnormalCondition.NORMAL, "Should detect abnormal vibration"
    
    def test_complete_simulation_step(self):
        """Test complete simulation step integration"""
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
            'wob': 20000,
            'rpm': 100,
            'flow_rate': 0.05
        }
        
        test_conditions = [
            (FormationType.SOFT_SAND, 0),
            (FormationType.SOFT_SAND, 45),
            (FormationType.HARD_SHALE, 0),
            (FormationType.HARD_SHALE, 45)
        ]
        
        for formation, angle in test_conditions:
            self.physics.current_formation = formation
            
            new_state, abnormal_condition = self.physics.simulate_step(
                current_state, action, 60, angle
            )
            
            # Check that new state has all required keys
            required_keys = ['depth', 'temperature', 'rop', 'effective_wob', 
                           'torque', 'bit_wear', 'condition']
            
            for key in required_keys:
                assert key in new_state, f"New state should contain {key}"
            
            # Check that values are reasonable
            assert new_state['depth'] >= current_state['depth'], "Depth should not decrease"
            assert 0 <= new_state['bit_wear'] <= 1, "Bit wear should be between 0 and 1"
            assert new_state['rop'] > 0, "ROP should be positive"
            assert new_state['torque'] > 0, "Torque should be positive"
            assert new_state['effective_wob'] > 0, "Effective WOB should be positive"
            
            # Check that abnormal condition is valid
            assert isinstance(abnormal_condition, AbnormalCondition), "Should return valid abnormal condition"


def test_formation_types():
    """Test that all formation types are properly defined"""
    formations = [FormationType.SOFT_SAND, FormationType.MEDIUM_SAND, 
                 FormationType.HARD_SAND, FormationType.SOFT_SHALE,
                 FormationType.HARD_SHALE, FormationType.LIMESTONE,
                 FormationType.DOLOMITE]
    
    for formation in formations:
        assert hasattr(formation, 'value'), f"Formation {formation} should have a value"
        assert isinstance(formation.value, str), f"Formation value should be string"


def test_abnormal_conditions():
    """Test that all abnormal conditions are properly defined"""
    conditions = [AbnormalCondition.NORMAL, AbnormalCondition.BIT_BALLING,
                 AbnormalCondition.STICK_SLIP, AbnormalCondition.HIGH_VIBRATION]
    
    for condition in conditions:
        assert hasattr(condition, 'value'), f"Condition {condition} should have a value"
        assert isinstance(condition.value, str), f"Condition value should be string"


if __name__ == "__main__":
    pytest.main([__file__])
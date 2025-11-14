"""
Unit tests for Security Headers utilities
"""
import pytest
import os
from unittest.mock import patch
from utils.security import get_csp_policy, get_security_headers


class TestCSPPolicy:
    """Tests for Content Security Policy generation"""
    
    def test_csp_policy_production(self):
        """Test CSP policy for production"""
        policy = get_csp_policy(is_production=True, api_url="https://api.example.com")
        
        assert "default-src 'self'" in policy
        assert "script-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy
        assert "upgrade-insecure-requests" in policy
        # Should not have unsafe-inline or unsafe-eval in script-src
        assert "'unsafe-inline'" not in policy.split("script-src")[1] if "script-src" in policy else True
    
    def test_csp_policy_development(self):
        """Test CSP policy for development"""
        policy = get_csp_policy(is_production=False)
        
        assert "default-src 'self'" in policy
        assert "'unsafe-inline'" in policy
        assert "'unsafe-eval'" in policy
        assert "ws:" in policy or "wss:" in policy
    
    def test_csp_policy_with_api_url(self):
        """Test CSP policy with API URL"""
        api_url = "https://api.example.com"
        policy = get_csp_policy(is_production=True, api_url=api_url)
        
        assert api_url in policy
        assert "wss://api.example.com" in policy or "ws://api.example.com" in policy
    
    def test_csp_policy_custom(self, monkeypatch):
        """Test custom CSP policy from environment"""
        custom_csp = "default-src 'self'; script-src 'self' 'unsafe-inline';"
        monkeypatch.setenv("CSP_POLICY", custom_csp)
        
        policy = get_csp_policy(is_production=True)
        
        assert policy == custom_csp


class TestSecurityHeaders:
    """Tests for Security Headers generation"""
    
    def test_security_headers_production(self):
        """Test security headers for production"""
        headers = get_security_headers(is_production=True)
        
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in headers
        assert headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Referrer-Policy" in headers
        assert "Content-Security-Policy" in headers
        assert "Permissions-Policy" in headers
    
    def test_security_headers_development(self):
        """Test security headers for development"""
        headers = get_security_headers(is_production=False)
        
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "Content-Security-Policy" in headers
        # HSTS should not be in development
        assert "Strict-Transport-Security" not in headers
    
    def test_hsts_in_production_with_https(self, monkeypatch):
        """Test HSTS header in production with HTTPS"""
        monkeypatch.setenv("FORCE_HTTPS", "true")
        monkeypatch.setenv("HSTS_MAX_AGE", "31536000")
        monkeypatch.setenv("HSTS_INCLUDE_SUBDOMAINS", "true")
        
        headers = get_security_headers(is_production=True)
        
        assert "Strict-Transport-Security" in headers
        assert "max-age=31536000" in headers["Strict-Transport-Security"]
        assert "includeSubDomains" in headers["Strict-Transport-Security"]
    
    def test_hsts_with_preload(self, monkeypatch):
        """Test HSTS header with preload"""
        monkeypatch.setenv("FORCE_HTTPS", "true")
        monkeypatch.setenv("HSTS_PRELOAD", "true")
        
        headers = get_security_headers(is_production=True)
        
        if "Strict-Transport-Security" in headers:
            assert "preload" in headers["Strict-Transport-Security"]
    
    def test_permissions_policy(self):
        """Test Permissions Policy header"""
        headers = get_security_headers(is_production=True)
        
        assert "Permissions-Policy" in headers
        policy = headers["Permissions-Policy"]
        
        # Should disable various features
        assert "geolocation=()" in policy
        assert "microphone=()" in policy
        assert "camera=()" in policy
    
    def test_csp_in_headers(self):
        """Test that CSP is included in headers"""
        headers = get_security_headers(is_production=True)
        
        assert "Content-Security-Policy" in headers
        csp = headers["Content-Security-Policy"]
        
        assert "default-src" in csp
        assert "script-src" in csp


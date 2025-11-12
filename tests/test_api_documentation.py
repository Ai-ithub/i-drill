"""
Tests for API documentation completeness
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json


def test_openapi_schema_exists(app):
    """Test that OpenAPI schema is accessible"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    assert schema["info"]["title"] == "i-Drill API"
    assert schema["info"]["version"] == "1.0.0"


def test_openapi_schema_structure(app):
    """Test OpenAPI schema structure"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    # Check required OpenAPI fields
    assert "openapi" in schema
    assert schema["openapi"].startswith("3.")
    assert "info" in schema
    assert "paths" in schema
    assert "components" in schema
    
    # Check info section
    info = schema["info"]
    assert "title" in info
    assert "version" in info
    assert "description" in info


def test_all_endpoints_have_descriptions(app):
    """Test that all endpoints have descriptions"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    paths = schema.get("paths", {})
    
    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                assert "summary" in details or "description" in details, \
                    f"Endpoint {method.upper()} {path} missing description"


def test_all_endpoints_have_responses(app):
    """Test that all endpoints define responses"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    paths = schema.get("paths", {})
    
    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                assert "responses" in details, \
                    f"Endpoint {method.upper()} {path} missing responses"
                
                # Check for at least one response code
                responses = details["responses"]
                assert len(responses) > 0, \
                    f"Endpoint {method.upper()} {path} has no response codes"


def test_error_responses_defined(app):
    """Test that error responses are defined in schema"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    # Check that error schemas exist
    components = schema.get("components", {})
    schemas = components.get("schemas", {})
    
    # Check for error response schemas
    error_schemas = [name for name in schemas.keys() if "error" in name.lower() or "Error" in name]
    assert len(error_schemas) > 0, "No error response schemas found"


def test_request_models_defined(app):
    """Test that request models are properly defined"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    components = schema.get("components", {})
    schemas = components.get("schemas", {})
    
    # Check for common request models
    request_models = [
        "SensorDataPoint",
        "RULPredictionRequest",
        "CreateMaintenanceAlertRequest"
    ]
    
    for model_name in request_models:
        if model_name in schemas:
            model = schemas[model_name]
            assert "type" in model or "properties" in model, \
                f"Request model {model_name} is not properly defined"


def test_response_models_defined(app):
    """Test that response models are properly defined"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    components = schema.get("components", {})
    schemas = components.get("schemas", {})
    
    # Check for common response models
    response_models = [
        "SensorDataResponse",
        "RULPredictionResponse",
        "MaintenanceAlert"
    ]
    
    found_models = []
    for model_name in response_models:
        if model_name in schemas:
            found_models.append(model_name)
            model = schemas[model_name]
            assert "type" in model or "properties" in model, \
                f"Response model {model_name} is not properly defined"
    
    assert len(found_models) > 0, "No response models found in schema"


def test_tags_defined(app):
    """Test that API tags are defined"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    tags = schema.get("tags", [])
    assert len(tags) > 0, "No tags defined in OpenAPI schema"
    
    # Check for common tags
    tag_names = [tag.get("name", "") for tag in tags]
    expected_tags = ["Health", "Sensor Data", "Predictions", "Maintenance", "Authentication"]
    
    for expected_tag in expected_tags:
        assert any(expected_tag.lower() in name.lower() for name in tag_names), \
            f"Tag '{expected_tag}' not found in schema"


def test_servers_defined(app):
    """Test that servers are defined in schema"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    servers = schema.get("servers", [])
    assert len(servers) > 0, "No servers defined in OpenAPI schema"
    
    for server in servers:
        assert "url" in server, "Server missing URL"
        assert "description" in server, "Server missing description"


def test_contact_info_defined(app):
    """Test that contact information is defined"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    info = schema.get("info", {})
    assert "contact" in info, "Contact information not defined"
    
    contact = info["contact"]
    assert "name" in contact or "url" in contact, "Contact information incomplete"


def test_license_info_defined(app):
    """Test that license information is defined"""
    client = TestClient(app)
    response = client.get("/openapi.json")
    schema = response.json()
    
    info = schema.get("info", {})
    assert "license" in info, "License information not defined"
    
    license_info = info["license"]
    assert "name" in license_info, "License name not defined"


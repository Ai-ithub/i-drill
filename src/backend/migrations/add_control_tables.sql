-- Migration: Add Control Tables
-- Description: Adds table for change request tracking
-- Date: 2025-01-XX

-- Create change_requests table
CREATE TABLE IF NOT EXISTS change_requests (
    id SERIAL PRIMARY KEY,
    rig_id VARCHAR(50) NOT NULL,
    change_type VARCHAR(20) NOT NULL,
    component VARCHAR(100) NOT NULL,
    parameter VARCHAR(100) NOT NULL,
    old_value TEXT NULL,
    new_value TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' NOT NULL,
    auto_execute BOOLEAN DEFAULT FALSE,
    requested_by INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
    approved_by INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
    applied_by INTEGER NULL REFERENCES users(id) ON DELETE SET NULL,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    approved_at TIMESTAMP NULL,
    applied_at TIMESTAMP NULL,
    rejection_reason TEXT NULL,
    error_message TEXT NULL,
    metadata JSONB NULL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_change_requests_rig_id ON change_requests(rig_id);
CREATE INDEX IF NOT EXISTS idx_change_requests_change_type ON change_requests(change_type);
CREATE INDEX IF NOT EXISTS idx_change_requests_status ON change_requests(status);
CREATE INDEX IF NOT EXISTS idx_change_requests_requested_at ON change_requests(requested_at);
CREATE INDEX IF NOT EXISTS idx_change_requests_requested_by ON change_requests(requested_by);

-- Add comments
COMMENT ON TABLE change_requests IS 'Tracks all change requests for drilling parameters';
COMMENT ON COLUMN change_requests.change_type IS 'Type of change: optimization, maintenance, or validation';
COMMENT ON COLUMN change_requests.status IS 'Status: pending, approved, rejected, applied, or failed';
COMMENT ON COLUMN change_requests.auto_execute IS 'Whether the change was auto-executed';
COMMENT ON COLUMN change_requests.metadata IS 'Additional JSON data about the change';


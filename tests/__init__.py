"""
🎾 Tennis Match Prediction - Tests Package
==========================================

Comprehensive testing suite for the tennis match prediction system.

Test Categories:
- System validation and integration tests
- Performance verification and benchmarking  
- Component testing and diagnostics
- Error handling and edge case testing

Test Files:
- comprehensive_test.py: Complete end-to-end system validation
- final_verification.py: Performance verification and metrics
- test_components.py: Individual component testing
- error_diagnostic.py: System diagnostics and troubleshooting
- quick_h2h_test.py: Head-to-head feature testing
"""

__version__ = "2.0.0"
__author__ = "Tennis Prediction Team"

# Test configuration
TEST_CONFIG = {
    "test_data_size": 100,  # Matches to use for testing
    "performance_threshold": 1.0,  # Max seconds per prediction
    "cache_hit_rate_threshold": 0.8,  # Minimum cache hit rate
    "memory_limit_mb": 2048,  # Memory usage limit
}

__all__ = ['TEST_CONFIG']

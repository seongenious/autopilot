"""Tests for Registry class."""

import pytest

from autopilot.utils.registry import Registry


class TestRegistry:
    """Test cases for Registry class."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        registry = Registry('test')

        @registry.register_module()
        class TestModule:
            pass

        assert 'TestModule' in registry
        assert registry.get('TestModule') is TestModule

    def test_register_with_name(self):
        """Test registration with custom name."""
        registry = Registry('test')

        @registry.register_module(name='CustomName')
        class TestModule:
            pass

        assert 'CustomName' in registry
        assert 'TestModule' not in registry

    def test_build(self):
        """Test building module from config."""
        registry = Registry('test')

        @registry.register_module()
        class TestModule:
            def __init__(self, value: int):
                self.value = value

        module = registry.build({'type': 'TestModule', 'value': 42})
        assert module.value == 42

    def test_build_missing_type(self):
        """Test build fails without type key."""
        registry = Registry('test')

        with pytest.raises(KeyError):
            registry.build({'value': 42})

    def test_build_unregistered_type(self):
        """Test build fails for unregistered type."""
        registry = Registry('test')

        with pytest.raises(KeyError):
            registry.build({'type': 'NonExistent'})

    def test_duplicate_registration(self):
        """Test duplicate registration raises error."""
        registry = Registry('test')

        @registry.register_module()
        class TestModule:
            pass

        with pytest.raises(KeyError):
            @registry.register_module()
            class TestModule:  # noqa: F811
                pass

    def test_force_registration(self):
        """Test force registration overrides existing."""
        registry = Registry('test')

        @registry.register_module()
        class TestModule:
            value = 1

        @registry.register_module(force=True)
        class TestModule:  # noqa: F811
            value = 2

        assert registry.get('TestModule').value == 2

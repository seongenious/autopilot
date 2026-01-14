"""Registry module for managing model components.

Inspired by mmdetection3d's registry pattern, this module provides a centralized
way to register and retrieve model components (backbones, necks, heads, etc.)
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar('T')


class Registry:
    """A registry to map strings to classes or functions.

    This allows for dynamic instantiation of components based on configuration,
    enabling modular and extensible architecture.

    Example:
        >>> BACKBONES = Registry('backbones')
        >>> @BACKBONES.register_module()
        ... class ResNet:
        ...     pass
        >>> model = BACKBONES.build({'type': 'ResNet', 'depth': 50})
    """

    def __init__(self, name: str, parent: Optional['Registry'] = None) -> None:
        """Initialize a Registry.

        Args:
            name: Registry name (e.g., 'backbones', 'heads').
            parent: Parent registry for hierarchical registration.
        """
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}
        self._parent = parent

        if parent is not None:
            parent._add_child(self)

        self._children: Dict[str, 'Registry'] = {}

    @property
    def name(self) -> str:
        """Get registry name."""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type[Any]]:
        """Get registered modules."""
        return self._module_dict

    def _add_child(self, child: 'Registry') -> None:
        """Add a child registry."""
        self._children[child.name] = child

    def get(self, key: str) -> Optional[Type[Any]]:
        """Get a registered module by key.

        Args:
            key: Module name to retrieve.

        Returns:
            Registered class or None if not found.
        """
        if key in self._module_dict:
            return self._module_dict[key]

        if self._parent is not None:
            return self._parent.get(key)

        return None

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module: Optional[Type[T]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """Register a module.

        A module can be registered in two ways:
            1. As a decorator: @REGISTRY.register_module()
            2. As a function: REGISTRY.register_module(module=SomeClass)

        Args:
            name: Module name (defaults to class name).
            force: Whether to override existing registration.
            module: Module class to register (if not using as decorator).

        Returns:
            Decorator function or the module itself.
        """
        if module is not None:
            self._register_module(module, name, force)
            return module  # type: ignore

        def decorator(cls: Type[T]) -> Type[T]:
            self._register_module(cls, name, force)
            return cls

        return decorator

    def _register_module(
        self,
        module: Type[Any],
        name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Internal method to register a module."""
        module_name = name or module.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(
                f'\'{module_name}\' is already registered in {self._name} registry'
            )

        self._module_dict[module_name] = module

    def build(self, cfg: Dict[str, Any], **default_args: Any) -> Any:
        """Build a module from config dict.

        Args:
            cfg: Config dict with 'type' key specifying module name.
            **default_args: Default arguments to pass to module.

        Returns:
            Instantiated module.

        Raises:
            KeyError: If module type is not registered.
            TypeError: If cfg is not a dict or missing 'type'.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

        if 'type' not in cfg:
            raise KeyError('cfg must contain \'type\' key')

        cfg = cfg.copy()
        module_type = cfg.pop('type')

        module_cls = self.get(module_type)
        if module_cls is None:
            raise KeyError(
                f'\'{module_type}\' is not registered in {self._name} registry. '
                f'Available modules: {list(self._module_dict.keys())}'
            )

        for key, value in default_args.items():
            cfg.setdefault(key, value)

        return module_cls(**cfg)

    def __contains__(self, key: str) -> bool:
        """Check if a module is registered."""
        return self.get(key) is not None

    def __repr__(self) -> str:
        """Return string representation."""
        return f'Registry(name={self._name}, modules={list(self._module_dict.keys())})'


# Global registries for different component types
BACKBONES = Registry('backbones')
NECKS = Registry('necks')
ENCODERS = Registry('encoders')
HEADS = Registry('heads')
LOSSES = Registry('losses')
DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
METRICS = Registry('metrics')

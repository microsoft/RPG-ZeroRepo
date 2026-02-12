#!/usr/bin/env python3
"""
Complete test suite for _extract_class_from_code and _extract_function_from_code methods.
Tests various edge cases and complex scenarios to ensure robustness.
"""

from repo_encoder.rebuild import Rebuild
import sys

# Test code with comprehensive cases
comprehensive_test_code = '''
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from pathlib import Path
import json
import time

# ============================================================================
# FUNCTION TEST CASES
# ============================================================================

# Case 1: Simple function with basic signature
def simple_add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Case 2: Single-line function
def single_line_func(x: int) -> int: return x * 2

# Case 3: Function with no type hints
def no_hints_func(a, b):
    return a + b

# Case 4: Function with no docstring
def no_docstring(x, y):
    result = x * y
    return result

# Case 5: Function with complex decorators
@lru_cache(maxsize=128)
@wraps(some_other_function)
def decorated_function(data: str) -> str:
    """Function with multiple decorators."""
    return data.upper()

# Case 6: Async function
async def simple_async(timeout: float = 5.0) -> Optional[str]:
    """Simple async function."""
    await asyncio.sleep(0.1)
    return "done"

# Case 7: Complex async function with decorators
@asyncio.coroutine
@wraps(async_helper)
@custom_decorator(
    param="complex_value",
    timeout=30,
    retry_count=3
)
async def complex_async_function(
    param1: str,
    param2: Optional[int] = None,
    *args: Any,
    **kwargs: Dict[str, Any]
) -> List[str]:
    """
    Complex async function with multi-line signature and decorators.
    
    This function demonstrates:
    - Multiple decorators with complex parameters
    - Multi-line function signature
    - Type hints with generics
    - Variable arguments
    - Comprehensive docstring
    
    Args:
        param1: Primary string parameter
        param2: Optional integer parameter with default
        *args: Variable positional arguments
        **kwargs: Variable keyword arguments
        
    Returns:
        List of processed strings
        
    Raises:
        ValueError: If param1 is invalid
        RuntimeError: If processing fails
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    results = []
    async for item in some_async_generator():
        processed = await process_item(item, param2)
        results.append(processed)
    
    return results

# Case 8: Function with default mutable arguments
def func_with_defaults(
    name: str,
    items: List[str] = None,
    metadata: Dict[str, Any] = None,
    **options: Any
) -> Dict[str, Any]:
    """Function with complex default arguments."""
    if items is None:
        items = []
    if metadata is None:
        metadata = {}
    
    return {
        "name": name,
        "items": items,
        "metadata": metadata,
        "options": options
    }

# Case 9: Lambda-style single expression function
def lambda_style(x: int) -> int: return x ** 2 if x > 0 else 0

# Case 10: Function with multi-line string literal in signature
def func_with_string_literal(
    query: str = """
    SELECT * FROM users 
    WHERE name = 'test' 
    AND active = true
    """,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Function with multi-line string in default argument."""
    # Complex query execution
    results = execute_query(query, limit)
    return results

# ============================================================================
# CLASS TEST CASES
# ============================================================================

# Case 1: Simple class with basic methods
class SimpleCalculator:
    """A simple calculator class."""
    
    def __init__(self, initial_value: int = 0):
        """Initialize calculator."""
        self.value = initial_value
        self.history = []
    
    def add(self, x: int) -> int:
        """Add value."""
        self.value += x
        self.history.append(f"add {x}")
        return self.value
    
    def reset(self) -> None:
        """Reset calculator."""
        self.value = 0
        self.history.clear()

# Case 2: Dataclass
@dataclass
class UserProfile:
    """User profile data class."""
    
    # Class variables
    DEFAULT_ROLE: str = "user"
    MAX_LOGIN_ATTEMPTS: int = 3
    
    # Instance variables with defaults
    username: str = ""
    email: str = ""
    age: int = 0
    roles: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.roles:
            self.roles = [self.DEFAULT_ROLE]
        
        if not self.email and self.username:
            self.email = f"{self.username}@example.com"

# Case 3: Abstract base class with complex inheritance
@abstractmethod
class DatabaseConnection(ABC):
    """Abstract database connection interface."""
    
    # Class constants
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    def __init__(self, connection_string: str, timeout: int = None):
        """Initialize connection."""
        self.connection_string = connection_string
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._connection = None
        self.is_connected = False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DatabaseConnection":
        """Create connection from configuration."""
        conn_str = config["connection_string"]
        timeout = config.get("timeout", cls.DEFAULT_TIMEOUT)
        return cls(conn_str, timeout)
    
    @staticmethod
    def validate_connection_string(conn_str: str) -> bool:
        """Validate connection string format."""
        if not conn_str or not isinstance(conn_str, str):
            return False
        return "://" in conn_str and len(conn_str.split("://")) == 2
    
    @property
    def status(self) -> str:
        """Get connection status."""
        return "connected" if self.is_connected else "disconnected"
    
    @status.setter  
    def status(self, value: str) -> None:
        """Set connection status."""
        if value not in ("connected", "disconnected"):
            raise ValueError(f"Invalid status: {value}")
        self.is_connected = (value == "connected")
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection (abstract)."""
        pass
    
    @abstractmethod
    async def async_connect(self) -> None:
        """Establish connection asynchronously (abstract)."""
        pass
    
    def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self.is_connected = False

# Case 4: Complex class with nested classes and advanced features
class AdvancedDataProcessor:
    """
    Advanced data processing class with comprehensive features.
    
    This class demonstrates various Python features:
    - Class variables and instance variables
    - Multiple decorator types
    - Property getters and setters
    - Static and class methods
    - Async methods
    - Private methods
    - Type hints with generics
    """
    
    # Class variables
    VERSION = "2.1.0"
    SUPPORTED_FORMATS = {"json", "xml", "csv", "yaml"}
    DEFAULT_CONFIG = {
        "batch_size": 1000,
        "timeout": 300,
        "retries": 3
    }
    
    def __init__(
        self, 
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize processor.
        
        Args:
            name: Processor name
            config: Optional configuration override
        """
        self.name = name
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._processed_count = 0
        self._errors: List[str] = []
        self._start_time: Optional[float] = None
    
    @classmethod
    def create_default(cls, name: str) -> "AdvancedDataProcessor":
        """Create processor with default settings."""
        return cls(name, cls.DEFAULT_CONFIG.copy())
    
    @classmethod  
    def from_file(cls, config_path: Path) -> "AdvancedDataProcessor":
        """Load processor from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config["name"], config.get("settings", {}))
    
    @staticmethod
    def validate_format(format_name: str) -> bool:
        """Validate if format is supported."""
        return format_name.lower() in AdvancedDataProcessor.SUPPORTED_FORMATS
    
    @property
    def processed_count(self) -> int:
        """Get number of processed items."""
        return self._processed_count
    
    @property
    def error_count(self) -> int:
        """Get number of errors encountered."""
        return len(self._errors)
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate."""
        if self._processed_count == 0:
            return 0.0
        return (self._processed_count - len(self._errors)) / self._processed_count
    
    @property
    def is_running(self) -> bool:
        """Check if processor is currently running."""
        return self._start_time is not None
    
    @is_running.setter
    def is_running(self, value: bool) -> None:
        """Set running state."""
        if value and self._start_time is None:
            self._start_time = time.time()
        elif not value and self._start_time is not None:
            self._start_time = None
    
    async def process_async(
        self, 
        data: List[Dict[str, Any]],
        callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> List[Any]:
        """
        Process data asynchronously.
        
        Args:
            data: List of data items to process
            callback: Optional callback for each processed item
            
        Returns:
            List of processed results
        """
        results = []
        self.is_running = True
        
        try:
            for batch in self._create_batches(data):
                batch_results = await self._process_batch_async(batch, callback)
                results.extend(batch_results)
                self._processed_count += len(batch)
        except Exception as e:
            self._errors.append(str(e))
            logging.error(f"Processing failed: {e}")
            raise
        finally:
            self.is_running = False
            
        return results
    
    def process_sync(self, data: List[Dict[str, Any]]) -> List[Any]:
        """Process data synchronously."""
        results = []
        self.is_running = True
        
        try:
            for item in data:
                try:
                    result = self._process_item(item)
                    results.append(result)
                    self._processed_count += 1
                except Exception as e:
                    self._errors.append(f"Item {self._processed_count}: {e}")
                    logging.warning(f"Failed to process item: {e}")
        finally:
            self.is_running = False
            
        return results
    
    def _create_batches(self, data: List[Any]) -> List[List[Any]]:
        """Create batches from data."""
        batch_size = self.config["batch_size"]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    async def _process_batch_async(
        self, 
        batch: List[Dict[str, Any]],
        callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process a single batch asynchronously."""
        tasks = []
        for item in batch:
            task = asyncio.create_task(self._process_item_async(item))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if callback:
            for result in results:
                if not isinstance(result, Exception):
                    await callback(result)
                    
        return [r for r in results if not isinstance(r, Exception)]
    
    async def _process_item_async(self, item: Dict[str, Any]) -> Any:
        """Process single item asynchronously."""
        # Simulate async processing
        await asyncio.sleep(0.01)
        return self._process_item(item)
    
    def _process_item(self, item: Dict[str, Any]) -> Any:
        """Process single item (private method)."""
        # Complex processing logic would go here
        processed = {
            "original": item,
            "processed_at": time.time(),
            "processor": self.name
        }
        return processed

# Case 5: Empty class
class EmptyClass:
    pass

# Case 6: Class with only class variables
class ConstantsClass:
    """Class containing only constants."""
    API_VERSION = "v1"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    SUPPORTED_METHODS = ["GET", "POST", "PUT", "DELETE"]

# Case 7: Class with only docstring
class DocstringOnlyClass:
    """
    This class has only a docstring.
    
    It serves as a placeholder or interface definition.
    """

# Case 8: Nested class scenario
class OuterClass:
    """Outer class with nested components."""
    
    OUTER_CONSTANT = "outer_value"
    
    class InnerClass:
        """Inner class definition."""
        INNER_CONSTANT = "inner_value"
        
        def inner_method(self) -> str:
            """Inner method."""
            return self.INNER_CONSTANT
    
    def __init__(self, name: str):
        """Initialize outer class."""
        self.name = name
        self.inner = self.InnerClass()
    
    def get_inner_value(self) -> str:
        """Get value from inner class."""
        return self.inner.inner_method()

# ============================================================================
# EDGE CASES
# ============================================================================

# Edge case 1: Function with triple-quoted string parameters  
def func_with_triple_quotes(
    sql_query: str = """
        SELECT users.name, users.email, profiles.bio
        FROM users 
        JOIN profiles ON users.id = profiles.user_id
        WHERE users.active = true
        ORDER BY users.created_at DESC
        LIMIT 100
    """,
    format_str: str = "Name: {name}, Email: {email}, Bio: {bio}"
) -> str:
    """Function with triple-quoted string parameters."""
    return f"Query: {sql_query}; Format: {format_str}"

# Edge case 2: Function with complex nested parentheses
def complex_signature_func(
    data: Dict[str, List[Tuple[int, str]]],
    processor: Callable[[Tuple[int, str]], Dict[str, Any]] = lambda x: {"id": x[0], "name": x[1]},
    filters: Optional[List[Callable[[Dict[str, Any]], bool]]] = None
) -> List[Dict[str, Any]]:
    """Function with complex nested type signatures."""
    results = []
    for key, items in data.items():
        for item in items:
            processed = processor(item)
            processed["category"] = key
            
            if filters:
                if all(f(processed) for f in filters):
                    results.append(processed)
            else:
                results.append(processed)
    
    return results

# Edge case 3: Class with unusual method combinations
@dataclass
class EdgeCaseClass:
    """Class with various edge cases."""
    
    # Class variable with complex type
    COMPLEX_DEFAULT: Dict[str, List[Tuple[str, int]]] = {
        "group_a": [("item1", 1), ("item2", 2)],
        "group_b": [("item3", 3), ("item4", 4)]
    }
    
    value: str = "default"
    
    @property
    def dynamic_value(self) -> str:
        """Dynamic property."""
        return f"dynamic_{self.value}"
    
    @dynamic_value.setter
    def dynamic_value(self, val: str) -> None:
        """Set dynamic value."""
        self.value = val.replace("dynamic_", "")
    
    @dynamic_value.deleter  
    def dynamic_value(self) -> None:
        """Delete dynamic value."""
        self.value = ""
    
    @classmethod
    async def async_class_method(cls, config: Dict[str, Any]) -> "EdgeCaseClass":
        """Async class method."""
        await asyncio.sleep(0.1)  # Simulate async work
        return cls(config.get("value", "async_default"))

'''

def run_comprehensive_tests():
    """Run comprehensive tests on all cases."""
    rebuild = Rebuild(".", "test", ".", None, None)
    
    # Define all test cases
    function_test_cases = [
        "simple_add",
        "single_line_func", 
        "no_hints_func",
        "no_docstring",
        "decorated_function",
        "simple_async",
        "complex_async_function",
        "func_with_defaults",
        "lambda_style",
        "func_with_string_literal",
        "func_with_triple_quotes",
        "complex_signature_func"
    ]
    
    class_test_cases = [
        "SimpleCalculator",
        "UserProfile", 
        "DatabaseConnection",
        "AdvancedDataProcessor",
        "EmptyClass",
        "ConstantsClass",
        "DocstringOnlyClass",
        "OuterClass",
        "EdgeCaseClass"
    ]
    
    print("🧪 COMPREHENSIVE EXTRACTION TEST SUITE")
    print("=" * 80)
    print(f"Testing {len(function_test_cases)} function cases and {len(class_test_cases)} class cases\n")
    
    # Test Functions
    print("📋 FUNCTION EXTRACTION TESTS")
    print("-" * 50)
    
    function_results = {}
    for i, func_name in enumerate(function_test_cases, 1):
        print(f"\n[{i:2d}/{len(function_test_cases)}] Testing function: {func_name}")
        try:
            result = rebuild._extract_function_from_code(comprehensive_test_code, func_name)
            if result:
                function_results[func_name] = "✅ SUCCESS"
                # Show first few lines for verification
                lines = result.split('\n')
                preview = lines[0] if lines else "Empty result"
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                print(f"    Result: {preview}")
            else:
                function_results[func_name] = "❌ FAILED - No result"
                print(f"    Result: ❌ FAILED - No result returned")
        except Exception as e:
            function_results[func_name] = f"❌ ERROR - {str(e)}"
            print(f"    Result: ❌ ERROR - {e}")
    
    # Test Classes  
    print(f"\n\n📋 CLASS EXTRACTION TESTS")
    print("-" * 50)
    
    class_results = {}
    for i, class_name in enumerate(class_test_cases, 1):
        print(f"\n[{i:2d}/{len(class_test_cases)}] Testing class: {class_name}")
        try:
            result = rebuild._extract_class_from_code(comprehensive_test_code, class_name)
            if result:
                class_results[class_name] = "✅ SUCCESS"
                # Show first few lines for verification
                lines = result.split('\n')
                preview = lines[0] if lines else "Empty result"
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                print(f"    Result: {preview}")
            else:
                class_results[class_name] = "❌ FAILED - No result"
                print(f"    Result: ❌ FAILED - No result returned")
        except Exception as e:
            class_results[class_name] = f"❌ ERROR - {str(e)}"
            print(f"    Result: ❌ ERROR - {e}")
    
    # Summary
    print(f"\n\n📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    func_success = sum(1 for v in function_results.values() if v == "✅ SUCCESS")
    class_success = sum(1 for v in class_results.values() if v == "✅ SUCCESS")
    total_tests = len(function_test_cases) + len(class_test_cases)
    total_success = func_success + class_success
    
    print(f"Function Tests: {func_success}/{len(function_test_cases)} passed")
    print(f"Class Tests:    {class_success}/{len(class_test_cases)} passed")
    print(f"Overall:        {total_success}/{total_tests} passed ({total_success/total_tests*100:.1f}%)")
    
    # Show failures
    failures = []
    for name, result in {**function_results, **class_results}.items():
        if not result.startswith("✅"):
            failures.append((name, result))
    
    if failures:
        print(f"\n❌ FAILED CASES:")
        for name, error in failures:
            print(f"   {name}: {error}")
    else:
        print(f"\n🎉 ALL TESTS PASSED!")
    
    return function_results, class_results

def show_detailed_example(item_type: str, item_name: str):
    """Show detailed extraction result for a specific item."""
    rebuild = Rebuild(".", "test", ".", None, None)
    
    print(f"\n🔍 DETAILED EXAMPLE: {item_type} '{item_name}'")
    print("=" * 80)
    
    try:
        if item_type.lower() == "function":
            result = rebuild._extract_function_from_code(comprehensive_test_code, item_name)
        else:
            result = rebuild._extract_class_from_code(comprehensive_test_code, item_name)
        
        if result:
            print("✅ EXTRACTION SUCCESSFUL")
            print("\n📄 EXTRACTED CODE:")
            print("-" * 40)
            print(result)
            print("-" * 40)
            print(f"📏 Lines: {len(result.split(chr(10)))}")
            print(f"📏 Characters: {len(result)}")
        else:
            print("❌ EXTRACTION FAILED - No result returned")
            
    except Exception as e:
        print(f"❌ EXTRACTION ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "detailed":
        # Show detailed example
        if len(sys.argv) >= 4:
            show_detailed_example(sys.argv[2], sys.argv[3])
        else:
            print("Usage: python test_extraction_complete.py detailed <function|class> <name>")
            print("Example: python test_extraction_complete.py detailed class AdvancedDataProcessor")
    else:
        # Run all tests
        run_comprehensive_tests()
        
        print("\n💡 TIP: For detailed output of any specific case, run:")
        print("   python test_extraction_complete.py detailed function <function_name>")
        print("   python test_extraction_complete.py detailed class <class_name>")
        print("   Example: python test_extraction_complete.py detailed class AdvancedDataProcessor")
"""Test runner for laraq test cases."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def load_tests(test_file: Path) -> list[dict[str, Any]]:
    """Load tests from an XML file.

    Args:
        test_file: Path to XML test file

    Returns:
        List of test case dictionaries
    """
    tree = ET.parse(test_file)
    root = tree.getroot()

    tests = []
    for test_elem in root.findall('test'):
        test_case = {
            'name': test_elem.get('name'),
            'description': test_elem.findtext('description', ''),
            'query': test_elem.findtext('query'),
            'expected_code': test_elem.findtext('expected_code').strip(),
            'expected_output': _parse_output(test_elem.find('expected_output')),
            'tolerance': float(test_elem.findtext('tolerance', '1e-6'))
        }
        tests.append(test_case)

    return tests


def _parse_output(elem: ET.Element) -> Any:
    """Parse expected output from XML element with explicit types.

    Args:
        elem: XML element containing expected output

    Returns:
        Parsed Python object
    """
    output_type = elem.get('type')

    if output_type == 'list':
        return [_parse_item(item) for item in elem.findall('item')]
    elif output_type == 'tuple':
        return tuple(_parse_item(item) for item in elem.findall('item'))
    elif output_type == 'dict':
        result = {}
        for item in elem.findall('item'):
            key = item.get('key')
            result[key] = _parse_item(item)
        return result
    else:
        # Single value
        return _parse_item(elem)


def _parse_item(elem: ET.Element) -> Any:
    """Parse a single item with explicit type.

    Args:
        elem: XML element with type attribute

    Returns:
        Parsed value
    """
    item_type = elem.get('type')
    value = elem.text.strip() if elem.text else ''

    if item_type == 'int':
        return int(value)
    elif item_type == 'float':
        return float(value)
    elif item_type == 'str':
        return value
    elif item_type == 'bool':
        return value.lower() == 'true'
    elif item_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown type: {item_type}")


def execute_code(code: str) -> Any:
    """Execute Python code and return the result.

    Args:
        code: Python code string containing a function f()

    Returns:
        The return value from executing f()
    """
    namespace = {}
    exec(code, namespace)

    if 'f' in namespace:
        return namespace['f']()
    else:
        raise ValueError("Code does not define function f()")


def compare_outputs(actual: Any, expected: Any, tolerance: float) -> tuple[bool, str]:
    """Compare actual and expected outputs with tolerance.

    Args:
        actual: Actual output from execution
        expected: Expected output (parsed from XML)
        tolerance: Numeric tolerance for comparisons

    Returns:
        Tuple of (match: bool, message: str)
    """
    return _compare_values(actual, expected, tolerance)


def _compare_values(actual: Any, expected: Any, tolerance: float, path: str = "root") -> tuple[bool, str]:
    """Recursively compare values with tolerance.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Numeric tolerance
        path: Current path in nested structure (for error messages)

    Returns:
        Tuple of (match: bool, message: str)
    """
    # Handle None
    if actual is None and expected is None:
        return (True, "Both None")
    if actual is None or expected is None:
        return (False, f"At {path}: {actual} != {expected}")

    # Handle sequences (lists and tuples)
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return (False, f"At {path}: Expected sequence, got {type(actual).__name__}")
        if len(actual) != len(expected):
            return (False, f"At {path}: Length mismatch: {len(actual)} != {len(expected)}")

        for i, (a, e) in enumerate(zip(actual, expected)):
            match, msg = _compare_values(a, e, tolerance, f"{path}[{i}]")
            if not match:
                return (False, msg)
        return (True, "Values match")

    # Handle dicts
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return (False, f"At {path}: Expected dict, got {type(actual).__name__}")
        if set(actual.keys()) != set(expected.keys()):
            return (False, f"At {path}: Key mismatch")

        for key in expected.keys():
            match, msg = _compare_values(actual[key], expected[key], tolerance, f"{path}['{key}']")
            if not match:
                return (False, msg)
        return (True, "Values match")

    # Handle numbers
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        diff = abs(actual - expected)
        if diff <= tolerance:
            return (True, "Values match")
        else:
            return (False, f"At {path}: {actual} != {expected} (diff: {diff:.2e} > tol: {tolerance})")

    # Handle other types
    if actual == expected:
        return (True, "Values match")
    else:
        return (False, f"At {path}: {actual} != {expected}")

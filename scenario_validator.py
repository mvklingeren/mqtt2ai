"""Scenario Validator module for test assertions.

Validates simulation results against assertions defined in the scenario file.
Provides test reporting for both interactive and CI/CD use.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from event_bus import event_bus, EventType


@dataclass
class AssertionResult:
    """Result of checking a single assertion."""
    id: str
    passed: bool
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


class ScenarioValidator:
    """Validates collected events against scenario assertions.
    
    Assertion types supported:
    - trigger_count: Count of trigger events for a topic/field
    - tool_called: AI tool was called with optional argument matching
    - rule_executed: Direct rule execution count or specific rule
    - file_state: Check JSON file contents (e.g., learned_rules.json)
    """

    def __init__(self, assertions: Dict[str, dict]):
        """Initialize with assertions dictionary from scenario.
        
        Args:
            assertions: Dict mapping assertion IDs to assertion configs
        """
        self.assertions = assertions

    def validate(self) -> List[AssertionResult]:
        """Validate all assertions against collected events.
        
        Returns:
            List of AssertionResult for each assertion
        """
        results = []
        for assertion_id, assertion in self.assertions.items():
            result = self._check_assertion(assertion_id, assertion)
            results.append(result)
        return results

    def _check_assertion(self, assertion_id: str, assertion: dict) -> AssertionResult:
        """Check a single assertion."""
        assertion_type = assertion.get("type", "unknown")
        
        handlers = {
            "trigger_count": self._check_trigger_count,
            "tool_called": self._check_tool_called,
            "rule_executed": self._check_rule_executed,
            "rule_not_matched": self._check_rule_not_matched,
            "file_state": self._check_file_state,
        }
        
        handler = handlers.get(assertion_type)
        if handler:
            return handler(assertion_id, assertion)
        
        return AssertionResult(
            id=assertion_id,
            passed=False,
            message=f"Unknown assertion type: {assertion_type}"
        )

    def _check_trigger_count(self, assertion_id: str, assertion: dict) -> AssertionResult:
        """Check count of TRIGGER_FIRED events matching criteria."""
        events = event_bus.get_events(EventType.TRIGGER_FIRED)
        
        topic = assertion.get("topic")
        field = assertion.get("field")
        min_count = assertion.get("min", 1)
        max_count = assertion.get("max")
        
        matching = events
        if topic:
            matching = [e for e in matching if e.data.get("topic") == topic]
        if field:
            matching = [e for e in matching if e.data.get("field") == field]
        
        count = len(matching)
        
        # Check min/max constraints
        passed = count >= min_count
        if max_count is not None and count > max_count:
            passed = False
        
        expected = f">= {min_count}"
        if max_count is not None:
            expected = f"{min_count}-{max_count}"
        
        filter_desc = ""
        if topic:
            filter_desc += f" topic={topic}"
        if field:
            filter_desc += f" field={field}"
        
        return AssertionResult(
            id=assertion_id,
            passed=passed,
            message=f"Trigger count:{filter_desc} = {count}",
            expected=expected,
            actual=str(count)
        )

    def _check_tool_called(self, assertion_id: str, assertion: dict) -> AssertionResult:
        """Check if AI called a specific tool."""
        events = event_bus.get_events(EventType.AI_TOOL_CALLED)
        
        tool_name = assertion.get("tool")
        args_contain = assertion.get("args_contain", {})
        count_min = assertion.get("count_min", 1)
        
        # Filter by tool name
        matching = [e for e in events if e.data.get("tool") == tool_name]
        
        # Filter by args if specified
        if args_contain:
            filtered = []
            for e in matching:
                args = e.data.get("arguments", {})
                if self._dict_contains(args, args_contain):
                    filtered.append(e)
            matching = filtered
        
        count = len(matching)
        passed = count >= count_min
        
        args_desc = ""
        if args_contain:
            args_desc = f" with {args_contain}"
        
        return AssertionResult(
            id=assertion_id,
            passed=passed,
            message=f"Tool {tool_name}{args_desc} called {count} time(s)",
            expected=f">= {count_min}",
            actual=str(count)
        )

    def _check_rule_executed(self, assertion_id: str, assertion: dict) -> AssertionResult:
        """Check if rules were executed by RuleEngine."""
        events = event_bus.get_events(EventType.RULE_EXECUTED)
        
        rule_id = assertion.get("rule_id")
        rule_id_contains = assertion.get("rule_id_contains")
        count_min = assertion.get("count_min", 1)
        
        matching = events
        if rule_id:
            matching = [e for e in matching if e.data.get("rule_id") == rule_id]
        elif rule_id_contains:
            matching = [e for e in matching 
                       if rule_id_contains in str(e.data.get("rule_id", ""))]
        
        count = len(matching)
        passed = count >= count_min
        
        filter_desc = ""
        if rule_id:
            filter_desc = f" rule_id={rule_id}"
        elif rule_id_contains:
            filter_desc = f" rule_id contains '{rule_id_contains}'"
        
        return AssertionResult(
            id=assertion_id,
            passed=passed,
            message=f"Rule executed{filter_desc}: {count} time(s)",
            expected=f">= {count_min}",
            actual=str(count)
        )

    def _check_rule_not_matched(self, assertion_id: str, assertion: dict) -> AssertionResult:
        """Check count of triggers that went to AI (no rule matched)."""
        events = event_bus.get_events(EventType.RULE_NOT_MATCHED)
        
        count_min = assertion.get("count_min", 1)
        count_max = assertion.get("count_max")
        
        count = len(events)
        passed = count >= count_min
        if count_max is not None and count > count_max:
            passed = False
        
        expected = f">= {count_min}"
        if count_max is not None:
            expected = f"{count_min}-{count_max}"
        
        return AssertionResult(
            id=assertion_id,
            passed=passed,
            message=f"Triggers sent to AI (no rule matched): {count}",
            expected=expected,
            actual=str(count)
        )

    def _check_file_state(self, assertion_id: str, assertion: dict) -> AssertionResult:
        """Check contents of a JSON file."""
        file_path = assertion.get("file")
        rules_min = assertion.get("rules_min")
        patterns_min = assertion.get("patterns_min")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return AssertionResult(
                id=assertion_id,
                passed=False,
                message=f"File not found: {file_path}",
                expected="File exists",
                actual="File not found"
            )
        except json.JSONDecodeError as e:
            return AssertionResult(
                id=assertion_id,
                passed=False,
                message=f"Invalid JSON in {file_path}: {e}",
                expected="Valid JSON",
                actual="Parse error"
            )
        
        # Check rules count
        if rules_min is not None:
            rules = data.get("rules", [])
            count = len(rules)
            passed = count >= rules_min
            return AssertionResult(
                id=assertion_id,
                passed=passed,
                message=f"{file_path} has {count} rules",
                expected=f">= {rules_min}",
                actual=str(count)
            )
        
        # Check patterns count
        if patterns_min is not None:
            patterns = data.get("patterns", [])
            count = len(patterns)
            passed = count >= patterns_min
            return AssertionResult(
                id=assertion_id,
                passed=passed,
                message=f"{file_path} has {count} patterns",
                expected=f">= {patterns_min}",
                actual=str(count)
            )
        
        return AssertionResult(
            id=assertion_id,
            passed=True,
            message=f"File {file_path} exists and is valid JSON"
        )

    def _dict_contains(self, haystack: dict, needle: dict) -> bool:
        """Check if haystack dict contains all key-value pairs from needle."""
        for key, value in needle.items():
            if key not in haystack:
                return False
            if haystack[key] != value:
                # Try string comparison for type flexibility
                if str(haystack[key]).lower() != str(value).lower():
                    return False
        return True


def print_test_report(results: List[AssertionResult], scenario_name: str = "") -> bool:
    """Print colored test report to terminal.
    
    Args:
        results: List of assertion results
        scenario_name: Optional scenario name for header
        
    Returns:
        True if all assertions passed, False otherwise
    """
    green, red, yellow, reset = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
    bold, dim = "\033[1m", "\033[2m"
    
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    all_passed = passed_count == total
    
    print(f"\n{'='*60}")
    print(f"{bold}TEST RESULTS{reset}")
    if scenario_name:
        print(f"{dim}Scenario: {scenario_name}{reset}")
    print('='*60)
    
    for r in results:
        if r.passed:
            status = f"{green}PASS{reset}"
        else:
            status = f"{red}FAIL{reset}"
        
        print(f"[{status}] {bold}{r.id}{reset}: {r.message}")
        
        if not r.passed and r.expected:
            print(f"       {dim}Expected: {r.expected}{reset}")
            print(f"       {dim}Actual:   {r.actual}{reset}")
    
    print()
    if all_passed:
        print(f"{green}{bold}{passed_count}/{total} assertions passed{reset}")
    else:
        print(f"{red}{bold}{passed_count}/{total} assertions passed{reset}")
    print()
    
    return all_passed


def write_json_report(results: List[AssertionResult], 
                      output_path: str,
                      scenario_name: str = "",
                      duration_seconds: float = 0.0) -> None:
    """Write test results to a JSON file for CI/CD integration.
    
    Args:
        results: List of assertion results
        output_path: Path to write JSON report
        scenario_name: Optional scenario name
        duration_seconds: Total test duration
    """
    from datetime import datetime
    
    report = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration_seconds,
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "total": len(results),
        "all_passed": all(r.passed for r in results),
        "assertions": [
            {
                "id": r.id,
                "status": "pass" if r.passed else "fail",
                "message": r.message,
                "expected": r.expected,
                "actual": r.actual
            }
            for r in results
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logging.info("Test report written to %s", output_path)


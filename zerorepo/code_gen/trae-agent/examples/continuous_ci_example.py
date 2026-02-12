#!/usr/bin/env python3
"""
Example of using ContinuousTraeAgentRunner for CI/CD-like workflows.
This simulates processing pull requests or code changes continuously.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import ContinuousTraeAgentRunner

class ContinuousCI:
    """Simulated CI system using Trae Agent."""
    
    def __init__(self, config_file="trae_config.yaml", working_dir="./ci_workspace"):
        self.config_file = config_file
        self.working_dir = working_dir
        self.runner = None
        
    def start(self):
        """Start the CI runner."""
        self.runner = ContinuousTraeAgentRunner(
            provider="openai",
            config_file=self.config_file,
            working_dir=self.working_dir,
            docker_image="ubuntu:22.04",
            docker_container_name="trae-ci-runner",
        )
        self.runner.start()
        print("CI Runner started!")
        
    def stop(self):
        """Stop the CI runner."""
        if self.runner:
            self.runner.stop()
            
    def process_pull_request(self, pr_data: Dict) -> Dict:
        """
        Process a pull request through various checks.
        
        Args:
            pr_data: Dictionary containing PR information
            
        Returns:
            Dictionary with check results
        """
        pr_id = pr_data["id"]
        pr_title = pr_data["title"]
        pr_description = pr_data["description"]
        files_changed = pr_data.get("files_changed", [])
        
        print(f"\nProcessing PR #{pr_id}: {pr_title}")
        print(f"Files changed: {len(files_changed)}")
        
        # Define CI checks as Trae Agent tasks
        checks = []
        
        # 1. Code style check
        if any(f.endswith('.py') for f in files_changed):
            task_id = f"pr_{pr_id}_style_check"
            self.runner.submit_task(
                f"Check Python code style issues in the following files: {files_changed}. "
                f"Look for PEP 8 violations, naming conventions, and formatting issues.",
                task_id=task_id
            )
            checks.append(("Style Check", task_id))
            
        # 2. Security scan
        task_id = f"pr_{pr_id}_security_scan"
        self.runner.submit_task(
            f"Scan for potential security vulnerabilities in PR '{pr_title}'. "
            f"Check for hardcoded credentials, SQL injection risks, or unsafe operations.",
            task_id=task_id
        )
        checks.append(("Security Scan", task_id))
        
        # 3. Test coverage analysis
        if any(f.endswith('.py') for f in files_changed):
            task_id = f"pr_{pr_id}_test_coverage"
            self.runner.submit_task(
                f"Analyze if the changes in {files_changed} have adequate test coverage. "
                f"Suggest missing test cases if needed.",
                task_id=task_id
            )
            checks.append(("Test Coverage", task_id))
            
        # 4. Documentation check
        task_id = f"pr_{pr_id}_docs_check"
        self.runner.submit_task(
            f"Check if the PR '{pr_title}' includes necessary documentation updates. "
            f"Description: {pr_description}",
            task_id=task_id
        )
        checks.append(("Documentation", task_id))
        
        # 5. Performance impact analysis
        task_id = f"pr_{pr_id}_performance"
        self.runner.submit_task(
            f"Analyze potential performance impact of changes in PR '{pr_title}'.",
            task_id=task_id
        )
        checks.append(("Performance", task_id))
        
        # Wait for all checks to complete
        results = {}
        for check_name, task_id in checks:
            result = self.runner.get_result(timeout=120)  # 2 minute timeout per check
            if result:
                results[check_name] = {
                    "status": "passed" if result["status"] == "success" else "failed",
                    "details": result.get("output", "")[:500]  # First 500 chars
                }
            else:
                results[check_name] = {
                    "status": "timeout",
                    "details": "Check timed out"
                }
                
        return {
            "pr_id": pr_id,
            "overall_status": "passed" if all(r["status"] == "passed" for r in results.values()) else "failed",
            "checks": results
        }
        
    def generate_pr_comment(self, pr_data: Dict, check_results: Dict) -> str:
        """Generate a comment for the PR based on check results."""
        task_id = f"pr_{pr_data['id']}_comment"
        
        checks_summary = json.dumps(check_results["checks"], indent=2)
        
        self.runner.submit_task(
            f"Generate a helpful, constructive PR review comment based on these check results:\n"
            f"PR: {pr_data['title']}\n"
            f"Overall Status: {check_results['overall_status']}\n"
            f"Checks:\n{checks_summary}\n"
            f"Make the comment friendly and actionable.",
            task_id=task_id
        )
        
        result = self.runner.get_result(timeout=60)
        if result and result["status"] == "success":
            return result["output"]
        return "Unable to generate PR comment."


def simulate_pr_queue():
    """Simulate processing a queue of pull requests."""
    
    # Simulated PRs
    pull_requests = [
        {
            "id": 123,
            "title": "Add user authentication module",
            "description": "Implements JWT-based authentication for the API",
            "files_changed": ["auth.py", "models/user.py", "tests/test_auth.py"]
        },
        {
            "id": 124,
            "title": "Fix memory leak in data processor",
            "description": "Fixes issue where large datasets cause memory overflow",
            "files_changed": ["processors/data_processor.py", "utils/memory.py"]
        },
        {
            "id": 125,
            "title": "Update README with new features",
            "description": "Documents the new API endpoints",
            "files_changed": ["README.md", "docs/api.md"]
        },
    ]
    
    ci = ContinuousCI()
    ci.start()
    
    try:
        print("Starting CI Pipeline")
        print("=" * 50)
        
        for pr in pull_requests:
            # Process PR
            check_results = ci.process_pull_request(pr)
            
            # Generate comment
            comment = ci.generate_pr_comment(pr, check_results)
            
            # Display results
            print(f"\n{'='*50}")
            print(f"PR #{pr['id']}: {pr['title']}")
            print(f"Overall Status: {check_results['overall_status']}")
            print("\nCheck Results:")
            for check_name, result in check_results["checks"].items():
                status_emoji = "✓" if result["status"] == "passed" else "✗"
                print(f"  {status_emoji} {check_name}: {result['status']}")
            
            print(f"\nGenerated Comment Preview:")
            print("-" * 30)
            print(comment[:300] + "..." if len(comment) > 300 else comment)
            print("-" * 30)
            
            # Simulate delay between PRs
            time.sleep(2)
            
    finally:
        ci.stop()


def run_continuous_monitoring():
    """Example of continuous monitoring and alerting."""
    
    with ContinuousTraeAgentRunner(
        provider="openai",
        config_file="trae_config.yaml",
        working_dir="./monitoring_workspace",
    ) as runner:
        
        print("Continuous Monitoring System")
        print("=" * 40)
        print("Monitoring for issues... Press Ctrl+C to stop.")
        
        # Simulate monitoring different aspects
        monitoring_tasks = [
            {
                "name": "Error Log Analysis",
                "interval": 30,  # seconds
                "task": "Analyze application error logs for critical issues or patterns"
            },
            {
                "name": "Performance Metrics",
                "interval": 60,
                "task": "Check system performance metrics and identify bottlenecks"
            },
            {
                "name": "Security Alerts",
                "interval": 45,
                "task": "Scan for security vulnerabilities or suspicious activities"
            },
        ]
        
        last_run = {task["name"]: 0 for task in monitoring_tasks}
        
        try:
            while True:
                current_time = time.time()
                
                # Submit monitoring tasks based on their intervals
                for task in monitoring_tasks:
                    if current_time - last_run[task["name"]] >= task["interval"]:
                        task_id = f"{task['name'].replace(' ', '_')}_{int(current_time)}"
                        runner.submit_task(task["task"], task_id=task_id)
                        last_run[task["name"]] = current_time
                        print(f"\n[{time.strftime('%H:%M:%S')}] Running: {task['name']}")
                
                # Check for results
                while True:
                    result = runner.get_result(timeout=0.1)
                    if result:
                        if result["status"] == "success":
                            # Check if the result contains any alerts
                            output_lower = result["output"].lower()
                            if any(word in output_lower for word in ["critical", "error", "warning", "alert"]):
                                print(f"\n🚨 ALERT from {result['id']}:")
                                print(result["output"][:200])
                        else:
                            print(f"\n❌ Task {result['id']} failed")
                    else:
                        break
                        
                # Wait before next check
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring system...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CI/CD examples using ContinuousTraeAgentRunner")
    parser.add_argument(
        "example",
        choices=["pr-queue", "monitoring"],
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "pr-queue":
        simulate_pr_queue()
    elif args.example == "monitoring":
        run_continuous_monitoring()
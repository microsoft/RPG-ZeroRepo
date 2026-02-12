# Trae Agent Examples

This directory contains examples of how to use Trae Agent, particularly the new Continuous Runner functionality.

## Continuous Trae Agent Runner

The Continuous Runner allows you to run Trae Agent persistently in a Docker container, processing multiple tasks in sequence without needing to recreate the environment each time.

### Key Features

- **Persistent Environment**: Container stays alive between tasks
- **Task Queue**: Submit multiple tasks and process them asynchronously
- **Result Collection**: Get results as tasks complete
- **Resource Efficient**: Reuses the same container for multiple tasks
- **Isolated Execution**: Each task runs in a clean workspace within the container

### Basic Usage

#### 1. Command Line Interface

```bash
# Run in continuous mode with interactive prompt
python main.py continuous

# Run with demo tasks
python main.py continuous --demo-tasks

# Customize configuration
python main.py continuous --provider openai --model gpt-4 --config-file my_config.yaml
```

#### 2. Programmatic Usage

```python
from main import ContinuousTraeAgentRunner

# Use as context manager
with ContinuousTraeAgentRunner(
    provider="openai",
    config_file="trae_config.yaml",
    working_dir="./workspace"
) as runner:
    # Submit tasks
    task_id = runner.submit_task("Create a Python function to calculate fibonacci numbers")
    
    # Get results
    result = runner.get_result(timeout=300)
    print(f"Task completed: {result['status']}")
```

### Examples

#### 1. Basic Examples (`continuous_runner_example.py`)

Run different types of continuous workflows:

```bash
# Batch processing example
python examples/continuous_runner_example.py batch

# Interactive session
python examples/continuous_runner_example.py interactive

# Code review pipeline
python examples/continuous_runner_example.py review
```

#### 2. CI/CD Examples (`continuous_ci_example.py`)

Simulate continuous integration workflows:

```bash
# Process a queue of pull requests
python examples/continuous_ci_example.py pr-queue

# Continuous monitoring system
python examples/continuous_ci_example.py monitoring
```

#### 3. External Container Examples (`external_container_example.py`)

Use existing Docker containers directly:

```bash
# Create and use external container
python examples/external_container_example.py external

# Reuse persistent container
python examples/external_container_example.py reuse

# Single task with external container
python examples/external_container_example.py single
```

### Use Cases

1. **Batch Processing**: Process multiple related tasks in sequence
2. **Code Review Automation**: Automated code analysis and review
3. **Continuous Integration**: CI/CD pipeline automation
4. **Content Generation**: Generate multiple pieces of content
5. **Testing Automation**: Run comprehensive test suites
6. **Monitoring & Alerting**: Continuous system monitoring
7. **External Container Integration**: Use pre-configured or specialized containers
8. **Container Reuse**: Maintain persistent development environments

### Configuration

The continuous runner uses the same configuration as regular Trae Agent:

- **Provider**: LLM provider (openai, anthropic, etc.)
- **Model**: Specific model to use
- **Config File**: Trae Agent configuration file
- **Working Directory**: Host directory mounted in container
- **Docker Image**: Base Docker image (default: ubuntu:22.04)

### Environment Setup

1. Ensure Docker is installed and running
2. Set up your LLM provider API keys
3. Create a `trae_config.yaml` configuration file

Example `trae_config.yaml`:
```yaml
provider: openai
model: gpt-4
api_key: your-api-key-here
```

### Error Handling

The continuous runner includes robust error handling:

- Task failures don't stop the runner
- Container issues are automatically handled
- Timeout protection for long-running tasks
- Graceful shutdown and cleanup

### Performance Tips

1. **Container Reuse**: The same container is reused for efficiency
2. **Parallel Tasks**: Submit multiple tasks for better throughput
3. **Resource Management**: Monitor container resource usage
4. **Working Directory**: Use a persistent working directory for file sharing

### Troubleshooting

Common issues and solutions:

1. **Docker Permission Issues**: Ensure your user is in the docker group
2. **Container Startup Failures**: Check Docker daemon is running
3. **Task Timeouts**: Increase timeout values for complex tasks
4. **Memory Issues**: Monitor container memory usage

### Contributing

To add new examples:

1. Create a new Python file in this directory
2. Follow the existing example patterns
3. Include proper documentation and error handling
4. Test with different configurations
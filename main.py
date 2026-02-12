#!/usr/bin/env python3
"""
ZeroRepo Main Entry Script

This script serves as the main entry point for the ZeroRepo system, responsible for:
1. Parsing command line arguments
2. Initializing system configuration
3. Creating and setting up global checkpoint manager
4. Starting the complete end-to-end code generation pipeline
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from zerorepo.zerorepo import ZeroRepo
from zerorepo.rpg_gen.base.llm_client import LLMConfig
from zerorepo.config.checkpoint_config import create_default_manager
from zerorepo.utils.logs import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ZeroRepo: LLM-based Automated Code Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Basic usage with default configuration
  python main.py --config configs/zerorepo_config.yaml --checkpoint ./checkpoints --repo ./target_repo

  # Specify all parameters
  python main.py \\
    --config configs/zerorepo_config.yaml \\
    --checkpoint ./checkpoints \\
    --repo ./target_repo \\
    --log-level DEBUG \\
    --force-rebuild

  # Set API key via environment variable
  export OPENAI_API_KEY="your-api-key"
  python main.py --config configs/zerorepo_config.yaml --checkpoint ./checkpoints --repo ./target_repo

Supported LLM Models:
  - OpenAI: gpt-4, gpt-3.5-turbo
  - Anthropic: claude-3-sonnet, claude-3-haiku
  - Google: gemini-pro
  - Local models: ollama/*
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="ZeroRepo configuration file path (YAML format)"
    )
    
    parser.add_argument(
        "--checkpoint", "-k", 
        type=str,
        required=True,
        help="Checkpoint directory path for saving intermediate states"
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        required=True,
        help="Target repository path where generated code will be saved"
    )
    
    # Optional arguments
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path, outputs to console if not specified"
    )
    
    parser.add_argument(
        "--force-rebuild", "-f",
        action="store_true",
        help="Force rebuild Docker containers and reset state"
    )
    
    parser.add_argument(
        "--resume", "-res",
        action="store_true", 
        help="Resume execution from checkpoint (skip completed phases)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Dry run mode - only validate configuration without execution"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "design", "implementation"],
        default="all",
        help="Specify execution phase: 'all'=full pipeline, 'design'=design only, 'implementation'=implementation only"
    )

    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        help="Path to LLM configuration file (YAML/JSON). "
             "Overrides the 'llm' section in the main config file."
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="ZeroRepo 1.0.0"
    )
    
    return parser.parse_args()


def validate_args(args) -> bool:
    """Validate command line arguments"""
    errors = []
    
    # Check configuration file
    config_path = Path(args.config)
    if not config_path.exists():
        errors.append(f"Configuration file does not exist: {config_path}")
    elif not config_path.suffix.lower() in ['.yaml', '.yml']:
        errors.append(f"Configuration file must be YAML format: {config_path}")
    
    # Check checkpoint directory
    checkpoint_path = Path(args.checkpoint)
    try:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create checkpoint directory {checkpoint_path}: {e}")
    
    # Check target repository path
    repo_path = Path(args.repo)
    try:
        repo_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create target repository directory {repo_path}: {e}")
    
    # Output error messages
    if errors:
        print("❌ Argument validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def setup_logging(args) -> logging.Logger:
    """Initialize the logging system for ZeroRepo main entrypoint."""
    # 规范化 log level
    level_name = str(args.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    # 拿一个带名字的 logger
    logger = logging.getLogger("zerorepo-main")

    logger = setup_logger(
        logger=logger,
        file_path=args.log_file,
        level=level,
    )

    sep = "=" * 60
    logger.info(sep)
    logger.info("🚀 ZeroRepo Automated Code Generation System Starting")
    logger.info(sep)
    logger.info("📋 Configuration file: %s", args.config)
    logger.info("💾 Checkpoint directory: %s", args.checkpoint)
    logger.info("📁 Target repository: %s", args.repo)
    logger.info("🔧 Execution phase: %s", args.phase)
    logger.info("📊 Log level: %s", level_name)

    if getattr(args, "force_rebuild", False):
        logger.info("🔄 Force rebuild mode enabled")
    if getattr(args, "resume", False):
        logger.info("⏯️ Resume mode enabled")
    if getattr(args, "dry_run", False):
        logger.info("🧪 Dry run mode enabled")

    return logger

def run_zerorepo(args, logger: logging.Logger) -> bool:
    """Run ZeroRepo main pipeline"""
    try:
        # Convert paths
        config_path = Path(args.config)
        checkpoint_dir = Path(args.checkpoint)
        repo_path = Path(args.repo)
        
        logger.info("🔧 Initializing ZeroRepo system...")
        
        # Create global checkpoint manager
        logger.info(f"📋 Setting up global checkpoint manager: {checkpoint_dir}")
        create_default_manager(checkpoint_dir)
        
        # Dry run mode
        if args.dry_run:
            logger.info("🧪 Dry run mode - validating configuration only")
            zero_repo = ZeroRepo(config_path, checkpoint_dir, repo_path)
            logger.info("✅ Configuration validation passed, system initialized successfully")
            return True
        
        # Create ZeroRepo instance
        zero_repo = ZeroRepo(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            repo_path=repo_path,
            logger=logger  # Pass the logger with file configuration
        )

        # Override LLM config from --llm-config file if provided
        if args.llm_config:
            llm_cfg = LLMConfig.from_source(args.llm_config)
            zero_repo.config["llm"] = llm_cfg.to_dict()
            logger.info(f"LLM config overridden from: {args.llm_config}")
        
        # Execute based on specified phase
        if args.phase == "design":
            logger.info("🎨 Executing design phase...")
            result = zero_repo.run_design_phase(resume=args.resume)
            logger.info("✅ Design phase completed")
            
        elif args.phase == "implementation":
            logger.info("⚡ Executing implementation phase...")
            result = zero_repo.run_implementation_phase(
                resume=args.resume,
                force_rebuild=args.force_rebuild
            )
            logger.info("✅ Implementation phase completed")
            
        else:  # phase == "all"
            logger.info("🔄 Executing full pipeline...")
            result = zero_repo.run()

        if result:
            logger.info("=" * 60)
            logger.info("🎉 ZeroRepo execution completed successfully!")
            logger.info(f"📁 Generated code saved in: {repo_path}")
            logger.info(f"💾 Intermediate files saved in: {checkpoint_dir}")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("❌ ZeroRepo execution failed, check logs for details")
            logger.error("=" * 60)

        return bool(result)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ User interrupted execution")
        return False
        
    except Exception as e:
        logger.error(f"❌ ZeroRepo execution failed: {e}")
        logger.debug("Detailed error information:", exc_info=True)
        return False


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(args)
    
    try:
        # Run ZeroRepo
        success = run_zerorepo(args, logger)
        
        if success:
            logger.info("🚀 ZeroRepo execution successful")
            sys.exit(0)
        else:
            logger.error("❌ ZeroRepo execution failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Main program exception: {e}")
        logger.debug("Detailed exception information:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
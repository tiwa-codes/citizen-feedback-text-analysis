"""
Command-line interface for citizen feedback analysis pipeline.

Provides convenience commands for running the entire pipeline.
"""
import argparse
import sys
from pathlib import Path
import subprocess


def generate_data(args):
    """Generate synthetic feedback data."""
    print("="*60)
    print("Generating Synthetic Feedback Data")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.data.generate_synthetic_feedback',
        '--n', str(args.n),
        '--months', str(args.months),
        '--seed', str(args.seed)
    ]
    
    if args.output:
        cmd.extend(['--output', args.output])
    
    subprocess.run(cmd, check=True)


def clean_data(args):
    """Clean raw feedback data."""
    print("="*60)
    print("Cleaning Feedback Data")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.text.cleaning',
        '--input', args.input,
        '--output', args.output
    ]
    
    subprocess.run(cmd, check=True)


def run_topics(args):
    """Run topic modeling."""
    print("="*60)
    print("Running Topic Modeling")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.text.topic_modeling',
        '--input', args.input,
        '--output', args.output,
        '--method', args.method
    ]
    
    if args.n_topics:
        cmd.extend(['--n-topics', str(args.n_topics)])
    
    subprocess.run(cmd, check=True)


def compute_sentiment(args):
    """Compute sentiment scores."""
    print("="*60)
    print("Computing Sentiment")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.text.sentiment'
    ]
    
    subprocess.run(cmd, check=True)


def extract_features(args):
    """Extract text features."""
    print("="*60)
    print("Extracting Text Features")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.text.features'
    ]
    
    subprocess.run(cmd, check=True)


def generate_plots(args):
    """Generate visualization plots."""
    print("="*60)
    print("Generating Visualizations")
    print("="*60)
    
    cmd = [
        sys.executable, '-m', 'src.viz.plots'
    ]
    
    subprocess.run(cmd, check=True)


def run_dashboard(args):
    """Launch Streamlit dashboard."""
    print("="*60)
    print("Launching Dashboard")
    print("="*60)
    
    cmd = [
        'streamlit', 'run', 'dashboards/app.py'
    ]
    
    if args.port:
        cmd.extend(['--server.port', str(args.port)])
    
    subprocess.run(cmd, check=True)


def run_full_pipeline(args):
    """Run the complete analysis pipeline."""
    print("="*60)
    print("Running Full Analysis Pipeline")
    print("="*60)
    print()
    
    # Step 1: Generate data
    if not args.skip_generate:
        print("Step 1/5: Generating synthetic data...")
        generate_data(argparse.Namespace(
            n=50000,
            months=24,
            seed=42,
            output=None
        ))
        print()
    
    # Step 2: Clean data
    print("Step 2/5: Cleaning data...")
    clean_data(argparse.Namespace(
        input='data/raw/citizen_feedback.csv',
        output='data/processed/citizen_feedback_clean.parquet'
    ))
    print()
    
    # Step 3: Compute sentiment
    print("Step 3/5: Computing sentiment...")
    compute_sentiment(argparse.Namespace())
    print()
    
    # Step 4: Run topic modeling
    print("Step 4/5: Running topic modeling...")
    run_topics(argparse.Namespace(
        input='data/processed/citizen_feedback_clean.parquet',
        output='data/processed/topic_assignments.csv',
        method='lda',
        n_topics=None
    ))
    print()
    
    # Step 5: Generate visualizations
    print("Step 5/5: Generating visualizations...")
    generate_plots(argparse.Namespace())
    print()
    
    print("="*60)
    print("✓ Full pipeline complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("  - Explore notebook: jupyter lab notebooks/01_citizen_feedback_eda.ipynb")
    print("  - Launch dashboard: python -m src.cli run-dashboard")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Citizen Feedback Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic data
  python -m src.cli generate-data --n 50000 --months 24
  
  # Clean data
  python -m src.cli clean
  
  # Run topic modeling
  python -m src.cli run-topics --method lda --n-topics 10
  
  # Compute sentiment
  python -m src.cli compute-sentiment
  
  # Run full pipeline
  python -m src.cli run-pipeline
  
  # Launch dashboard
  python -m src.cli run-dashboard --port 8501
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data command
    gen_parser = subparsers.add_parser(
        'generate-data',
        help='Generate synthetic feedback data'
    )
    gen_parser.add_argument('--n', type=int, default=50000, help='Number of records')
    gen_parser.add_argument('--months', type=int, default=24, help='Number of months')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    gen_parser.add_argument('--output', type=str, help='Output CSV path')
    gen_parser.set_defaults(func=generate_data)
    
    # Clean data command
    clean_parser = subparsers.add_parser('clean', help='Clean feedback data')
    clean_parser.add_argument(
        '--input',
        default='data/raw/citizen_feedback.csv',
        help='Input CSV file'
    )
    clean_parser.add_argument(
        '--output',
        default='data/processed/citizen_feedback_clean.parquet',
        help='Output parquet file'
    )
    clean_parser.set_defaults(func=clean_data)
    
    # Topic modeling command
    topic_parser = subparsers.add_parser('run-topics', help='Run topic modeling')
    topic_parser.add_argument(
        '--input',
        default='data/processed/citizen_feedback_clean.parquet',
        help='Input parquet file'
    )
    topic_parser.add_argument(
        '--output',
        default='data/processed/topic_assignments.csv',
        help='Output CSV file'
    )
    topic_parser.add_argument(
        '--method',
        choices=['lda', 'nmf'],
        default='lda',
        help='Topic modeling method'
    )
    topic_parser.add_argument('--n-topics', type=int, help='Number of topics')
    topic_parser.set_defaults(func=run_topics)
    
    # Sentiment command
    sentiment_parser = subparsers.add_parser(
        'compute-sentiment',
        help='Compute sentiment scores'
    )
    sentiment_parser.set_defaults(func=compute_sentiment)
    
    # Features command
    features_parser = subparsers.add_parser(
        'extract-features',
        help='Extract text features'
    )
    features_parser.set_defaults(func=extract_features)
    
    # Plots command
    plots_parser = subparsers.add_parser(
        'generate-plots',
        help='Generate visualization plots'
    )
    plots_parser.set_defaults(func=generate_plots)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        'run-dashboard',
        help='Launch Streamlit dashboard'
    )
    dashboard_parser.add_argument('--port', type=int, help='Server port')
    dashboard_parser.set_defaults(func=run_dashboard)
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        'run-pipeline',
        help='Run complete analysis pipeline'
    )
    pipeline_parser.add_argument(
        '--skip-generate',
        action='store_true',
        help='Skip data generation step'
    )
    pipeline_parser.set_defaults(func=run_full_pipeline)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

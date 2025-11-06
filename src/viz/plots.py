"""
Visualization functions for citizen feedback analysis.

Provides plotting functions for topics, sentiment trends, and keywords.
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_topic_trends(
    df: pd.DataFrame,
    topic_col: str = 'dominant_topic',
    date_col: str = 'created_at',
    time_freq: str = 'M',
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot topic proportions over time.
    
    Args:
        df: DataFrame with topic assignments and dates
        topic_col: Name of topic column
        date_col: Name of date column
        time_freq: Time frequency ('D', 'W', 'M', 'Q')
        output_path: Path to save figure
        show: Whether to display plot
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by time period and topic
    df['period'] = df[date_col].dt.to_period(time_freq)
    
    # Calculate topic proportions per period
    topic_counts = df.groupby(['period', topic_col]).size().unstack(fill_value=0)
    topic_props = topic_counts.div(topic_counts.sum(axis=1), axis=0)
    
    # Convert period to timestamp for plotting
    topic_props.index = topic_props.index.to_timestamp()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Area chart
    topic_props.plot.area(ax=ax, alpha=0.7, stacked=True)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Topic Proportion')
    ax.set_title(f'Topic Trends Over Time ({time_freq} frequency)')
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved topic trends to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_sentiment_trends(
    df: pd.DataFrame,
    sentiment_col: str = 'sentiment_label',
    date_col: str = 'created_at',
    by: str = 'national',
    time_freq: str = 'M',
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot sentiment trends over time.
    
    Args:
        df: DataFrame with sentiment and dates
        sentiment_col: Name of sentiment column
        date_col: Name of date column
        by: 'national' or column name for grouping (e.g., 'state', 'channel')
        time_freq: Time frequency ('D', 'W', 'M', 'Q')
        output_path: Path to save figure
        show: Whether to display plot
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['period'] = df[date_col].dt.to_period(time_freq)
    
    if by == 'national':
        # National level
        sentiment_counts = df.groupby(['period', sentiment_col]).size().unstack(fill_value=0)
        sentiment_props = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)
        
        sentiment_props.index = sentiment_props.index.to_timestamp()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_props.columns:
                ax.plot(
                    sentiment_props.index,
                    sentiment_props[sentiment],
                    marker='o',
                    label=sentiment.capitalize(),
                    color=colors.get(sentiment, 'blue'),
                    linewidth=2
                )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Proportion')
        ax.set_title(f'Sentiment Trends Over Time ({time_freq} frequency)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        # By group (e.g., state, channel)
        # Show top groups
        top_groups = df[by].value_counts().head(5).index.tolist()
        df_filtered = df[df[by].isin(top_groups)]
        
        fig, axes = plt.subplots(len(top_groups), 1, figsize=(14, 4 * len(top_groups)))
        if len(top_groups) == 1:
            axes = [axes]
        
        for idx, group in enumerate(top_groups):
            group_df = df_filtered[df_filtered[by] == group]
            sentiment_counts = group_df.groupby(['period', sentiment_col]).size().unstack(fill_value=0)
            sentiment_props = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)
            
            sentiment_props.index = sentiment_props.index.to_timestamp()
            
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment in sentiment_props.columns:
                    axes[idx].plot(
                        sentiment_props.index,
                        sentiment_props[sentiment],
                        marker='o',
                        label=sentiment.capitalize(),
                        color=colors.get(sentiment, 'blue'),
                        linewidth=2
                    )
            
            axes[idx].set_title(f'{by.capitalize()}: {group}')
            axes[idx].set_ylabel('Proportion')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment trends to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_top_keywords(
    keywords: List[Tuple[str, float]],
    top_n: int = 20,
    output_path: Optional[Path] = None,
    show: bool = False,
    title: str = "Top Keywords by TF-IDF Score"
):
    """
    Plot top keywords as horizontal bar chart.
    
    Args:
        keywords: List of (keyword, score) tuples
        top_n: Number of top keywords to plot
        output_path: Path to save figure
        show: Whether to display plot
        title: Plot title
    """
    keywords = keywords[:top_n]
    words, scores = zip(*keywords)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    y_pos = np.arange(len(words))
    ax.barh(y_pos, scores, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('TF-IDF Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved keywords plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_topic_sentiment_heatmap(
    df: pd.DataFrame,
    topic_col: str = 'dominant_topic',
    sentiment_col: str = 'sentiment_label',
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot heatmap of topic vs sentiment distribution.
    
    Args:
        df: DataFrame with topics and sentiment
        topic_col: Name of topic column
        sentiment_col: Name of sentiment column
        output_path: Path to save figure
        show: Whether to display plot
    """
    # Create crosstab
    crosstab = pd.crosstab(df[topic_col], df[sentiment_col], normalize='index')
    
    # Reorder columns for consistent display
    col_order = ['positive', 'neutral', 'negative']
    crosstab = crosstab[[col for col in col_order if col in crosstab.columns]]
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(crosstab) * 0.5)))
    
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.33,
        cbar_kws={'label': 'Proportion'},
        ax=ax
    )
    
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Topic')
    ax.set_title('Topic × Sentiment Distribution')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_channel_distribution(
    df: pd.DataFrame,
    channel_col: str = 'channel',
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot distribution of feedback by channel.
    
    Args:
        df: DataFrame with channel column
        channel_col: Name of channel column
        output_path: Path to save figure
        show: Whether to display plot
    """
    channel_counts = df[channel_col].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    channel_counts.plot.bar(ax=ax1, color='steelblue')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Count')
    ax1.set_title('Feedback Count by Channel')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    ax2.pie(
        channel_counts.values,
        labels=channel_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('Feedback Distribution by Channel')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved channel distribution to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_state_distribution(
    df: pd.DataFrame,
    state_col: str = 'state',
    top_n: int = 15,
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot distribution of feedback by state.
    
    Args:
        df: DataFrame with state column
        state_col: Name of state column
        top_n: Number of top states to show
        output_path: Path to save figure
        show: Whether to display plot
    """
    state_counts = df[state_col].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
    
    y_pos = np.arange(len(state_counts))
    ax.barh(y_pos, state_counts.values, color='coral')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(state_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel('Feedback Count')
    ax.set_title(f'Top {top_n} States by Feedback Volume')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved state distribution to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_word_length_distribution(
    df: pd.DataFrame,
    word_count_col: str = 'word_count',
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot distribution of message word counts.
    
    Args:
        df: DataFrame with word count column
        word_count_col: Name of word count column
        output_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df[word_count_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Message Lengths')
    ax.axvline(df[word_count_col].median(), color='red', linestyle='--', 
               label=f'Median: {df[word_count_col].median():.0f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved word length distribution to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def main():
    """Example usage of visualization functions."""
    from pathlib import Path
    
    # Load data with topic assignments
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'processed' / 'topic_assignments.csv'
    
    if not data_path.exists():
        print(f"Error: Topic assignments not found at {data_path}")
        print("Please run topic modeling first: python -m src.text.topic_modeling")
        return
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} records")
    
    # Create output directory
    output_dir = project_root / 'reports' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_channel_distribution(df, output_path=output_dir / 'channel_distribution.png')
    plot_state_distribution(df, output_path=output_dir / 'state_distribution.png')
    plot_word_length_distribution(df, output_path=output_dir / 'word_length_distribution.png')
    
    if 'sentiment_label' in df.columns:
        plot_sentiment_trends(
            df,
            by='national',
            output_path=output_dir / 'sentiment_trends_national.png'
        )
        plot_sentiment_trends(
            df,
            by='channel',
            output_path=output_dir / 'sentiment_trends_by_channel.png'
        )
    
    if 'dominant_topic' in df.columns:
        plot_topic_trends(df, output_path=output_dir / 'topic_trends.png')
        
        if 'sentiment_label' in df.columns:
            plot_topic_sentiment_heatmap(
                df,
                output_path=output_dir / 'topic_sentiment_heatmap.png'
            )
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

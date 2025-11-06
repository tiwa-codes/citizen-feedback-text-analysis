"""
Interactive Streamlit dashboard for citizen feedback analysis.

Provides interactive exploration of feedback data with filters and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


# Page configuration
st.set_page_config(
    page_title="Citizen Feedback Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    """Load processed feedback data with caching."""
    project_root = Path(__file__).parent.parent
    
    # Try to load topic assignments (includes all previous processing)
    topic_path = project_root / 'data' / 'processed' / 'topic_assignments.csv'
    clean_path = project_root / 'data' / 'processed' / 'citizen_feedback_clean.parquet'
    
    if topic_path.exists():
        df = pd.read_csv(topic_path)
    elif clean_path.exists():
        df = pd.read_parquet(clean_path)
    else:
        st.error("No processed data found. Please run the pipeline first.")
        st.stop()
    
    # Ensure date column is datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    return df


def apply_filters(df, states, date_range, channels, departments, topics, sentiments):
    """Apply user-selected filters to DataFrame."""
    filtered_df = df.copy()
    
    if states and 'All' not in states:
        filtered_df = filtered_df[filtered_df['state'].isin(states)]
    
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['created_at'] >= pd.Timestamp(start_date)) &
            (filtered_df['created_at'] <= pd.Timestamp(end_date))
        ]
    
    if channels and 'All' not in channels:
        filtered_df = filtered_df[filtered_df['channel'].isin(channels)]
    
    if departments and 'All' not in departments:
        filtered_df = filtered_df[filtered_df['assigned_dept'].isin(departments)]
    
    if 'dominant_topic' in filtered_df.columns and topics and 'All' not in topics:
        filtered_df = filtered_df[filtered_df['dominant_topic'].isin(topics)]
    
    if 'sentiment_label' in filtered_df.columns and sentiments and 'All' not in sentiments:
        filtered_df = filtered_df[filtered_df['sentiment_label'].isin(sentiments)]
    
    return filtered_df


def display_kpis(df):
    """Display key performance indicators."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Messages", f"{len(df):,}")
    
    with col2:
        if 'sentiment_label' in df.columns:
            negative_pct = (df['sentiment_label'] == 'negative').sum() / len(df) * 100
            st.metric("% Negative", f"{negative_pct:.1f}%")
        else:
            st.metric("% Negative", "N/A")
    
    with col3:
        if 'resolved' in df.columns:
            unresolved_pct = (df['resolved'] == 'False').sum() / len(df) * 100
            st.metric("% Unresolved", f"{unresolved_pct:.1f}%")
        else:
            st.metric("% Unresolved", "N/A")
    
    with col4:
        if 'response_time_days' in df.columns:
            resolved_df = df[df['response_time_days'].notna() & (df['response_time_days'] != '')]
            if len(resolved_df) > 0:
                avg_response = pd.to_numeric(resolved_df['response_time_days'], errors='coerce').mean()
                st.metric("Avg Response (days)", f"{avg_response:.1f}")
            else:
                st.metric("Avg Response (days)", "N/A")
        else:
            st.metric("Avg Response (days)", "N/A")
    
    with col5:
        if 'word_count' in df.columns:
            avg_words = df['word_count'].mean()
            st.metric("Avg Words", f"{avg_words:.0f}")
        else:
            st.metric("Avg Words", "N/A")


def plot_topic_trend_interactive(df):
    """Create interactive topic trend plot."""
    if 'dominant_topic' not in df.columns:
        st.info("Topic assignments not available. Run topic modeling first.")
        return
    
    # Aggregate by month and topic
    df_copy = df.copy()
    df_copy['month'] = df_copy['created_at'].dt.to_period('M').dt.to_timestamp()
    
    topic_monthly = df_copy.groupby(['month', 'dominant_topic']).size().reset_index(name='count')
    
    fig = px.area(
        topic_monthly,
        x='month',
        y='count',
        color='dominant_topic',
        title='Topic Trends Over Time',
        labels={'month': 'Month', 'count': 'Message Count', 'dominant_topic': 'Topic'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_sentiment_over_time(df):
    """Create interactive sentiment trend plot."""
    if 'sentiment_label' not in df.columns:
        st.info("Sentiment not available. Run sentiment analysis first.")
        return
    
    # Aggregate by month and sentiment
    df_copy = df.copy()
    df_copy['month'] = df_copy['created_at'].dt.to_period('M').dt.to_timestamp()
    
    sentiment_monthly = df_copy.groupby(['month', 'sentiment_label']).size().reset_index(name='count')
    
    # Calculate proportions
    total_monthly = sentiment_monthly.groupby('month')['count'].transform('sum')
    sentiment_monthly['proportion'] = sentiment_monthly['count'] / total_monthly
    
    fig = px.line(
        sentiment_monthly,
        x='month',
        y='proportion',
        color='sentiment_label',
        title='Sentiment Trends Over Time',
        labels={'month': 'Month', 'proportion': 'Proportion', 'sentiment_label': 'Sentiment'},
        color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'},
        markers=True
    )
    
    fig.update_layout(hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)


def plot_channel_breakdown(df):
    """Create channel distribution plot."""
    channel_counts = df['channel'].value_counts().reset_index()
    channel_counts.columns = ['channel', 'count']
    
    fig = px.bar(
        channel_counts,
        x='channel',
        y='count',
        title='Feedback by Channel',
        labels={'channel': 'Channel', 'count': 'Count'},
        color='count',
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_top_keywords_interactive(df):
    """Display top keywords from text."""
    if 'cleaned_text' not in df.columns:
        return
    
    from collections import Counter
    import re
    
    # Extract words
    all_text = ' '.join(df['cleaned_text'].astype(str).tolist())
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Basic stopwords
    stopwords = {
        'the', 'and', 'for', 'are', 'this', 'that', 'with', 'from',
        'have', 'has', 'was', 'were', 'been', 'they', 'their'
    }
    
    words = [w for w in words if len(w) > 3 and w not in stopwords]
    
    # Get top 20
    word_counts = Counter(words).most_common(20)
    
    if word_counts:
        keywords_df = pd.DataFrame(word_counts, columns=['keyword', 'count'])
        
        fig = px.bar(
            keywords_df,
            x='count',
            y='keyword',
            orientation='h',
            title='Top 20 Keywords',
            labels={'count': 'Frequency', 'keyword': 'Keyword'},
            color='count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)


def show_representative_messages(df):
    """Show table of representative messages."""
    st.subheader("Representative Messages")
    
    if 'dominant_topic' in df.columns:
        selected_topic = st.selectbox(
            "Select Topic",
            options=['All'] + sorted(df['dominant_topic'].unique().tolist())
        )
        
        if selected_topic != 'All':
            display_df = df[df['dominant_topic'] == selected_topic]
        else:
            display_df = df
    else:
        display_df = df
    
    # Sample messages
    sample_size = min(10, len(display_df))
    sample_df = display_df.sample(sample_size) if len(display_df) > 0 else display_df
    
    # Display
    for idx, row in sample_df.iterrows():
        with st.expander(f"ID: {row.get('feedback_id', idx)} | {row.get('state', 'N/A')} | {row.get('channel', 'N/A')}"):
            st.write(f"**Date:** {row['created_at'].strftime('%Y-%m-%d')}")
            st.write(f"**Facility:** {row.get('facility_or_service', 'N/A')}")
            if 'sentiment_label' in row:
                st.write(f"**Sentiment:** {row['sentiment_label']}")
            if 'dominant_topic' in row:
                st.write(f"**Topic:** {row['dominant_topic']}")
            st.write(f"**Message:** {row.get('cleaned_text', row.get('raw_text', 'N/A'))}")


def main():
    """Main dashboard application."""
    st.title("📊 Citizen Feedback Analysis Dashboard")
    st.markdown("Interactive exploration of citizen feedback about public services in Nigeria")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # State filter
    states = ['All'] + sorted(df['state'].unique().tolist())
    selected_states = st.sidebar.multiselect(
        "States",
        options=states,
        default=['All']
    )
    
    # Date range filter
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Channel filter
    channels = ['All'] + sorted(df['channel'].unique().tolist())
    selected_channels = st.sidebar.multiselect(
        "Channels",
        options=channels,
        default=['All']
    )
    
    # Department filter
    departments = ['All'] + sorted(df['assigned_dept'].unique().tolist())
    selected_departments = st.sidebar.multiselect(
        "Departments",
        options=departments,
        default=['All']
    )
    
    # Topic filter (if available)
    if 'dominant_topic' in df.columns:
        topics = ['All'] + sorted(df['dominant_topic'].unique().tolist())
        selected_topics = st.sidebar.multiselect(
            "Topics",
            options=topics,
            default=['All']
        )
    else:
        selected_topics = ['All']
    
    # Sentiment filter (if available)
    if 'sentiment_label' in df.columns:
        sentiments = ['All'] + sorted(df['sentiment_label'].unique().tolist())
        selected_sentiments = st.sidebar.multiselect(
            "Sentiments",
            options=sentiments,
            default=['All']
        )
    else:
        selected_sentiments = ['All']
    
    # Apply filters
    if len(date_range) == 2:
        filtered_df = apply_filters(
            df,
            selected_states,
            date_range,
            selected_channels,
            selected_departments,
            selected_topics,
            selected_sentiments
        )
    else:
        filtered_df = df
    
    # Display KPIs
    st.header("Key Metrics")
    display_kpis(filtered_df)
    
    st.markdown("---")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Distributions", "💬 Messages", "📥 Export"])
    
    with tab1:
        st.header("Trends Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_topic_trend_interactive(filtered_df)
        
        with col2:
            plot_sentiment_over_time(filtered_df)
    
    with tab2:
        st.header("Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_channel_breakdown(filtered_df)
        
        with col2:
            plot_top_keywords_interactive(filtered_df)
        
        # State distribution
        st.subheader("Top States by Volume")
        state_counts = filtered_df['state'].value_counts().head(15).reset_index()
        state_counts.columns = ['state', 'count']
        
        fig = px.bar(
            state_counts,
            x='count',
            y='state',
            orientation='h',
            title='Feedback Volume by State',
            labels={'count': 'Count', 'state': 'State'},
            color='count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        show_representative_messages(filtered_df)
    
    with tab4:
        st.header("Export Filtered Data")
        
        st.write(f"Current selection: **{len(filtered_df):,}** records")
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"citizen_feedback_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Show sample
        st.subheader("Preview (first 100 rows)")
        st.dataframe(filtered_df.head(100))
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Records:** {len(df):,}")
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,}")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard provides interactive exploration of citizen feedback data. "
        "Use filters to drill down into specific segments."
    )


if __name__ == '__main__':
    main()

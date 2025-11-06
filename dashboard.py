import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SentimentDashboard:
    """
    Interactive dashboard for visualizing sentiment analysis results.
    
    This dashboard provides business stakeholders with intuitive visualizations
    of customer sentiment trends, enabling data-driven decisions about product
    improvements and customer service strategies.
    """
    
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        """Configure the main page layout and styling."""
        st.title("🎯 Product Review Sentiment Analysis Dashboard")
        st.markdown("""
        Welcome to your comprehensive sentiment analysis dashboard! This tool helps you understand
        customer opinions at scale, identify trends, and make data-driven decisions to improve
        your products and customer experience.
        """)
        
        # Add some custom CSS for better styling
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .neutral { color: #6c757d; }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_sample_data(self, n_reviews=1000):
        """
        Generate realistic sample data for demonstration purposes.
        
        In a real implementation, this would be replaced with actual
        review data from your e-commerce platform or database.
        """
        np.random.seed(42)
        
        # Generate date range for the last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Product categories and their typical sentiment distributions
        products = {
            'Electronics': {'positive': 0.6, 'neutral': 0.25, 'negative': 0.15},
            'Clothing': {'positive': 0.7, 'neutral': 0.2, 'negative': 0.1},
            'Home & Garden': {'positive': 0.65, 'neutral': 0.22, 'negative': 0.13},
            'Sports': {'positive': 0.68, 'neutral': 0.18, 'negative': 0.14},
            'Books': {'positive': 0.75, 'neutral': 0.15, 'negative': 0.1}
        }
        
        reviews = []
        for i in range(n_reviews):
            date = random.choice(dates)
            product = random.choice(list(products.keys()))
            
            # Generate sentiment based on product distribution
            sentiment_prob = random.random()
            probs = products[product]
            
            if sentiment_prob < probs['positive']:
                sentiment = 'Positive'
                rating = random.choice([4, 5])
                confidence = random.uniform(0.7, 0.95)
            elif sentiment_prob < probs['positive'] + probs['neutral']:
                sentiment = 'Neutral'
                rating = 3
                confidence = random.uniform(0.6, 0.8)
            else:
                sentiment = 'Negative'
                rating = random.choice([1, 2])
                confidence = random.uniform(0.75, 0.9)
            
            reviews.append({
                'date': date,
                'product_category': product,
                'sentiment': sentiment,
                'rating': rating,
                'confidence': confidence,
                'review_length': random.randint(50, 500)
            })
        
        return pd.DataFrame(reviews)
    
    def create_sentiment_overview(self, df):
        """
        Create overview metrics showing overall sentiment distribution.
        
        This section provides executives with immediate insights into
        overall customer satisfaction levels across all products.
        """
        st.subheader("📈 Sentiment Overview")
        
        # Calculate sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        total_reviews = len(df)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_pct = (sentiment_counts.get('Positive', 0) / total_reviews) * 100
            st.metric(
                label="😊 Positive Reviews",
                value=f"{positive_pct:.1f}%",
                delta=f"{sentiment_counts.get('Positive', 0)} reviews"
            )
        
        with col2:
            neutral_pct = (sentiment_counts.get('Neutral', 0) / total_reviews) * 100
            st.metric(
                label="😐 Neutral Reviews", 
                value=f"{neutral_pct:.1f}%",
                delta=f"{sentiment_counts.get('Neutral', 0)} reviews"
            )
        
        with col3:
            negative_pct = (sentiment_counts.get('Negative', 0) / total_reviews) * 100
            st.metric(
                label="😞 Negative Reviews",
                value=f"{negative_pct:.1f}%",
                delta=f"{sentiment_counts.get('Negative', 0)} reviews"
            )
        
        # Overall sentiment score (weighted average)
        sentiment_weights = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        overall_score = sum(sentiment_counts.get(s, 0) * w for s, w in sentiment_weights.items()) / total_reviews
        
        st.markdown(f"""
        **Overall Sentiment Score**: {overall_score:.2f} (Range: -1 to +1)
        
        This score gives you a quick health check of customer satisfaction. 
        Values closer to +1 indicate overwhelmingly positive sentiment, while 
        values closer to -1 suggest significant customer dissatisfaction requiring attention.
        """)
    
    def create_sentiment_trends(self, df):
        """
        Visualize sentiment trends over time.
        
        Time series analysis helps identify patterns, seasonal effects,
        and the impact of business decisions on customer sentiment.
        """
        st.subheader("📊 Sentiment Trends Over Time")
        
        # Prepare data for time series analysis
        df['date'] = pd.to_datetime(df['date'])
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate rolling averages for smoother trends
        daily_sentiment_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
        rolling_avg = daily_sentiment_pct.rolling(window=7, center=True).mean()
        
        # Create interactive plot
        fig = go.Figure()
        
        colors = {'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment in rolling_avg.columns:
                fig.add_trace(go.Scatter(
                    x=rolling_avg.index,
                    y=rolling_avg[sentiment],
                    mode='lines',
                    name=sentiment,
                    line=dict(color=colors[sentiment], width=2),
                    hovertemplate=f'{sentiment}: %{{y:.1f}}%<extra></extra>'
                ))
        
        fig.update_layout(
            title="Sentiment Trends (7-day Moving Average)",
            xaxis_title="Date",
            yaxis_title="Percentage of Reviews",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights section
        st.markdown("""
        **💡 How to interpret this chart:**
        
        This visualization shows how customer sentiment has evolved over time. Look for:
        - **Sudden drops** in positive sentiment that might indicate product issues
        - **Seasonal patterns** that could help with inventory and marketing planning  
        - **Long-term trends** that show whether your customer satisfaction is improving
        - **Correlation with business events** like product launches or marketing campaigns
        """)
    
    def create_category_analysis(self, df):
        """
        Analyze sentiment by product category.
        
        This analysis helps identify which product categories are performing
        well and which need attention from a customer satisfaction perspective.
        """
        st.subheader("🏷️ Sentiment by Product Category")
        
        # Calculate sentiment distribution by category
        category_sentiment = df.groupby(['product_category', 'sentiment']).size().unstack(fill_value=0)
        category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
        
        # Create stacked bar chart
        fig = px.bar(
            category_sentiment_pct,
            x=category_sentiment_pct.index,
            y=['Positive', 'Neutral', 'Negative'],
            title="Sentiment Distribution by Product Category",
            labels={'value': 'Percentage of Reviews', 'index': 'Product Category'},
            color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category performance table
        st.subheader("📋 Category Performance Summary")
        
        # Calculate key metrics per category
        category_metrics = df.groupby('product_category').agg({
            'sentiment': lambda x: (x == 'Positive').mean() * 100,
            'rating': 'mean',
            'confidence': 'mean'
        }).round(2)
        
        category_metrics.columns = ['% Positive', 'Avg Rating', 'Avg Confidence']
        category_metrics = category_metrics.sort_values('% Positive', ascending=False)
        
        # Style the dataframe
        st.dataframe(
            category_metrics.style.format({
                '% Positive': '{:.1f}%',
                'Avg Rating': '{:.2f}',
                'Avg Confidence': '{:.2f}'
            }).background_gradient(subset=['% Positive'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    def create_confidence_analysis(self, df):
        """
        Analyze model confidence levels across predictions.
        
        Understanding model confidence helps identify which predictions
        to trust and which reviews might need manual verification.
        """
        st.subheader("🎯 Model Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig = px.histogram(
                df, 
                x='confidence', 
                color='sentiment',
                title="Confidence Distribution by Sentiment",
                nbins=20,
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average confidence by sentiment
            avg_confidence = df.groupby('sentiment')['confidence'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                title="Average Model Confidence by Sentiment",
                color=avg_confidence.index,
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#6c757d', 'Negative': '#dc3545'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Low confidence reviews that need attention
        low_confidence_threshold = 0.7
        low_confidence_reviews = df[df['confidence'] < low_confidence_threshold]
        
        if not low_confidence_reviews.empty:
            st.warning(f"""
            **⚠️ Reviews Requiring Manual Review**: {len(low_confidence_reviews)} reviews have 
            confidence below {low_confidence_threshold:.0%}. These predictions might be uncertain 
            and could benefit from manual verification.
            """)
    
    def create_actionable_insights(self, df):
        """
        Generate actionable business insights based on sentiment analysis.
        
        This section translates data insights into concrete business
        recommendations that stakeholders can act upon.
        """
        st.subheader("🚀 Actionable Insights & Recommendations")
        
        # Calculate key metrics for insights
        total_reviews = len(df)
        negative_reviews = df[df['sentiment'] == 'Negative']
        positive_reviews = df[df['sentiment'] == 'Positive']
        
        # Recent trend analysis (last 30 days)
        recent_date = df['date'].max() - timedelta(days=30)
        recent_reviews = df[df['date'] > recent_date]
        recent_negative_pct = (recent_reviews['sentiment'] == 'Negative').mean() * 100
        
        # Generate insights
        insights = []
        
        # Insight 1: Overall sentiment health
        negative_pct = len(negative_reviews) / total_reviews * 100
        if negative_pct > 20:
            insights.append({
                'type': 'warning',
                'title': 'High Negative Sentiment Alert',
                'content': f'{negative_pct:.1f}% of reviews are negative. This is above the healthy threshold of 15%.',
                'action': 'Investigate common complaint themes and prioritize product improvements.'
            })
        elif negative_pct < 10:
            insights.append({
                'type': 'success',
                'title': 'Excellent Customer Satisfaction',
                'content': f'Only {negative_pct:.1f}% of reviews are negative. This indicates strong product quality and customer satisfaction.',
                'action': 'Maintain current quality standards and consider expanding successful product lines.'
            })
        
        # Insight 2: Category performance
        category_performance = df.groupby('product_category')['sentiment'].apply(lambda x: (x == 'Negative').mean() * 100)
        worst_category = category_performance.idxmax()
        best_category = category_performance.idxmin()
        
        insights.append({
            'type': 'info',
            'title': 'Category Performance Variation',
            'content': f'{worst_category} has the highest negative sentiment ({category_performance[worst_category]:.1f}%) while {best_category} performs best ({category_performance[best_category]:.1f}%).',
            'action': f'Study {best_category} success factors and apply learnings to improve {worst_category}.'
        })
        
        # Insight 3: Recent trends
        if recent_negative_pct > negative_pct:
            insights.append({
                'type': 'warning',
                'title': 'Recent Sentiment Decline',
                'content': f'Negative sentiment has increased to {recent_negative_pct:.1f}% in the last 30 days.',
                'action': 'Investigate recent changes in products, shipping, or customer service that might be causing issues.'
            })
        
        # Display insights
        for insight in insights:
            if insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**\n\n{insight['content']}\n\n💡 **Recommended Action:** {insight['action']}")
            elif insight['type'] == 'success':
                st.success(f"**{insight['title']}**\n\n{insight['content']}\n\n💡 **Recommended Action:** {insight['action']}")
            else:
                st.info(f"**{insight['title']}**\n\n{insight['content']}\n\n💡 **Recommended Action:** {insight['action']}")
    
    def create_sidebar_filters(self, df):
        """
        Create interactive filters for the dashboard.
        
        Filters allow users to drill down into specific time periods,
        product categories, or sentiment types for focused analysis.
        """
        st.sidebar.header("🔍 Dashboard Filters")
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
        
        # Product category filter
        categories = st.sidebar.multiselect(
            "Select Product Categories",
            options=df['product_category'].unique(),
            default=df['product_category'].unique()
        )
        
        # Sentiment filter
        sentiments = st.sidebar.multiselect(
            "Select Sentiment Types",
            options=df['sentiment'].unique(),
            default=df['sentiment'].unique()
        )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Minimum Confidence Level",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
        
        # Apply filters
        if len(date_range) == 2:
            df = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                    (df['date'] <= pd.to_datetime(date_range[1]))]
        
        df = df[df['product_category'].isin(categories)]
        df = df[df['sentiment'].isin(sentiments)]
        df = df[df['confidence'] >= confidence_threshold]
        
        # Show filtered data stats
        st.sidebar.markdown(f"""
        **Filtered Data Summary:**
        - Total Reviews: {len(df):,}
        - Date Range: {len(date_range) == 2 and f"{date_range[0]} to {date_range[1]}" or "All dates"}
        - Categories: {len(categories)}
        - Avg Confidence: {df['confidence'].mean():.2f}
        """)
        
        return df
    
    def run_dashboard(self):
        """
        Main dashboard execution function.
        
        This orchestrates all dashboard components and handles the
        overall user interaction flow.
        """
        # Generate or load data
        df = self.generate_sample_data()
        
        # Apply sidebar filters
        filtered_df = self.create_sidebar_filters(df)
        
        # Check if data exists after filtering
        if filtered_df.empty:
            st.error("No data matches your current filters. Please adjust your selection.")
            return
        
        # Create dashboard sections
        self.create_sentiment_overview(filtered_df)
        st.divider()
        
        self.create_sentiment_trends(filtered_df)
        st.divider()
        
        self.create_category_analysis(filtered_df)
        st.divider()
        
        self.create_confidence_analysis(filtered_df)
        st.divider()
        
        self.create_actionable_insights(filtered_df)
        
        # Footer with additional information
        st.markdown("---")
        st.markdown("""
        **About this Dashboard:**
        
        This sentiment analysis dashboard processes customer reviews using advanced natural language processing 
        to automatically classify sentiment and provide actionable business insights. The system analyzes text 
        patterns, emotional indicators, and contextual clues to determine whether customers are satisfied, 
        neutral, or dissatisfied with products.
        
        **Key Features:**
        - **Real-time sentiment classification** with confidence scores
        - **Trend analysis** to identify patterns over time
        - **Category-based insights** for targeted improvements
        - **Actionable recommendations** based on data patterns
        - **Interactive filtering** for detailed analysis
        
        **Data Sources:** This demo uses simulated data. In production, connect to your review database or API.
        """)

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = SentimentDashboard()
    dashboard.run_dashboard()
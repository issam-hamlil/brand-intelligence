"""
Streamlit Dashboard - Professional Brand Intelligence Platform
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Import backend
from backend import BrandIntelligencePipeline

# Page config
st.set_page_config(
    page_title="Brand Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Persona cards */
    .persona-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .persona-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown('<div class="main-header">Brand Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Customer Insights & Regional Sentiment Analysis</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    
    # System status (hidden from end users, just visual indicator)
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    system_ready = False
    
    with st.expander("System Status", expanded=False):
        try:
            import requests
            response = requests.get(f"{ollama_host}/api/tags", timeout=2)
            if response.status_code == 200:
                st.markdown('<span class="status-badge status-success">● System Online</span>', unsafe_allow_html=True)
                system_ready = True
                models = response.json().get('models', [])
                if not models:
                    st.caption("⚠️ AI model required. Contact administrator.")
            else:
                st.markdown('<span class="status-badge status-error">● System Offline</span>', unsafe_allow_html=True)
                st.caption("Please contact system administrator.")
        except Exception as e:
            st.markdown('<span class="status-badge status-error">● System Offline</span>', unsafe_allow_html=True)
            st.caption("Please contact system administrator.")
    
    st.divider()
    
    st.markdown("### Data Upload")
    uploaded_file = st.file_uploader(
        "Upload brand mentions (CSV format)",
        type=['csv'],
        help="CSV file with columns: text, region"
    )
    
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name} loaded")
    
    st.divider()
    
    st.markdown("### Analysis Parameters")
    
    brand_keywords = st.text_input(
        "Brand Keywords",
        value="sustainable, quality, innovation",
        help="Comma-separated keywords that define your brand identity"
    )
    
    num_personas = st.slider(
        "Customer Personas to Generate", 
        min_value=2, 
        max_value=5, 
        value=3,
        help="Number of distinct customer segments to identify"
    )
    
    st.divider()
    
    analyze_button = st.button(
        "Generate Intelligence Report",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_file or st.session_state.processing
    )
    
    if analyze_button and uploaded_file:
        st.session_state.processing = True
        
        with st.spinner("Analyzing data..."):
            # Save uploaded file
            csv_path = f"/tmp/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize pipeline
            try:
                if st.session_state.pipeline is None:
                    st.session_state.pipeline = BrandIntelligencePipeline()
                
                # Run pipeline
                keywords = [k.strip() for k in brand_keywords.split(',')]
                st.session_state.report = st.session_state.pipeline.run_full_pipeline(
                    csv_path=csv_path,
                    brand_keywords=keywords,
                    num_personas=num_personas
                )
                st.session_state.processing = False
                st.success("✓ Analysis complete!")
                st.rerun()
            except Exception as e:
                st.session_state.processing = False
                st.error(f"Analysis failed: {str(e)}")
                st.caption("Please check your data format and try again.")

# Main content
if st.session_state.report:
    report = st.session_state.report
    
    # Key Metrics Dashboard
    regional_summary = report.get('regional_summary', [])
    personas = report.get('personas', [])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Geographic Regions",
            value=len(regional_summary),
            help="Number of regions analyzed"
        )
    
    with col2:
        st.metric(
            label="Customer Personas",
            value=len(personas),
            help="Identified customer segments"
        )
    
    with col3:
        total_mentions = sum(r.get('mention_count', 0) for r in regional_summary)
        st.metric(
            label="Data Points Analyzed",
            value=f"{total_mentions:,}",
            help="Total brand mentions processed"
        )
    
    with col4:
        if regional_summary:
            avg_sentiment = sum(r.get('avg_sentiment', 0) for r in regional_summary) / len(regional_summary)
            sentiment_delta = (avg_sentiment - 0.5) * 100
            st.metric(
                label="Overall Sentiment",
                value=f"{avg_sentiment:.2f}",
                delta=f"{sentiment_delta:+.1f}%",
                help="Average brand sentiment score (0-1 scale)"
            )
    
    st.divider()
    
    # Main Analysis Section
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="section-header">Regional Sentiment Analysis</div>', unsafe_allow_html=True)
        
        if regional_summary:
            df_regions = pd.DataFrame(regional_summary)
            
            # Create professional bar chart
            fig = go.Figure()
            
            # Color mapping based on sentiment
            colors = ['#ef4444' if x < 0.4 else '#f59e0b' if x < 0.7 else '#10b981' 
                     for x in df_regions['avg_sentiment']]
            
            fig.add_trace(go.Bar(
                x=df_regions['region'],
                y=df_regions['avg_sentiment'],
                marker_color=colors,
                text=df_regions['avg_sentiment'].apply(lambda x: f'{x:.2f}'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<br><extra></extra>'
            ))
            
            fig.update_layout(
                title="Sentiment Score by Region",
                xaxis_title="Region",
                yaxis_title="Sentiment Score",
                yaxis=dict(range=[0, 1.0]),
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color="#1a1a2e"),
                showlegend=False,
                margin=dict(t=40, b=40, l=40, r=40)
            )
            
            fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='#e2e8f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table with professional styling
            st.markdown("##### Detailed Regional Metrics")
            
            display_df = df_regions.copy()
            display_df['avg_sentiment'] = display_df['avg_sentiment'].apply(lambda x: f'{x:.3f}')
            display_df.columns = ['Region', 'Mentions', 'Avg Sentiment', 'Emotions']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
    
    with col_right:
        st.markdown('<div class="section-header">Insights Summary</div>', unsafe_allow_html=True)
        
        if regional_summary:
            # Top performing region
            top_region = max(regional_summary, key=lambda x: x.get('avg_sentiment', 0))
            st.markdown("**Best Performing Region**")
            st.info(f"**{top_region['region'].upper()}**\n\nSentiment: {top_region['avg_sentiment']:.3f}\n\n{top_region['mention_count']:,} mentions analyzed")
            
            # Lowest performing region
            low_region = min(regional_summary, key=lambda x: x.get('avg_sentiment', 0))
            if low_region['avg_sentiment'] < 0.5:
                st.markdown("**Attention Required**")
                st.warning(f"**{low_region['region'].upper()}**\n\nSentiment: {low_region['avg_sentiment']:.3f}\n\nConsider targeted engagement strategy")
            
            # Emotion distribution
            st.markdown("**Emotion Distribution**")
            all_emotions = []
            for r in regional_summary:
                emotions = r.get('emotions', '')
                if emotions:
                    all_emotions.extend(emotions.split(','))
            
            if all_emotions:
                emotion_counts = pd.Series(all_emotions).value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=emotion_counts.index,
                    values=emotion_counts.values,
                    hole=.4,
                    marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'])
                )])
                
                fig_pie.update_layout(
                    height=250,
                    margin=dict(t=0, b=0, l=0, r=0),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1)
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Customer Personas Section
    st.divider()
    st.markdown('<div class="section-header">Customer Persona Analysis</div>', unsafe_allow_html=True)
    
    if personas:
        # Display personas in a grid
        cols = st.columns(min(len(personas), 3))
        
        for idx, persona in enumerate(personas):
            with cols[idx % 3]:
                # Persona card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {persona['name']}
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">
                        Age {persona['age_range']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Match score indicator
                if persona.get('best_regions'):
                    avg_score = sum(r['score'] for r in persona['best_regions']) / len(persona['best_regions'])
                    st.progress(avg_score, text=f"Regional Match: {avg_score:.0%}")
                
                # Detailed information
                with st.expander("View Detailed Profile", expanded=False):
                    st.markdown("**Key Interests**")
                    for interest in persona.get('interests', []):
                        st.markdown(f"• {interest}")
                    
                    st.markdown("**Primary Motivation**")
                    st.info(persona.get('motivation', 'N/A'))
                    
                    st.markdown("**Recommended Messaging Tone**")
                    st.text(persona.get('tone', 'N/A'))
                    
                    st.markdown("**Sample Campaign Message**")
                    st.success(f'"{persona.get("ad_sample", "N/A")}"')
                    
                    st.markdown("**Target Regions**")
                    for region in persona.get('best_regions', [])[:3]:
                        match_pct = region['score'] * 100
                        st.markdown(f"• **{region['region']}** - {match_pct:.0f}% match confidence")
    
    # Export Options
    st.divider()
    st.markdown('<div class="section-header">Export & Reporting</div>', unsafe_allow_html=True)
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        json_data = json.dumps(report, indent=2)
        st.download_button(
            label="Download Complete Report (JSON)",
            data=json_data,
            file_name=f"brand_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
            help="Full analysis data in JSON format"
        )
    
    with col_exp2:
        if personas:
            personas_df = pd.DataFrame([{
                'Persona Name': p['name'],
                'Age Range': p['age_range'],
                'Primary Motivation': p['motivation'],
                'Messaging Tone': p['tone']
            } for p in personas])
            
            st.download_button(
                label="Download Persona Profiles (CSV)",
                data=personas_df.to_csv(index=False),
                file_name=f"customer_personas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Customer persona details in spreadsheet format"
            )
    
    with col_exp3:
        if regional_summary:
            regional_df = pd.DataFrame(regional_summary)
            st.download_button(
                label="Download Regional Data (CSV)",
                data=regional_df.to_csv(index=False),
                file_name=f"regional_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Regional sentiment metrics in spreadsheet format"
            )

else:
    # Welcome screen for first-time users
    st.markdown("### Welcome to Brand Intelligence Platform")
    st.markdown("""
    This platform combines advanced AI analysis with customer segmentation to provide 
    actionable insights about your brand perception across different markets.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Customer Persona Analysis")
        st.markdown("""
        - Automatically identify distinct customer segments
        - Generate detailed demographic and psychographic profiles
        - Receive tailored messaging recommendations
        - Map personas to optimal geographic regions
        """)
    
    with col2:
        st.markdown("#### 📊 Regional Sentiment Mapping")
        st.markdown("""
        - Analyze brand sentiment across multiple regions
        - Track emotional responses to your brand
        - Identify high-performing and opportunity markets
        - Quantify brand perception with sentiment scores
        """)
    
    st.divider()
    
    st.markdown("### Getting Started")
    st.info("👈 Upload your brand mention data in the sidebar to begin analysis")
    
    st.markdown("#### Required Data Format")
    st.markdown("Your CSV file should contain the following columns:")
    
    sample_df = pd.DataFrame({
        'text': [
            'Excellent product quality and great customer service',
            'Good value for the price, will buy again',
            'Love the sustainable packaging and eco-friendly approach'
        ],
        'region': ['northeast', 'midwest', 'west']
    })
    
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    # Download sample template
    csv_sample = sample_df.to_csv(index=False)
    st.download_button(
        label="Download Sample Template",
        data=csv_sample,
        file_name="brand_mentions_template.csv",
        mime="text/csv",
        help="Download a CSV template to format your data correctly"
    )
    
    st.divider()
    
    with st.expander("📖 Analysis Methodology"):
        st.markdown("""
        **Data Processing Pipeline:**
        
        1. **Sentiment Classification** - Each mention is analyzed for emotional tone and sentiment polarity
        2. **Regional Aggregation** - Data is grouped by geographic region for comparative analysis
        3. **Persona Generation** - Machine learning algorithms identify distinct customer segments based on language patterns and sentiment
        4. **Strategic Matching** - Personas are mapped to regions where they're most likely to resonate
        
        **Output Deliverables:**
        - Regional sentiment scores and emotion distribution
        - Detailed customer persona profiles with targeting recommendations
        - Confidence-scored persona-to-region matches
        - Exportable reports in multiple formats
        """)
    
    with st.expander("🔒 Data Privacy & Security"):
        st.markdown("""
        - All analysis is performed in a secure, isolated environment
        - Your data is never shared with third parties
        - Processing occurs entirely within your infrastructure
        - No data retention beyond the active session
        - Export capabilities allow you to maintain full control of insights
        """)
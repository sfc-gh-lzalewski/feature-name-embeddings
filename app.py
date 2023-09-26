import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

margins_css = """
        <style>
        .main > div {
            padding-left: 2rem;
            padding-right: 0rem;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 5rem;
        }
        </style>
        """

st.markdown(margins_css, unsafe_allow_html=True)

# Title of the web app
st.title('Feature Descriptions Embedding Space')

df_embeddings_openai_tsne = pd.read_csv('results/feature_embeddings_openai_tsne.csv')
df_embeddings_openai_umap = pd.read_csv('results/feature_embeddings_openai_umap.csv')
df_embeddings_bert_tsne = pd.read_csv('results/feature_embeddings_bert_tsne.csv')

st.subheader('OpenAI TSNE')
fig = px.scatter(df_embeddings_openai_tsne, x='x', y='y', hover_data=['feature_name', "num_unique_values", "percent_missing", "dataset_id", 'description', 'data_type', 'examples'])
fig.update_traces(hovertemplate='name: %{hovertext}<br>dataset_id: %{customdata[2]}<br>type: %{customdata[3]}<br>unique_values: %{customdata[0]}<br>percent_missing: %{customdata[1]}<br>description: %{customdata[5]}',
                hovertext=df_embeddings_openai_tsne['feature_name'],
                customdata=df_embeddings_openai_tsne[['num_unique_values', 'percent_missing', 'dataset_id', 'data_type', 'examples', 'description']].values)

# Set size of the plot
fig.update_layout(
    autosize=False,
    width=1400,
    height=1000,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
)
st.plotly_chart(fig)

st.subheader('OpenAI UMAP')
fig = px.scatter(df_embeddings_openai_umap, x='x', y='y', hover_data=['feature_name', "num_unique_values", "percent_missing", "dataset_id", 'description', 'data_type', 'examples'])
fig.update_traces(hovertemplate='name: %{hovertext}<br>dataset_id: %{customdata[2]}<br>type: %{customdata[3]}<br>unique_values: %{customdata[0]}<br>percent_missing: %{customdata[1]}<br>description: %{customdata[5]}',
                hovertext=df_embeddings_openai_umap['feature_name'],
                customdata=df_embeddings_openai_umap[['num_unique_values', 'percent_missing', 'dataset_id', 'data_type', 'examples', 'description']].values)

# Set size of the plot
fig.update_layout(
    autosize=False,
    width=1400,
    height=1000,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
)
st.plotly_chart(fig)


st.subheader('bert-base-uncased UMAP')
fig = px.scatter(df_embeddings_bert_tsne, x='x', y='y', hover_data=['feature_name', "num_unique_values", "percent_missing", "dataset_id", 'description', 'data_type', 'examples'])
fig.update_traces(hovertemplate='name: %{hovertext}<br>dataset_id: %{customdata[2]}<br>type: %{customdata[3]}<br>unique_values: %{customdata[0]}<br>percent_missing: %{customdata[1]}<br>description: %{customdata[5]}',
                hovertext=df_embeddings_bert_tsne['feature_name'],
                customdata=df_embeddings_bert_tsne[['num_unique_values', 'percent_missing', 'dataset_id', 'data_type', 'examples', 'description']].values)

# Set size of the plot
fig.update_layout(
    autosize=False,
    width=1400,
    height=1000,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
)
st.plotly_chart(fig)


# Optional: Display the raw data as a table (comment out if not needed)
# if st.checkbox('Show raw data'):
st.write(df_embeddings_openai_tsne)



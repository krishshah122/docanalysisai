import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import PyPDF2
import io
import base64
from together import Together
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from utils import DataAnalyzer
import os
load_dotenv()
st.set_page_config(page_title="Document Analysis Agent", layout="wide")
@st.cache_resource
def api_setup():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        st.error("Together api key not set")
        st.stop()
    return Together(api_key=api_key)
client = api_setup()
class DocumentProcessor:
    @staticmethod
    def process_txt(file):
        content = file.read().decode('utf-8')
        return {"type": "text", "content": content, "summary": f"Text document with {len(content)} characters"}
    @staticmethod
    def process_csv(file):
        df = pd.read_csv(file)
        summary = DataAnalyzer.get_data_profile(df)
        return {"type": "csv", "content": df, "summary": summary}
    @staticmethod
    def process_pdf(file):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return {"type": "pdf", "content": text, "summary": f"PDF with {len(pdf_reader.pages)} pages and {len(text)} characters"}
    @staticmethod
    def process_image(file):
        image = Image.open(file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        summary = {
            "format": image.format,
            "size": image.size,
            "mode": image.mode
        }
        return {"type": "image", "content": img_str, "summary": summary, "pil_image": image}
class AIAgent:
    def __init__(self, client):
        self.client = client
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.conversation_history = []
    def analyze_document(self, doc_data: Dict[str, Any]) -> str:
        if doc_data["type"] == "csv":
           prompt = DataAnalyzer.csvprompt(doc_data["summary"])
        elif doc_data["type"] in ["text", "pdf"]:
            prompt =DataAnalyzer.textpdfprompt(doc_data["type"], doc_data["content"])
        elif doc_data["type"] == "image":
            prompt = DataAnalyzer.imageprompt(doc_data["summary"])
        return self.airesponse(prompt) 
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        if context["type"] == "csv":
            info = f"""
            Dataset context:
            - Shape: {context['summary']['rows']} rows, {context['summary']['columns']} columns
            - Columns: {context['summary']['column_names']}
            - Sample data: {context['summary']['sample_data']}
            """
        else:
            info = f"Document type: {context['type']}\nContent summary: {context['summary']}"
        prompt = f"""
        Based on the following document context:
        {info}
        Question: {question}
        Provide a detailed answer based on the available data. If you need specific calculations or data analysis, explain what would be needed.
        """
        response = self.airesponse(prompt)
        self.conversation_history.append({"question": question, "answer": response})
        return response
    def suggest_visualizations(self, doc_data: Dict[str, Any]) -> List[str]:
        if doc_data["type"] != "csv":
            return ["No visualizations available for this document type"]
        df = doc_data["content"]
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(numeric_cols) >= 2:
            suggestions.append("Scatter plot between numeric variables")
            suggestions.append("Correlation heatmap")
        if len(numeric_cols) >= 1:
            suggestions.append("Distribution histogram")
            suggestions.append("Box plot for outlier detection")
        if len(categorical_cols) >= 1:
            suggestions.append("Bar chart for categorical data")
            suggestions.append("Pie chart for proportions")
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            suggestions.append("Grouped analysis by category")
        return suggestions
    def airesponse(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting AI response: {str(e)}"
class Visualizer:
    @staticmethod
    def create_visualization(df: pd.DataFrame, viz_type: str, columns: List[str] = None):
        if df is None or df.empty:
            st.warning("DataFrame is empty or missing.")
            return None
        try:
            if viz_type == "bar":
                return Visualizer._bar_chart(df, columns)
            elif viz_type == "histogram":
                return Visualizer._histogram(df, columns)

            elif viz_type == "scatter":
                return Visualizer._scatter(df, columns)

            elif viz_type == "correlation":
                return Visualizer._correlation_heatmap(df)

            elif viz_type == "box":
                return Visualizer._box_plot(df, columns)
            elif viz_type == "pie":
                return Visualizer._pie_chart(df, columns)
            else:
                st.warning("Unknown visualization type.")
                return None
        except Exception as e:
            st.error(f"Error generating {viz_type} plot: {e}")
            return None
    @staticmethod
    def _bar_chart(df: pd.DataFrame, columns: List[str]):
        if not columns or len(columns) < 1:
            st.warning("Please select one column for the bar chart.")
            return None
        col = columns[0]
        df[col] = df[col].astype(str)
        counts = df[col].value_counts().nlargest(10).reset_index()
        counts.columns = [col, 'Count']
        fig = px.bar(counts, x=col, y='Count', title=f"Top {len(counts)} values in '{col}'", text='Count')
        fig.update_layout(xaxis_title=col, yaxis_title="Frequency")
        return fig
    @staticmethod
    def _histogram(df: pd.DataFrame, columns: List[str]):
        if not columns or len(columns) < 1:
            st.warning("Please select one numeric column for histogram.")
            return None
        col = columns[0]
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.warning(f"Column '{col}' is not numeric.")
            return None
        fig = px.histogram(df, x=col, title=f"Histogram of '{col}'", nbins=30)
        fig.update_layout(xaxis_title=col, yaxis_title="Frequency")
        return fig
    @staticmethod
    def _scatter(df: pd.DataFrame, columns: List[str]):
        if not columns or len(columns) < 2:
            st.warning("Please select two numeric columns for scatter plot.")
            return None
        x_col, y_col = columns[0], columns[1]
        if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
            st.warning(f"Both '{x_col}' and '{y_col}' must be numeric.")
            return None
        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        return fig
    @staticmethod
    def _correlation_heatmap(df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.warning("At least two numeric columns are required for correlation heatmap.")
            return None
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        return fig
    @staticmethod
    def _box_plot(df: pd.DataFrame, columns: List[str]):
        if not columns or len(columns) < 1:
            st.warning("Please select one column for box plot.")
            return None
        col = columns[0]
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.warning(f"Column '{col}' is not numeric.")
            return None
        fig = px.box(df, y=col, title=f"Box Plot of '{col}'")
        fig.update_layout(yaxis_title=col)
        return fig
    @staticmethod
    def _pie_chart(df: pd.DataFrame, columns: List[str]):
         if not columns or len(columns) < 1:
           st.warning("Please select one column for the pie chart.")
           return None
         col = columns[0]
         df[col] = df[col].astype(str)
         counts = df[col].value_counts().nlargest(10).reset_index()
         counts.columns = [col, 'Count']
         fig = px.pie(counts, names=col, values='Count', title=f"Pie Chart of Top {len(counts)} '{col}' Values")
         return fig
def main():
    st.title("Document Analysis Agent")
    st.markdown("Upload documents and get AI-powered analysis with visualizations")
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = {}
    if 'agent' not in st.session_state:
        st.session_state.agent = AIAgent(client)
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'csv', 'pdf', 'png', 'jpg', 'jpeg'],
            help="Supported formats: TXT, CSV, PDF, PNG, JPG"
        )
        if uploaded_file:
            with st.spinner("Processing document..."):
                processor = DocumentProcessor()
                if uploaded_file.type == "text/plain":
                    doc_data = processor.process_txt(uploaded_file)
                elif uploaded_file.type == "text/csv":
                    doc_data = processor.process_csv(uploaded_file)
                elif uploaded_file.type == "application/pdf":
                    doc_data = processor.process_pdf(uploaded_file)
                elif uploaded_file.type.startswith("image/"):
                    doc_data = processor.process_image(uploaded_file)
                st.session_state.processed_docs[uploaded_file.name] = doc_data
                st.session_state.current_doc = uploaded_file.name
                st.success(f" Processed {uploaded_file.name}")
        if st.session_state.processed_docs:
            st.header("Select Document")
            selected_doc = st.selectbox(
                "Choose document to analyze:",
                list(st.session_state.processed_docs.keys())
            )
            st.session_state.current_doc = selected_doc

    if st.session_state.current_doc:
        doc_data = st.session_state.processed_docs[st.session_state.current_doc]
        st.header(f"Analysis: {st.session_state.current_doc}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("AI Analysis")
            if st.button("Analyze Document", type="primary"):
                with st.spinner("AI is analyzing your document..."):
                    analysis = st.session_state.agent.analyze_document(doc_data)
                    st.markdown(analysis)
            st.subheader("Ask Questions")
            question = st.text_input("Ask anything about your document:")
            if question and st.button("Get Answer"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.agent.answer_question(question, doc_data)
                    st.markdown(f"Answer: {answer}")
            if st.session_state.agent.conversation_history:
                st.subheader("Conversation History")
                for i, conv in enumerate(st.session_state.agent.conversation_history[-3:]):
                    with st.expander(f"Q{i+1}: {conv['question'][:50]}..."):
                        st.markdown(f"Q: {conv['question']}")
                        st.markdown(f"A: {conv['answer']}")
        with col2:
            st.subheader("Document Summary")
            if doc_data["type"] == "csv":
                df = doc_data["content"]
                st.metric("Rows", doc_data["summary"]["rows"])
                st.metric("Columns", doc_data["summary"]["columns"])
                st.write("**Columns:**", ", ".join(doc_data["summary"]["column_names"]))
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            elif doc_data["type"] == "image":
                st.image(doc_data["pil_image"], caption="Uploaded Image", use_container_width=True)
            else:
                st.write(f"Type: {doc_data['type'].upper()}")
                st.write(f"Summary: {doc_data['summary']}")
        if doc_data["type"] == "csv":
            st.header("Visualizations")
            df = doc_data["content"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            all_cols = list(df.columns)
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                viz_type = st.selectbox(
                    "Visualization Type:",
                    ["histogram", "scatter", "bar", "correlation", "box","pie"]
                )
            with viz_col2:
                if viz_type in ["histogram", "bar", "box"]:
                    selected_cols = st.multiselect("Select Column:", all_cols, max_selections=1)
                elif viz_type == "scatter":
                    selected_cols = st.multiselect("Select Columns (X, Y):", numeric_cols, max_selections=2)
                elif viz_type == "pie":
                      selected_cols = st.multiselect("Select Category and Value Columns:", all_cols, max_selections=2)
                else:
                    selected_cols = []
            if st.button("Generate Visualization"):
                visualizer = Visualizer()
                fig = visualizer.create_visualization(df, viz_type, selected_cols)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not generate visualization with selected parameters")
            st.subheader("Suggested Visualizations")
            suggestions = st.session_state.agent.suggest_visualizations(doc_data)
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
    else:
        st.info("Upload a document from the sidebar to get started!")

if __name__ == "__main__":
    main()

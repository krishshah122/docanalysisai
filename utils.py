"""Utility functions for document processing and analysis"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import json

class DataAnalyzer:
    @staticmethod
    def getcsvanalysis(df: pd.DataFrame) -> Dict[str, Any]:
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head().to_dict()
        }
        return profile
    @staticmethod
    def csvprompt(summary: Dict[str, Any]) -> str:
        return f"""
       Analyze this CSV dataset with the following information:
    - Rows: {summary['rows']}
    - Columns: {summary['columns']}
    - Column names: {summary['column_names']}
    - Data types: {summary['dtypes']}
    - Missing values: {summary['missing_values']}
    - Sample data: {summary['sample_data']}
    Provide a comprehensive analysis including:
    1. Data overview and structure
    2. Key insights and patterns
    3. Data quality assessment
    4. Suggested visualizations
    5. Potential analysis directions
    """
    @staticmethod
    def textpdfprompt(doc_type: str, content: str) -> str:
      return f"""
    Analyze this {doc_type} document:
    Content preview: {content[:2000]}...
    Provide analysis including:
    1. Document summary
    2. Key topics and themes
    3. Important insights
    4. Suggested follow-up questions
    """
    def imageprompt(summary: Dict[str, Any]) -> str:
        return f"""
    Analyze this image with the following properties:
    - Format: {summary['format']}
    - Size: {summary['size']}
    - Mode: {summary['mode']}
    Describe what you see and provide insights about the image content.
    """

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[str]:
        patterns = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.7)
            for i, j in zip(high_corr[0], high_corr[1]):
                if i < j: 
                    patterns.append(f"High correlation ({corr_matrix.iloc[i,j]:.2f}) between {numeric_cols[i]} and {numeric_cols[j]}")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            if len(outliers) > 0:
                patterns.append(f"Found {len(outliers)} outliers in {col}")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            patterns.append(f"Missing data in columns: {', '.join(missing_cols)}")

        return patterns

class TextProcessor:
    @staticmethod
    def extract_key_phrases(text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases from text"""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    @staticmethod
    def get_text_stats(text: str) -> Dict[str, Any]:
        sentences = text.split('.')
        words = text.split()
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": len(text) / len(words) if words else 0
        }
def format_number(num: float) -> str:
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.1f}"
def safe_json_loads(json_str: str) -> Dict[str, Any]:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}

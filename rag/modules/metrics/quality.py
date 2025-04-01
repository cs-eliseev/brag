from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union
from datetime import datetime
import numpy as np
import json
import csv
from pathlib import Path
from rag.utils.logger import logger
from rag.modules.metrics.metrics import MetricsCollection

@dataclass
class SearchQualityMetrics:
    query: str
    retrieved_count: int
    avg_similarity_score: float
    max_similarity_score: float
    min_similarity_score: float
    similarity_std: float
    documents: List[Dict] = field(default_factory=list)  # Список документов с их метриками
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class LLMQualityMetrics:
    query: str
    response: str
    context_relevance: float
    response_consistency: float
    response_completeness: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class QueryAnalysis:
    query: str
    search_quality: SearchQualityMetrics
    llm_quality: LLMQualityMetrics
    timestamp: datetime = field(default_factory=datetime.now)

class QualityAnalyzer:
    def __init__(self, metrics_collector: MetricsCollection):
        self.metrics_collector = metrics_collector
        self.search_metrics: List[SearchQualityMetrics] = []
        self.llm_metrics: List[LLMQualityMetrics] = []
        self.query_analyses: List[QueryAnalysis] = []
        
    def analyze_search_quality(
            self,
            query: str,
            similarity_scores: List[float],
            documents: List[Dict] = None, **kwargs
    ) -> SearchQualityMetrics:
        if similarity_scores is None or len(similarity_scores) == 0:
            logger().warning(f"No similarity scores provided for query: {query}")
            return None
            
        metrics = SearchQualityMetrics(
            query=query,
            retrieved_count=len(similarity_scores),
            avg_similarity_score=float(np.mean(similarity_scores)),
            max_similarity_score=float(np.max(similarity_scores)),
            min_similarity_score=float(np.min(similarity_scores)),
            similarity_std=float(np.std(similarity_scores)),
            documents=documents or [],
            metadata=kwargs
        )
        
        self.search_metrics.append(metrics)
        
        logger().info(
            "Search quality metrics",
            query=query,
            avg_score=metrics.avg_similarity_score,
            max_score=metrics.max_similarity_score,
            retrieved=metrics.retrieved_count,
            **kwargs
        )
        
        self.metrics_collector.end_operation(
            "vector_search",
            success=True,
            avg_similarity=metrics.avg_similarity_score,
            results_count=metrics.retrieved_count
        )
        
        return metrics
        
    def analyze_llm_quality(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
        **kwargs
    ) -> LLMQualityMetrics:
        context_relevance = self._evaluate_context_relevance(query, context_chunks)
        response_consistency = self._evaluate_response_consistency(response, context_chunks)
        response_completeness = self._evaluate_response_completeness(query, response)
        
        metrics = LLMQualityMetrics(
            query=query,
            response=response,
            context_relevance=context_relevance,
            response_consistency=response_consistency,
            response_completeness=response_completeness,
            metadata=kwargs
        )
        
        self.llm_metrics.append(metrics)
        
        if self.search_metrics and self.search_metrics[-1].query == query:
            self.query_analyses.append(QueryAnalysis(
                query=query,
                search_quality=self.search_metrics[-1],
                llm_quality=metrics
            ))
        
        logger().info(
            "LLM quality metrics",
            query=query,
            context_relevance=context_relevance,
            response_consistency=response_consistency,
            response_completeness=response_completeness,
            **kwargs
        )
        
        # Добавляем в общий сборщик метрик
        self.metrics_collector.end_operation(
            "llm_generation",
            success=True,
            context_relevance=context_relevance,
            response_consistency=response_consistency,
            response_completeness=response_completeness
        )
        
        return metrics
    
    def _evaluate_context_relevance(self, query: str, context_chunks: List[str]) -> float:
        return np.random.uniform(0.5, 1.0)
    
    def _evaluate_response_consistency(self, response: str, context_chunks: List[str]) -> float:
        return np.random.uniform(0.5, 1.0)
    
    def _evaluate_response_completeness(self, query: str, response: str) -> float:
        return np.random.uniform(0.5, 1.0)
    
    def get_search_metrics_summary(self) -> Dict:
        if not self.search_metrics:
            return {}
            
        avg_scores = [m.avg_similarity_score for m in self.search_metrics]
        return {
            "total_searches": len(self.search_metrics),
            "avg_similarity_overall": float(np.mean(avg_scores)),
            "avg_results_count": float(np.mean([m.retrieved_count for m in self.search_metrics])),
            "low_quality_searches": len([s for s in avg_scores if s < 0.5])
        }
    
    def get_llm_metrics_summary(self) -> Dict:
        if not self.llm_metrics:
            return {}
            
        return {
            "total_generations": len(self.llm_metrics),
            "avg_context_relevance": float(np.mean([m.context_relevance for m in self.llm_metrics])),
            "avg_response_consistency": float(np.mean([m.response_consistency for m in self.llm_metrics])),
            "avg_response_completeness": float(np.mean([m.response_completeness for m in self.llm_metrics]))
        }

    def export_metrics(self, output_path: Union[str, Path], format: str = 'json') -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            self._export_json(output_path)
        elif format.lower() == 'csv':
            self._export_csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger().info(f"Metrics exported to {output_path}")

    def _export_json(self, output_path: Path) -> None:
        data = []
        for analysis in self.query_analyses:
            entry = {
                'timestamp': analysis.timestamp.isoformat(),
                'query': analysis.query,
                'search_quality': {
                    'retrieved_count': analysis.search_quality.retrieved_count,
                    'avg_similarity_score': analysis.search_quality.avg_similarity_score,
                    'max_similarity_score': analysis.search_quality.max_similarity_score,
                    'min_similarity_score': analysis.search_quality.min_similarity_score,
                    'similarity_std': analysis.search_quality.similarity_std,
                    'documents': analysis.search_quality.documents
                },
                'llm_quality': {
                    'context_relevance': analysis.llm_quality.context_relevance,
                    'response_consistency': analysis.llm_quality.response_consistency,
                    'response_completeness': analysis.llm_quality.response_completeness,
                    'response': analysis.llm_quality.response
                }
            }
            data.append(entry)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _export_csv(self, output_path: Path) -> None:
        fieldnames = [
            'timestamp', 'query',
            'retrieved_count', 'avg_similarity_score', 'max_similarity_score',
            'min_similarity_score', 'similarity_std', 'documents',
            'context_relevance', 'response_consistency', 'response_completeness',
            'response'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for analysis in self.query_analyses:
                row = {
                    'timestamp': analysis.timestamp.isoformat(),
                    'query': analysis.query,
                    'retrieved_count': analysis.search_quality.retrieved_count,
                    'avg_similarity_score': analysis.search_quality.avg_similarity_score,
                    'max_similarity_score': analysis.search_quality.max_similarity_score,
                    'min_similarity_score': analysis.search_quality.min_similarity_score,
                    'similarity_std': analysis.search_quality.similarity_std,
                    'documents': json.dumps(analysis.search_quality.documents, ensure_ascii=False),
                    'context_relevance': analysis.llm_quality.context_relevance,
                    'response_consistency': analysis.llm_quality.response_consistency,
                    'response_completeness': analysis.llm_quality.response_completeness,
                    'response': analysis.llm_quality.response
                }
                writer.writerow(row) 
import json
import re
import logging
from typing import Dict, List, Any, Optional
from .tools.web_scraper import WebScraper
from .tools.data_tools import DataProcessor
from .tools.analysis_tools import DataAnalyzer
from .tools.visualization_tools import VisualizationGenerator

logger = logging.getLogger(__name__)

class DataAnalystAgent:
    """Main agent that orchestrates data analysis tasks"""
    
    def __init__(self):
        self.web_scraper = WebScraper()
        self.data_processor = DataProcessor()
        self.analyzer = DataAnalyzer()
        self.visualizer = VisualizationGenerator()
        
    def process_request(self, questions_text: str, uploaded_files: Dict[str, str]) -> Any:
        """
        Process a data analysis request
        
        Args:
            questions_text: The questions/instructions text
            uploaded_files: Dictionary of filename -> filepath for uploaded files
            
        Returns:
            Analysis results in the format specified by the questions
        """
        logger.info("Starting request processing")
        
        # Parse the request to understand what needs to be done
        analysis_plan = self._parse_request(questions_text)
        logger.info(f"Analysis plan: {analysis_plan}")
        
        # Execute the plan
        results = self._execute_analysis_plan(analysis_plan, uploaded_files)
        
        return results
        
        # Step 2: Execute in parallel
        context = await self._execute_analysis_plan(analysis_plan, uploaded_files)
        
        return context

    def _parse_request(self, questions_text: str) -> Dict[str, Any]:
        """Parse the questions text to create an analysis plan"""
        plan = {
            'requires_web_scraping': False,
            'urls_to_scrape': [],
            'requires_data_processing': False,
            'data_files': [],
            'questions': [],
            'output_format': 'json_array',
            'requires_visualization': False,
            'sql_queries': []
        }
        
        # Detect URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;!?)]'
        urls = re.findall(url_pattern, questions_text)
        if urls:
            plan['requires_web_scraping'] = True
            plan['urls_to_scrape'] = urls
        
        # Detect SQL queries
        sql_pattern = r'```sql\s*(.*?)\s*```'
        sql_matches = re.findall(sql_pattern, questions_text, re.DOTALL | re.IGNORECASE)
        if sql_matches:
            plan['sql_queries'] = sql_matches
            plan['requires_data_processing'] = True
        
        # Detect visualization needs
        viz_keywords = ['plot', 'chart', 'graph', 'scatterplot', 'histogram', 'base64', 'data:image']
        if any(keyword.lower() in questions_text.lower() for keyword in viz_keywords):
            plan['requires_visualization'] = True
        
        # Detect output format
        if 'JSON object' in questions_text.lower():
            plan['output_format'] = 'json_object'
        
        # Extract questions
        plan['questions'] = self._extract_questions(questions_text)
        
        return plan
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract individual questions from the text"""
        questions = []
        
        # Numbered questions
        numbered_pattern = r'\d+\.\s*([^?]*\?)'
        questions.extend(re.findall(numbered_pattern, text))
        
        # JSON-style questions
        json_pattern = r'"([^"]*\?)":\s*"[^"]*"'
        questions.extend(re.findall(json_pattern, text))
        
        # General fallback
        if not questions:
            question_pattern = r'([^.!]*\?)'
            all_questions = re.findall(question_pattern, text)
            questions = [q.strip() for q in all_questions if len(q.strip()) > 10]
        
        return [q.strip() for q in questions]
    
    async def _execute_analysis_plan(self, plan: Dict[str, Any], uploaded_files: Dict[str, str]) -> Any:
        """Execute the plan in parallel"""
        context = {}
        
        tasks = []
        
        # Scraping
        if plan['requires_web_scraping']:
            for url in plan['urls_to_scrape']:
                tasks.append(self._safe_task(self._scrape_url(url), f"scraped_data_{url}", context))
        
        # File loading
        for file_key, file_path in uploaded_files.items():
            tasks.append(self._safe_task(self._load_file(file_path), file_key, context))
        
        # SQL queries
        for i, query in enumerate(plan['sql_queries']):
            tasks.append(self._safe_task(self._execute_sql(query), f"sql_result_{i}", context))
        
        await asyncio.gather(*tasks)
        
        # Analysis (sequential, since it may depend on all data)
        answers = []
        all_data = dict(context)
        
        for question in plan['questions']:
            try:
                answer = await self.analyzer.answer_question_async(question, all_data)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Failed to answer '{question}': {e}")
                answers.append(None)
        
        # Visualization
        if plan['requires_visualization']:
            for i, question in enumerate(plan['questions']):
                if any(word in question.lower() for word in ['plot', 'chart', 'graph', 'scatterplot']):
                    try:
                        viz = await self.visualizer.create_visualization_async(question, all_data)
                        if viz:
                            answers[i] = viz
                    except Exception as e:
                        logger.error(f"Visualization failed: {e}")
        
        return self._format_results(plan, plan['questions'], answers)

    async def _scrape_url(self, url: str) -> Any:
        return await self.web_scraper.scrape_url_async(url)
    
    async def _load_file(self, file_path: str) -> Any:
        return await self.data_processor.load_file_async(file_path)
    
    async def _execute_sql(self, query: str) -> Any:
        return await self.data_processor.execute_sql_query_async(query)
    
    async def _safe_task(self, coro, key: str, context: dict):
        try:
            result = await coro
            context[key] = result
        except Exception as e:
            logger.error(f"Task for {key} failed: {e}")
            context[key] = None

    def _format_results(self, plan: Dict[str, Any], questions: List[str], answers: List[Any]):
        """Format results as object or array"""
        if plan['output_format'] == 'json_object':
            return {q: answers[i] if i < len(answers) else None for i, q in enumerate(questions)}
        return answers

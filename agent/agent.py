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
        
        # Check for web scraping requirements (only if explicitly asking to scrape)
        scrape_keywords = ['scrape', 'crawl', 'extract from', 'get data from']
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;!?)]'
        urls = re.findall(url_pattern, questions_text)
        if urls and any(keyword in questions_text.lower() for keyword in scrape_keywords):
            plan['requires_web_scraping'] = True
            plan['urls_to_scrape'] = urls
        
        # Check for SQL queries
        sql_pattern = r'```sql\s*(.*?)\s*```'
        sql_matches = re.findall(sql_pattern, questions_text, re.DOTALL | re.IGNORECASE)
        if sql_matches:
            plan['sql_queries'] = sql_matches
            plan['requires_data_processing'] = True
        
        # Check for visualization requirements
        viz_keywords = ['plot', 'chart', 'graph', 'scatterplot', 'histogram', 'base64', 'data:image']
        if any(keyword.lower() in questions_text.lower() for keyword in viz_keywords):
            plan['requires_visualization'] = True
        
        # Determine output format
        if 'JSON array' in questions_text or 'json array' in questions_text:
            plan['output_format'] = 'json_array'
        elif 'JSON object' in questions_text or 'json object' in questions_text:
            plan['output_format'] = 'json_object'
        
        # Extract individual questions
        questions = self._extract_questions(questions_text)
        logger.info(f"Extracted {len(questions)} questions: {[q[:50]+'...' for q in questions]}")
        plan['questions'] = questions
        
        return plan
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract individual questions from the text.
        Prioritizes JSON format questions over other patterns.
        """
        questions: List[str] = []
        seen = set()
        
        # First, try to extract from JSON block containing questions
        # Look for the JSON block that comes after "Answer the following questions"
        questions_section_match = re.search(r'Answer the following questions.*?```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        
        if questions_section_match:
            json_content = questions_section_match.group(1)
            # Extract questions from JSON object format
            json_question_pattern = r'"([^"]+)"\s*:\s*"[^"]*"'
            for q in re.findall(json_question_pattern, json_content):
                q = q.strip()
                # Filter for actual questions
                if (len(q) > 15 and q not in seen and 
                    any(word in q.lower() for word in ['what', 'which', 'how', 'who', 'when', 'where', 'plot', 'show', 'calculate', 'find', 'create']) and
                    'data:image/' not in q):
                    questions.append(q)
                    seen.add(q)
        
        # If no JSON questions found, try other methods
        if not questions:
            # 1) Numbered lines
            for line in text.splitlines():
                m = re.match(r"\s*(\d+)\.\s*(.+)", line)
                if m:
                    q = m.group(2).strip()
                    if len(q) > 10 and q not in seen:
                        questions.append(q)
                        seen.add(q)
            
            # 2) Questions ending with '?'
            question_pattern = r'([A-Z][^.!?]*\?)'
            for q in re.findall(question_pattern, text):
                q = q.strip()
                if len(q) > 20 and q not in seen:
                    questions.append(q)
                    seen.add(q)
        
        return questions
    
    def _execute_analysis_plan(self, plan: Dict[str, Any], uploaded_files: Dict[str, str]) -> Any:
        """Execute the analysis plan and return results"""
        context = {}
        
        # Step 1: Web scraping if needed
        if plan['requires_web_scraping']:
            logger.info("Performing web scraping")
            for url in plan['urls_to_scrape']:
                try:
                    scraped_data = self.web_scraper.scrape_url(url)
                    context[f'scraped_data_{url}'] = scraped_data
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    context[f'scraped_data_{url}'] = None
        
        # Step 2: Process data files
        processed_data = {}
        for file_key, file_path in uploaded_files.items():
            try:
                data = self.data_processor.load_file(file_path)
                processed_data[file_key] = data
                logger.info(f"Loaded data from {file_key}: {type(data)}")
            except Exception as e:
                logger.error(f"Failed to load {file_key}: {e}")
                processed_data[file_key] = None
        
        # Step 3: Execute SQL queries if needed
        if plan['sql_queries']:
            for i, query in enumerate(plan['sql_queries']):
                try:
                    result = self.data_processor.execute_sql_query(query)
                    context[f'sql_result_{i}'] = result
                    logger.info(f"Executed SQL query {i}")
                except Exception as e:
                    logger.error(f"Failed to execute SQL query {i}: {e}")
                    context[f'sql_result_{i}'] = None
        
        # Step 4: Analyze data and answer questions
        answers = []
        
        # Combine all available data
        all_data = {**context, **processed_data}
        
        for question in plan['questions']:
            try:
                answer = self.analyzer.answer_question(question, all_data)
                answers.append(answer)
                logger.info(f"Answered question: {question[:50]}...")
            except Exception as e:
                logger.error(f"Failed to answer question '{question}': {e}")
                answers.append(None)
        
        # Visualization handling
        viz_added = False
        if plan['requires_visualization']:
            try:
                for i, question in enumerate(plan['questions']):
                    if any(word in question.lower() for word in ['plot', 'chart', 'graph', 'scatterplot']):
                        viz = self.visualizer.create_visualization(question, all_data)
                        if viz:
                            if i < len(answers):
                                answers[i] = viz
                            else:
                                answers.append(viz)
                            viz_added = True
                            break
            except Exception as e:
                logger.error(f"Failed to generate visualization: {e}")
        
        # If visualization was requested but not injected, append it as the last element
        if plan['requires_visualization'] and not viz_added:
            try:
                # Use the first visualization-like instruction or a default scatterplot of Rank vs Peak
                viz_question = next((q for q in plan['questions'] if any(w in q.lower() for w in ['plot','scatterplot','chart','graph'])), 'Draw a scatterplot of Rank and Peak with a red dotted regression line.')
                viz = self.visualizer.create_visualization(viz_question, all_data)
                answers.append(viz)
            except Exception as e:
                logger.error(f"Fallback visualization failed: {e}")
                answers.append(None)
        
        # Format output according to plan
        if plan['output_format'] == 'json_object':
            # Create object with questions as keys
            result = {}
            for i, question in enumerate(plan['questions']):
                if i < len(answers):
                    result[question] = answers[i]
            return result
        else:
            # Return as array
            return answers

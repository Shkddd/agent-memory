"""
Simple Agent that uses MemoryManager for context retrieval
"""
import logging
from typing import Dict, List

from memory_manager import MemoryManager, MemoryPriority

logger = logging.getLogger(__name__)

class MemoryAwareAgent:
    """
    Agent that leverages MemoryManager for: 
    - Contextual responses (working memory)
    - Knowledge retrieval (long-term memory)
    """
    
    def __init__(self, memory_manager: MemoryManager = None):
        """
        Initialize agent
        
        Args:
            memory_manager: MemoryManager instance
        """
        self.memory_manager = memory_manager or MemoryManager()
        logger.info("Initialized MemoryAwareAgent")
    
    def process_user_input(self, session_id: str, user_input: str) -> str:
        """
        Process user input and generate response
        
        Args: 
            session_id: Session identifier
            user_input: User message
            
        Returns:
            Agent response
        """
        try: 
            # Store user input in working memory
            self. memory_manager.add_interaction(
                session_id,
                "user",
                user_input,
                priority=MemoryPriority.MEDIUM
            )
            
            # Retrieve context (working + long-term)
            context = self.memory_manager.get_agent_context(
                session_id,
                query=user_input,
                include_long_term=True
            )
            
            # Generate response (placeholder - integrate with actual LLM)
            response = self._generate_response(context, user_input)
            
            # Store agent response in working memory
            self.memory_manager.add_interaction(
                session_id,
                "agent",
                response,
                priority=MemoryPriority. MEDIUM
            )
            
            return response
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "Error processing your request."
    
    def _generate_response(self, context: str, user_input: str) -> str:
        """
        Generate response (placeholder)
        In production, integrate with OpenAI, LLaMA, etc.
        """
        return f"[Agent] Understood your request:  '{user_input}'. Using context with {len(context)} chars."
    
    def add_knowledge(self, fact: str, user_id: str = None, 
                     tags: List[str] = None) -> int:
        """Add knowledge/fact to long-term memory"""
        return self.memory_manager.add_fact(fact, user_id, tags)
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        return self.memory_manager. get_stats()

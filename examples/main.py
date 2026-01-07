"""
Main entry point and basic usage example for Agent Memory System
"""
import logging
from src.memory_manager import MemoryManager, MemoryPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example demonstrating memory system usage"""
    
    logger.info("=== Agent Memory System Demo ===\n")
    
    # Initialize memory manager
    memory = MemoryManager()
    logger.info("✓ Initialized MemoryManager\n")
    
    # Session ID for this conversation
    session_id = "demo_session_001"
    
    # Example 1: Multi-turn conversation with working memory
    logger.info("--- Example 1: Multi-turn Conversation ---")
    conversations = [
        ("user", "你好，我叫张三，今年30岁"),
        ("agent", "很高兴认识您，张三。请问您对保险感兴趣吗？"),
        ("user", "是的，我想了解重疾险的产品"),
        ("agent", "根据您30岁的年龄，推荐保额50万以上、等待期90天的产品"),
        ("user", "月保费大概多少？"),
        ("agent", "月保费大概150-200元，具体取决于保额和等待期"),
    ]
    
    for role, content in conversations:
        success = memory.add_interaction(
            session_id,
            role,
            content,
            priority=MemoryPriority. MEDIUM
        )
        print(f"{role.upper()}: {content}")
        if success:
            logger.debug(f"✓ Stored {role} message")
    
    print("\n")
    
    # Example 2: Add important facts to long-term memory
    logger.info("--- Example 2: Add Knowledge to Long-term Memory ---")
    facts = [
        ("30岁男性重疾险优先选保额50万以上、等待期90天的产品", ["insurance", "30male"]),
        ("非标体客户（甲状腺结节）推荐核保宽松的线上重疾险", ["insurance", "underwriting"]),
        ("用户偏好低保费、高杠杆的消费型保险", ["insurance", "preference"]),
    ]
    
    for fact_text, tags in facts:
        memory_id = memory.add_fact(
            fact_text,
            user_id="user_001",
            tags=tags,
            priority=MemoryPriority.HIGH
        )
        logger.info(f"✓ Added fact (ID: {memory_id}): {fact_text[: 50]}...")
    
    print("\n")
    
    # Example 3: Retrieve context for agent
    logger.info("--- Example 3: Retrieve Agent Context ---")
    context = memory.get_agent_context(
        session_id,
        query="用户有什么特点？推荐什么保险？",
        include_long_term=True
    )
    
    logger.info("Agent Context:")
    print("-" * 60)
    print(context)
    print("-" * 60)
    
    print("\n")
    
    # Example 4: Search similar memories
    logger.info("--- Example 4: Search Similar Memories ---")
    query = "30岁男性买什么重疾险好？"
    similar = memory.long_term_memory.search_similar(query, top_k=2)
    
    logger.info(f"Searching for: '{query}'")
    for result in similar:
        similarity = result["similarity"]
        memory_text = result["memory"]["text"]
        logger.info(f"  相似度: {similarity:.2%} | {memory_text[:50]}...")
    
    print("\n")
    
    # Example 5: Get memory statistics
    logger.info("--- Example 5: Memory Statistics ---")
    stats = memory.get_stats()
    
    logger.info(f"Long-term Memory Stats:")
    for key, value in stats. get("long_term", {}).items():
        logger.info(f"  {key}: {value}")
    
    print("\n")
    
    # Example 6: Save memories to disk
    logger.info("--- Example 6: Save Memories to Disk ---")
    save_result = memory.save_memories()
    if save_result: 
        logger.info("✓ Memories saved to disk")
    else:
        logger.warning("⚠ Could not save memories (Redis might not be running)")
    
    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

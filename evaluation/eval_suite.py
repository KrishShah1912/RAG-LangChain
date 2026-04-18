import os
import asyncio
from dotenv import load_dotenv
from backend.core.agents.graph import app
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class EvalResult(BaseModel):
    score: int = Field(description="Score from 1 to 5, where 5 is perfectly accurate.")
    reasoning: str = Field(description="Brief explanation of the score.")

test_cases = [
    {
        "question": "Who created LangChain?",
        "expected": "Harrison Chase"
    },
    {
        "question": "What is the latest version of langchain-core?",
        "expected": "1.2.31" 
    },
    {
        "question": "Does LangChain support JavaScript?",
        "expected": "Yes, LangChain.js exists."
    }
]

async def run_evaluation():
    print("🧪 Starting Day 12 Evaluation Suite...\n" + "="*40)
    
    scored_results = []
    
    for case in test_cases:
        print(f"❓ Testing: {case['question']}")
        
        config = {"configurable": {"thread_id": "eval_test"}}
        result = app.invoke({"messages": [("user", case['question'])]}, config=config)
        agent_answer = result["messages"][-1].content
        
        judge_prompt = f"""
        System: You are an unbiased judge grading a RAG agent.
        User Question: {case['question']}
        Expected Answer: {case['expected']}
        Agent Answer: {agent_answer}
        
        Grade the Agent Answer based on how well it matches the Expected Answer.
        Give a score from 1-5 and a short reason.
        """
        
        grade = judge_llm.with_structured_output(EvalResult).invoke(judge_prompt)
        scored_results.append(grade.score)
        
        print(f"⭐ Score: {grade.score}/5")
        print(f"📝 Reason: {grade.reasoning}\n" + "-"*40)

    avg_score = sum(scored_results) / len(scored_results)
    print(f"🏁 FINAL EVALUATION SCORE: {avg_score:.2f} / 5.0")
    
    if avg_score >= 3.5:
        print("READY FOR DEPLOYMENT!")
    else:
        print("AGENT NEEDS TUNING BEFORE DEPLOYMENT.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
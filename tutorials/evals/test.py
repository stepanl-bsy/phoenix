# %% [markdown]
# ## QA Evaluation with Arize Phoenix and Gemini 2.5 Pro - LangChain Schema Version

# %%
# Install required packages
# %pip install pdfplumber pandas arize-phoenix openinference-instrumentation-langchain
# %pip install langchain-google-vertexai google-cloud-aiplatform

# %%
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import phoenix as px
from phoenix.otel import register
from phoenix.evals import GeminiModel, run_evals
from phoenix.evals import (
    QA_PROMPT_TEMPLATE,
    QA_PROMPT_RAILS_MAP,
    HUMAN_VS_AI_PROMPT_RAILS_MAP,
    HUMAN_VS_AI_PROMPT_TEMPLATE,
    llm_classify,
)
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation import using_attributes, using_tags
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import pdfplumber
import re
from google.cloud.aiplatform import init as init_vertexai

# %%
# Initialize Vertex AI
PROJECT_ID = "seequent-labs-dev"
LOCATION = "us-central1"
init_vertexai(project=PROJECT_ID, location=LOCATION)

# %%
# Configure Phoenix
project_name = "alex-qa-coscientist-gemini"

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
os.environ["PHOENIX_PROJECT_NAME"] = project_name

# Configure the Phoenix tracer
tracer_provider = register(
    project_name=project_name,
    auto_instrument=True
)

# Import the automatic instrumentor from OpenInference
from openinference.instrumentation.langchain import LangChainInstrumentor

# Finish automatic instrumentation
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)

# %%
# Initialize ChatVertexAI client
client = ChatVertexAI(
    model_name="gemini-2.5-pro",
    temperature=0.0,
)

# %%
# Initialize Phoenix evaluation model (for initial attempt)
eval_model = GeminiModel(
    model="gemini-2.5-flash",
    project=PROJECT_ID,
    location=LOCATION
)

# %%
# Initialize ChatVertexAI evaluation model for schema-based parsing
eval_client = ChatVertexAI(
    model_name="gemini-2.5-flash",
    temperature=0.0,
)

# %%
# Define LangChain Schema for Evaluation Response
class EvaluationResponse(BaseModel):
    """Schema for evaluation response with correct/incorrect classification and explanation."""
    
    correctness: str = Field(
        description="Either 'CORRECT' if the AI answer matches the correct_answer answer, or 'INCORRECT' if it doesn't match or is wrong"
    )
    explanation: str = Field(
        description="Brief explanation of why the AI answer is correct or incorrect compared to the correct_answer answer"
    )

# %%
# PDF parsing function
def parse_qa_pdf_to_dataframe_separate_regex(pdf_path: str) -> pd.DataFrame:
    """
    Parses a PDF file to extract questions and answers using separate regexes and returns them in a pandas DataFrame.
    """
    text = ''
    
    # 1. Open the PDF and extract text from each page
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return pd.DataFrame()

    # 2. Define separate regex patterns for questions and answers
    question_pattern = r'--Question:\s*(.*?)(?=(--Answer:|$))'
    answer_pattern = r'--Answer:\s*(.*?)(?=(--Question:|$))'

    # 3. Find all questions and answers
    questions = re.findall(question_pattern, text, re.DOTALL)
    answers = re.findall(answer_pattern, text, re.DOTALL)

    # 4. Clean up the extracted text
    questions_clean = [q[0].strip() for q in questions]
    answers_clean = [a[0].strip() for a in answers]

    # 5. Pair questions and answers (assume they are in order)
    min_len = min(len(questions_clean), len(answers_clean))
    qa_pairs = list(zip(questions_clean[:min_len], answers_clean[:min_len]))

    # 6. Create a pandas DataFrame
    df = pd.DataFrame(qa_pairs, columns=['Question', 'Answer'])
    return df

# %%
# Define paths
parent_path = Path().cwd().parent.parent
qa_data_path = parent_path / "data" / "qa"
alex_qa_path = qa_data_path / "Q and A for AI evaluation.pdf"

# %%
def parse_qa_data(pdf_path: Path) -> pd.DataFrame:
    """Parse QA data from PDF file."""
    print(f"Parsing QA data from: {pdf_path}")
    qa_df = parse_qa_pdf_to_dataframe_separate_regex(pdf_path)
    print(f"Successfully parsed {len(qa_df)} QA pairs")
    return qa_df

# %%
def convert_to_phoenix_format(qa_df: pd.DataFrame) -> pd.DataFrame:
    """Convert parsed QA data to Phoenix dataset format."""
    phoenix_data = []
    
    for idx, row in qa_df.iterrows():
        # Handle the correct column names from your dataframe
        question = row['Question'] if 'Question' in row else row.get('question', '')
        answer = row['Answer'] if 'Answer' in row else row.get('answer', '')
        
        # Skip rows with empty answers
        if pd.isna(answer) or str(answer).strip() == '':
            print(f"Skipping row {idx} due to empty answer")
            continue
            
        phoenix_data.append({
            'question': question,
            'correct_answer': answer,
            'id': f"qa_{idx}"
        })
    
    phoenix_df = pd.DataFrame(phoenix_data)
    print(f"Converted {len(phoenix_df)} QA pairs to Phoenix format")
    return phoenix_df

# %%
def upload_to_phoenix(df: pd.DataFrame, dataset_name: str) -> Any:
    """Upload dataset to Phoenix using the correct API."""
    print(f"Uploading dataset '{dataset_name}' to Phoenix...")
    
    # Initialize Phoenix client
    phoenix_client = px.Client()
    
    # Upload dataset using the correct method
    phoenix_dataset = phoenix_client.upload_dataset(
        dataframe=df,
        dataset_name=dataset_name,
        input_keys=["question"],
        output_keys=["correct_answer"],
        metadata_keys=["id"]
    )
    
    print(f"Successfully uploaded dataset with {len(df)} records")
    return phoenix_dataset

# %%
def create_agent_responses_with_tracing(phoenix_df: pd.DataFrame, client: ChatVertexAI) -> pd.DataFrame:
    """Generate agent responses for evaluation with Phoenix tracing."""
    responses = []
    
    print("Generating agent responses with Phoenix tracing...")
    
    for idx, row in phoenix_df.iterrows():
        question = row['question']
        
        system_instruction = """You are a helpful assistant providing clear, concise answers about geological and mining topics.

RESPONSE FORMAT REQUIREMENTS:
- Keep responses under 300 words to avoid token limits
- Use simple paragraph format without complex markdown
- Avoid tables, code blocks, and special formatting
- Use plain text with basic bullet points (- or *) only
- No headers (#), no bold (**text**), no italics (*text*)
- Replace tables with simple lists or comma-separated values
- Use numbered lists (1., 2., 3.) for step-by-step instructions
- Avoid special characters like pipes (|), backticks (`), or complex symbols
- Write in clear, direct sentences
- Focus on the most essential information to answer the question

CONTENT GUIDELINES:
- Provide accurate, technical information
- Be concise but complete
- Use simple language while maintaining technical accuracy
- If examples are needed, use simple text descriptions instead of formatted tables
- Break complex concepts into simple, digestible points"""

        # Create message in LangChain format
        messages = [SystemMessage(content=system_instruction), HumanMessage(content=question)]
        
        # Use tracing attributes
        with using_attributes(
            # question=question
        ):
            try:
                # Get response from Gemini with tracing
                response = client.invoke(messages)
                agent_answer = response.content
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                agent_answer = "Error: Could not generate response"
        
        responses.append({
            'question': row['question'],
            'correct_answer': row['correct_answer'],
            'ai_generated_answer': agent_answer,
            'metadata': {'id': row['id']}
        })
        
        print(f"Processed question {idx + 1}/{len(phoenix_df)}")
    
    return pd.DataFrame(responses)
# %%
def evaluate_with_langchain_schema(
    eval_df: pd.DataFrame,
    eval_client: ChatVertexAI,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Evaluate using ChatVertexAI with structured ai_generated_answer binding.
    
    Args:
        eval_df (pd.DataFrame): DataFrame with 'question', 'correct_answer', 'ai_generated_answer' columns
        eval_client (ChatVertexAI): ChatVertexAI client for evaluation
        max_retries (int): Maximum number of retry attempts for failed evaluations
    
    Returns:
        pd.DataFrame: DataFrame with evaluation results
    """
        
    # Bind the schema to the model using with_structured_ai_generated_answer
    structured_eval_client = eval_client.with_structured_output(EvaluationResponse)

    # Create evaluation prompt template
    evaluation_prompt = """You are an expert evaluator. Compare the AI-generated answer with the correct_answer answer and determine if the AI answer is correct.

    Question: {question}

    correct_answer Answer (Correct): {correct_answer}

    AI Generated Answer: {ai_generated_answer}

    Instructions:
    - Evaluate if the AI answer correctly addresses the question compared to the correct_answer answer
    - Consider semantic meaning, not just exact word matching
    - The AI answer should contain the key concepts and information from the correct_answer answer
    - Minor differences in wording are acceptable if the core meaning is preserved
    - If the AI answer is substantially wrong, incomplete, or contradicts the correct_answer, mark as INCORRECT

    Provide your evaluation with:
    - correctness: Either "CORRECT" or "INCORRECT"
    - explanation: Brief explanation of your decision"""

    results = []

    print("Starting structured ai_generated_answer evaluation...")

    for idx, row in eval_df.iterrows():
            # Extract relevant fields from the row

        
        # Format the prompt
        formatted_prompt = evaluation_prompt.format(
            question=row['question'],
            correct_answer=row['correct_answer'],
            ai_generated_answer=row['ai_generated_answer']
        )
        
        # Try evaluation with retries
        evaluation_result = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Use tracing attributes
                with using_attributes(
                    metadata = {
                        "description": "Evaluate with langchain structured ai_generated_answer fallback",

                    }
                ):
                    # Create messages
                    messages = [HumanMessage(content=formatted_prompt)]
                    
                    # Get structured response from evaluation model
                    evaluation_result = structured_eval_client.invoke(messages)
                    
                    print(f"✅ Successfully evaluated question {idx + 1}/{len(eval_df)} on attempt {attempt + 1}")
                    break
                    
            except Exception as e:
                last_error = e
                print(f"⚠️ Attempt {attempt + 1} failed for question {idx + 1}: {e}")
                
                if attempt == max_retries - 1:
                    print(f"❌ All {max_retries} attempts failed for question {idx + 1}")
                    # Create error result
                    evaluation_result = EvaluationResponse(
                        correctness="ERROR",
                        explanation=f"Evaluation failed after {max_retries} attempts: {str(last_error)}"
                    )
        
        # Store result
        results.append({
            'question': row['question'],
            'correct_answer': row['correct_answer'],
            'ai_generated_answer': row['ai_generated_answer'],
            'label': evaluation_result.correctness.lower(),
            'explanation': evaluation_result.explanation,
            'metadata': row.get('metadata', {})
        })
        
        print(f"Processed question {idx + 1}/{len(eval_df)}: {evaluation_result.correctness}")

    return pd.DataFrame(results)

# %%
def hybrid_evaluation_with_fallback(
    eval_df: pd.DataFrame,
    eval_model: GeminiModel,
    eval_client: ChatVertexAI
) -> pd.DataFrame:
    """
    Hybrid evaluation: try Phoenix first, then fall back to LangChain schema for unparsable results.
    """
    rails = list(HUMAN_VS_AI_PROMPT_RAILS_MAP.values())
    
    try:
        print("Attempting Phoenix evaluation first...")
        
        # First attempt with Phoenix
        phoenix_results = llm_classify(
            dataframe=eval_df,
            template=HUMAN_VS_AI_PROMPT_TEMPLATE,
            model=eval_model,
            rails=rails,
            provide_explanation=True,
            concurrency=5,
            max_retries=3,
            verbose=True
        )
        
        # Check for unparsable results
        unparsable_mask = phoenix_results['label'] == 'NOT_PARSABLE'
        unparsable_count = unparsable_mask.sum()
        
        print(f"Phoenix evaluation completed: {len(phoenix_results) - unparsable_count}/{len(phoenix_results)} successful")
        
        if unparsable_count > 0:
            print(f"Found {unparsable_count} unparsable results. Using LangChain schema evaluation for retry...")
            
            # Get unparsable rows
            unparsable_df = eval_df[unparsable_mask].copy()
            
            # Evaluate unparsable rows with LangChain schema
            schema_results = evaluate_with_langchain_schema(unparsable_df, eval_client)
            
            # Merge results back
            final_results = phoenix_results.copy()
            
            # Update unparsable rows with schema results
            for i, (phoenix_idx, schema_row) in enumerate(zip(unparsable_df.index, schema_results.itertuples())):
                final_results.loc[phoenix_idx, 'label'] = schema_row.label
                final_results.loc[phoenix_idx, 'explanation'] = schema_row.explanation
            
            print(f"✅ Successfully resolved {unparsable_count} unparsable results using LangChain schema")
            
            return final_results
        else:
            print("✅ All Phoenix evaluations successful, no fallback needed")
            return phoenix_results
            
    except Exception as e:
        print(f"❌ Phoenix evaluation failed completely: {e}")
        print("Falling back to full LangChain schema evaluation...")
        
        # Complete fallback to LangChain schema
        try:
            schema_results = evaluate_with_langchain_schema(eval_df, eval_client)
            print("✅ LangChain schema evaluation completed successfully")
            return schema_results
        except Exception as schema_error:
            print(f"❌ LangChain schema evaluation also failed: {schema_error}")
            # Raise a comprehensive error
            raise Exception(f"Both Phoenix and LangChain schema evaluations failed. Phoenix error: {e}. Schema error: {schema_error}")


# # %%
# Main execution code
print("Starting QA evaluation pipeline with LangChain schema fallback...")

# Step 1: Parse QA data
qa_df = parse_qa_data(alex_qa_path)
print(f"Sample parsed data:\n{qa_df.head()}")

# Step 2: Convert to Phoenix format
phoenix_df = convert_to_phoenix_format(qa_df)

# Step 3: Upload to Phoenix
dataset_name = f"{project_name}_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
dataset = upload_to_phoenix(phoenix_df, dataset_name)

# Step 4: Generate agent responses with tracing
eval_df = create_agent_responses_with_tracing(phoenix_df, client)

# Step 5: Run hybrid evaluation with LangChain schema fallback
try:
    eval_results = hybrid_evaluation_with_fallback(eval_df, eval_model, eval_client)
except Exception as e:
    print(f"🚨 CRITICAL ERROR: Evaluation pipeline failed completely: {e}")
    raise Exception(f"QA Evaluation Pipeline Failure: {e}")



# %%
# Log evaluations to Phoenix
from phoenix.trace import SpanEvaluations

px.Client().log_evaluations(
    SpanEvaluations(
        dataframe=eval_results,
        eval_name="QA_LangChain_Schema_Evaluation",
    ),
)
